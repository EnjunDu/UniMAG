import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Literal
from transformers import AutoModelForImageTextToText
from transformers import AutoProcessor
import logging
from tqdm import tqdm
from pathlib import Path

from ..base_encoder import BaseEncoder
from ..utils.dimension_reducer import DimensionReducer
from ..encoder_factory import EncoderFactory

logger = logging.getLogger(__name__)

def process_vision_info(messages_batch):
    """处理视觉信息的简化版本"""
    image_inputs = []
    for messages in messages_batch:
        for message in messages:
            if "content" in message:
                for content in message["content"]:
                    if content.get("type") == "image":
                        image_path = content.get("image", "").replace("file://", "")
                        if image_path:
                            try:
                                img = Image.open(image_path).convert('RGB')
                                image_inputs.append(img)
                            except Exception as e:
                                logger.warning(f"无法加载图像 {image_path}: {e}")
    return image_inputs, []

@EncoderFactory.register("qwen_vl")
class QwenVLEncoder(BaseEncoder):
    """
    Qwen2.5-VL多模态编码器实现。
    这个版本返回模型原生的嵌入维度。
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", cache_dir: Optional[str] = None, device: Optional[str] = None, torch_dtype: str = "auto", attn_implementation: str = "sdpa", **kwargs):
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        super().__init__(model_name, cache_dir, device, **kwargs)
    
    def _load_model(self, **kwargs) -> None:
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir, trust_remote_code=True)
            
            model_kwargs = {"cache_dir": self.cache_dir, "trust_remote_code": True}

            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            elif self.device and self.device.startswith("cuda"):
                model_kwargs["device_map"] = None
            else:
                model_kwargs["device_map"] = None

            if self.torch_dtype != "auto":
                model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
            elif self.device and self.device.startswith("cuda"):
                logger.info("在CUDA设备上，自动设置torch_dtype=torch.float16以启用FlashAttention")
                model_kwargs["torch_dtype"] = torch.float16
            
            if self.attn_implementation and self.device and self.device.startswith("cuda"):
                model_kwargs["attn_implementation"] = self.attn_implementation
            
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, **model_kwargs)
            
            if self.device and ("device_map" not in model_kwargs or model_kwargs.get("device_map") is None):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("模型加载完成")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_native_embedding_dim(self) -> int:
        return self.model.config.hidden_size
    
    def _extract_embeddings_from_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.device)
        masked_embeddings = hidden_states * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        mean_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        return mean_embeddings.detach().cpu().numpy()
    
    def encode_text(self, texts: List[str], output_feature_map: bool = False, **kwargs) -> np.ndarray:
        native_dim = self.model.config.hidden_size
        if not texts: return np.empty((0, native_dim))
        
        non_empty_texts, non_empty_indices = [], []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        if not non_empty_texts:
            # 如果需要特征图，返回一个空的、形状合适的数组
            if output_feature_map:
                # 注意：这里的形状可能需要根据模型的具体输出进行调整
                return np.empty((len(texts), 0, native_dim), dtype=np.float16)
            return np.zeros((len(texts), native_dim), dtype=np.float16)

        batch_size = kwargs.get('batch_size', 8)
        all_outputs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(non_empty_texts), batch_size), desc="编码文本"):
                batch_texts = non_empty_texts[i:i + batch_size]
                messages_batch = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in batch_texts]
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
                inputs = self.processor(text=texts_formatted, padding=True, return_tensors="pt").to(self.model.device)
                outputs = self.model.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
                
                if output_feature_map:
                    # 返回倒数第二层的隐藏状态作为特征图
                    # 注意：我们只保存有效文本的输出
                    all_outputs.append(outputs.hidden_states[-2].detach().cpu().numpy())
                else:
                    batch_embeddings = self._extract_embeddings_from_hidden_states(outputs.hidden_states[-1], inputs.attention_mask)
                    all_outputs.append(batch_embeddings)

        # 根据是否输出特征图，决定如何组合和返回结果
        if output_feature_map:
            if not all_outputs:
                return np.array([])
            return np.concatenate(all_outputs, axis=0)

        # 合并嵌入向量
        final_embeddings = np.zeros((len(texts), native_dim), dtype=np.float16)
        processed_count = 0
        for batch_embeddings in all_outputs:
            batch_size = batch_embeddings.shape[0]
            original_indices = non_empty_indices[processed_count : processed_count + batch_size]
            final_embeddings[original_indices, :] = batch_embeddings
            processed_count += batch_size
            
        return final_embeddings

    def encode_image(self, images: List[Union[str, Image.Image]], output_feature_map: bool = False, **kwargs) -> np.ndarray:
        native_dim = self.model.config.hidden_size
        if not images: return np.empty((0, native_dim))

        valid_images, valid_indices = [], []
        for i, item in enumerate(images):
            if isinstance(item, str):
                if item and item.strip() and Path(item).exists():
                    valid_images.append(Image.open(item).convert('RGB'))
                    valid_indices.append(i)
            elif isinstance(item, Image.Image):
                valid_images.append(item)
                valid_indices.append(i)

        if not valid_images:
            if output_feature_map:
                return np.empty((len(images), 0, native_dim), dtype=np.float16)
            return np.zeros((len(images), native_dim), dtype=np.float16)

        batch_size = kwargs.get('batch_size', 4)
        all_outputs = []

        with torch.no_grad():
            for i in tqdm(range(0, len(valid_images), batch_size), desc="编码图像"):
                batch_images = valid_images[i:i + batch_size]
                # 为每个图像创建一个符合Qwen-VL格式的虚拟文本提示
                dummy_texts = [f"Picture 1:<img>\nDescribe the image."] * len(batch_images)
                
                # 使用 apply_chat_template 来格式化输入
                messages_batch = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}] for _ in batch_images]
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]

                inputs = self.processor(text=texts_formatted, images=batch_images, padding=True, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                
                if output_feature_map:
                    all_outputs.append(outputs.hidden_states[-2].detach().cpu().numpy())
                else:
                    batch_embeddings = self._extract_embeddings_from_hidden_states(outputs.hidden_states[-1], inputs.attention_mask)
                    all_outputs.append(batch_embeddings)
        
        if output_feature_map:
            if not all_outputs:
                return np.array([])
            return np.concatenate(all_outputs, axis=0)

        final_embeddings = np.zeros((len(images), native_dim), dtype=np.float16)
        processed_count = 0
        for batch_embeddings in all_outputs:
            batch_size = batch_embeddings.shape[0]
            original_indices = valid_indices[processed_count : processed_count + batch_size]
            final_embeddings[original_indices, :] = batch_embeddings
            processed_count += batch_size

        return final_embeddings

    def encode_multimodal(self, texts: List[str], images: List[Union[str, Image.Image]], output_feature_map: bool = False, **kwargs) -> np.ndarray:
        native_dim = self.model.config.hidden_size
        if len(texts) != len(images): raise ValueError("文本和图像列表的长度必须相等")
        if not texts: return np.empty((0, native_dim))

        valid_indices, valid_texts, valid_images = [], [], []
        for i, (text, item) in enumerate(zip(texts, images)):
            if not text or not text.strip(): continue
            
            if isinstance(item, str):
                if item and item.strip() and Path(item).exists():
                    valid_images.append(Image.open(item).convert('RGB'))
                    valid_texts.append(text)
                    valid_indices.append(i)
            elif isinstance(item, Image.Image):
                valid_images.append(item)
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            if output_feature_map:
                return np.empty((len(texts), 0, native_dim), dtype=np.float16)
            return np.zeros((len(texts), native_dim), dtype=np.float16)

        batch_size = kwargs.get('batch_size', 4)
        all_outputs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_texts), batch_size), desc="编码多模态"):
                batch_texts = valid_texts[i:i + batch_size]
                batch_images = valid_images[i:i + batch_size]
                inputs = self.processor(text=batch_texts, images=batch_images, padding=True, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                
                if output_feature_map:
                    all_outputs.append(outputs.hidden_states[-2].detach().cpu().numpy())
                else:
                    batch_embeddings = self._extract_embeddings_from_hidden_states(outputs.hidden_states[-1], inputs.attention_mask)
                    all_outputs.append(batch_embeddings)

        if output_feature_map:
            if not all_outputs:
                return np.array([])
            return np.concatenate(all_outputs, axis=0)

        final_embeddings = np.zeros((len(texts), native_dim), dtype=np.float16)
        processed_count = 0
        for batch_embeddings in all_outputs:
            batch_size = batch_embeddings.shape[0]
            original_indices = valid_indices[processed_count : processed_count + batch_size]
            final_embeddings[original_indices, :] = batch_embeddings
            processed_count += batch_size
            
        return final_embeddings

@EncoderFactory.register("qwen_vl_with_dim")
class QwenVLEncoderWithDimension(QwenVLEncoder):
    """
    带维度变换的Qwen2.5-VL编码器。
    这个版本将原生维度的嵌入通过一个线性层，变换到指定的目标维度。
    """
    def __init__(self, target_dimension: int, reduction_method: Literal["linear", "pca"] = "linear", **kwargs):
        super().__init__(**kwargs)
        self.target_dimension = target_dimension
        model_native_dim = super().get_native_embedding_dim()
        self.dimension_reducer = DimensionReducer(input_dim=model_native_dim, output_dim=target_dimension, method=reduction_method)
    
    def _apply_dimension_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        return self.dimension_reducer.transform(embeddings)

    def encode_text(self, texts: List[str], output_feature_map: bool = False, **kwargs) -> np.ndarray:
        native_output = super().encode_text(texts, output_feature_map=output_feature_map, **kwargs)
        if output_feature_map:
            return native_output
        return self._apply_dimension_reduction(native_output)

    def encode_image(self, images: List[Union[str, Image.Image]], output_feature_map: bool = False, **kwargs) -> np.ndarray:
        native_output = super().encode_image(images, output_feature_map=output_feature_map, **kwargs)
        if output_feature_map:
            return native_output
        return self._apply_dimension_reduction(native_output)

    def encode_multimodal(self, texts: List[str], images: List[Union[str, Image.Image]], output_feature_map: bool = False, **kwargs) -> np.ndarray:
        native_output = super().encode_multimodal(texts, images, output_feature_map=output_feature_map, **kwargs)
        if output_feature_map:
            return native_output
        return self._apply_dimension_reduction(native_output)
    
    def get_native_embedding_dim(self) -> int:
        return self.target_dimension