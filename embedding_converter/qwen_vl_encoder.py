import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Literal
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import logging
from tqdm import tqdm
from pathlib import Path

from .base_encoder import BaseEncoder, ModalityType, EncoderConfig
from .dimension_reducer import DimensionReducer

logger = logging.getLogger(__name__)

def process_vision_info(messages_batch):
    """处理视觉信息的简化版本"""
    image_inputs = []
    video_inputs = []
    
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
    
    return image_inputs, video_inputs

class QwenVLEncoder(BaseEncoder):
    """
    Qwen2.5-VL多模态编码器实现
    """
    
    def __init__(self,
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 torch_dtype: str = "auto",
                 attn_implementation: str = "sdpa",
                 **kwargs):
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        super().__init__(model_name, cache_dir, device, **kwargs)
    
    def _load_model(self, **kwargs) -> None:
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir, trust_remote_code=True
            )
            
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True
            }
            
            if self.torch_dtype != "auto":
                model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
            elif self.device == "cuda":
                logger.info("在CUDA设备上，自动设置torch_dtype=torch.float16以启用FlashAttention")
                model_kwargs["torch_dtype"] = torch.float16
            
            # 对于Qwen2.5-VL，使用兼容的注意力实现
            if self.attn_implementation and self.device == "cuda":
                if self.attn_implementation == "flash_attention_2":
                    logger.warning("检测到flash_attention_2，由于兼容性问题，自动切换到sdpa")
                    model_kwargs["attn_implementation"] = "sdpa"
                else:
                    model_kwargs["attn_implementation"] = self.attn_implementation
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, **model_kwargs
            )
            
            if self.device and (self.device != "cuda" or "device_map" not in model_kwargs):
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
    
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码纯文本，并能处理代表缺失值的空字符串。"""
        if not texts:
            return np.empty((0, self.get_native_embedding_dim()))

        final_embeddings = np.zeros((len(texts), self.get_native_embedding_dim()), dtype=np.float16)
        
        non_empty_texts, non_empty_indices = [], []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        if not non_empty_texts:
            return final_embeddings

        batch_size = kwargs.get('batch_size', 8)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(non_empty_texts), batch_size), desc="编码文本"):
                batch_texts = non_empty_texts[i:i + batch_size]
                
                messages_batch = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in batch_texts]
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
                
                inputs = self.processor(text=texts_formatted, padding=True, return_tensors="pt").to(self.device)
                
                text_model = self.model.model
                outputs = text_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                batch_embeddings = self._extract_embeddings_from_hidden_states(hidden_states, inputs.attention_mask)
                
                original_indices_in_full_list = [non_empty_indices[j] for j in range(i, i + len(batch_texts))]
                final_embeddings[original_indices_in_full_list, :] = batch_embeddings

        return final_embeddings
    
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """编码纯图像，并能处理代表缺失值的空字符串路径。"""
        if not image_paths:
            return np.empty((0, self.get_native_embedding_dim()))

        final_embeddings = np.zeros((len(image_paths), self.get_native_embedding_dim()), dtype=np.float16)

        non_empty_paths, non_empty_indices = [], []
        for i, path in enumerate(image_paths):
            # 检查路径是否有效
            if path and str(path).strip() and Path(path).exists():
                non_empty_paths.append(path)
                non_empty_indices.append(i)

        if not non_empty_paths:
            return final_embeddings

        batch_size = kwargs.get('batch_size', 4)

        with torch.no_grad():
            for i in tqdm(range(0, len(non_empty_paths), batch_size), desc="编码图像"):
                batch_paths = non_empty_paths[i:i + batch_size]
                
                messages_batch = [[{"role": "user", "content": [{"type": "image", "image": f"file://{path}"}, {"type": "text", "text": "描述这张图片"}]}] for path in batch_paths]
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
                image_inputs, _ = process_vision_info(messages_batch)
                
                inputs = self.processor(text=texts_formatted, images=image_inputs, padding=True, return_tensors="pt").to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                batch_embeddings = self._extract_embeddings_from_hidden_states(hidden_states, inputs.attention_mask)
                
                original_indices_in_full_list = [non_empty_indices[j] for j in range(i, i + len(batch_paths))]
                final_embeddings[original_indices_in_full_list, :] = batch_embeddings

        return final_embeddings

    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        """编码多模态数据，并能处理缺失值。"""
        if len(texts) != len(image_paths):
            raise ValueError("文本和图像列表的长度必须相等")

        if not texts:
            return np.empty((0, self.get_native_embedding_dim()))

        final_embeddings = np.zeros((len(texts), self.get_native_embedding_dim()), dtype=np.float16)

        # 筛选出文本和图像都有效的对
        valid_indices = []
        valid_texts = []
        valid_image_paths = []

        for i, (text, path) in enumerate(zip(texts, image_paths)):
            if text and str(path).strip() and Path(path).exists():
                valid_indices.append(i)
                valid_texts.append(text)
                valid_image_paths.append(path)
        
        if not valid_texts:
            return final_embeddings

        batch_size = kwargs.get('batch_size', 4)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_texts), batch_size), desc="编码多模态"):
                batch_texts = valid_texts[i:i + len(valid_texts)]
                batch_paths = valid_image_paths[i:i + len(valid_texts)]
                
                messages_batch = []
                for text, path in zip(batch_texts, batch_paths):
                    messages_batch.append([{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{path}"},
                            {"type": "text", "text": text}
                        ]
                    }])
                
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
                image_inputs, _ = process_vision_info(messages_batch)
                
                inputs = self.processor(text=texts_formatted, images=image_inputs, padding=True, return_tensors="pt").to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                batch_embeddings = self._extract_embeddings_from_hidden_states(hidden_states, inputs.attention_mask)
                
                original_indices_in_full_list = [valid_indices[j] for j in range(i, i + len(batch_texts))]
                final_embeddings[original_indices_in_full_list, :] = batch_embeddings

        return final_embeddings

class QwenVLEncoderWithDimension(QwenVLEncoder):
    """带维度变换的Qwen2.5-VL编码器"""
    
    def __init__(self, target_dimension: int, reduction_method: Literal["linear", "pca"] = "linear", **kwargs):
        super().__init__(**kwargs)
        self.target_dimension = target_dimension
        self.dimension_reducer = DimensionReducer(
            input_dim=self.get_native_embedding_dim(),
            output_dim=target_dimension,
            method=reduction_method
        )
    
    def _apply_dimension_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        return self.dimension_reducer.transform(embeddings)

    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码纯文本，使用模型原生维度然后降维"""
        if not texts:
            return np.empty((0, self.target_dimension))

        # 临时重置get_native_embedding_dim以获取真实的模型维度
        model_native_dim = super().get_native_embedding_dim()
        
        final_embeddings = np.zeros((len(texts), model_native_dim), dtype=np.float16)
        
        non_empty_texts, non_empty_indices = [], []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        if not non_empty_texts:
            return self._apply_dimension_reduction(final_embeddings)

        batch_size = kwargs.get('batch_size', 8)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(non_empty_texts), batch_size), desc="编码文本"):
                batch_texts = non_empty_texts[i:i + batch_size]
                
                messages_batch = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in batch_texts]
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
                
                inputs = self.processor(text=texts_formatted, padding=True, return_tensors="pt").to(self.device)
                
                text_model = self.model.model
                outputs = text_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                batch_embeddings = self._extract_embeddings_from_hidden_states(hidden_states, inputs.attention_mask)
                
                original_indices_in_full_list = [non_empty_indices[j] for j in range(i, i + len(batch_texts))]
                final_embeddings[original_indices_in_full_list, :] = batch_embeddings

        return self._apply_dimension_reduction(final_embeddings)
    
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """编码纯图像，使用模型原生维度然后降维"""
        if not image_paths:
            return np.empty((0, self.target_dimension))

        # 获取真实的模型维度
        model_native_dim = super().get_native_embedding_dim()
        
        final_embeddings = np.zeros((len(image_paths), model_native_dim), dtype=np.float16)

        non_empty_paths, non_empty_indices = [], []
        for i, path in enumerate(image_paths):
            # 检查路径是否有效
            if path and str(path).strip() and Path(path).exists():
                non_empty_paths.append(path)
                non_empty_indices.append(i)

        if not non_empty_paths:
            return self._apply_dimension_reduction(final_embeddings)

        batch_size = kwargs.get('batch_size', 4)

        with torch.no_grad():
            for i in tqdm(range(0, len(non_empty_paths), batch_size), desc="编码图像"):
                batch_paths = non_empty_paths[i:i + batch_size]
                
                messages_batch = [[{"role": "user", "content": [{"type": "image", "image": f"file://{path}"}, {"type": "text", "text": "描述这张图片"}]}] for path in batch_paths]
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
                image_inputs, _ = process_vision_info(messages_batch)
                
                inputs = self.processor(text=texts_formatted, images=image_inputs, padding=True, return_tensors="pt").to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                batch_embeddings = self._extract_embeddings_from_hidden_states(hidden_states, inputs.attention_mask)
                
                original_indices_in_full_list = [non_empty_indices[j] for j in range(i, i + len(batch_paths))]
                final_embeddings[original_indices_in_full_list, :] = batch_embeddings

        return self._apply_dimension_reduction(final_embeddings)
    
    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        """编码多模态数据，使用模型原生维度然后降维"""
        if len(texts) != len(image_paths):
            raise ValueError("文本和图像列表的长度必须相等")

        if not texts:
            return np.empty((0, self.target_dimension))

        # 获取真实的模型维度
        model_native_dim = super().get_native_embedding_dim()
        
        final_embeddings = np.zeros((len(texts), model_native_dim), dtype=np.float16)

        # 筛选出文本和图像都有效的对
        valid_indices = []
        valid_texts = []
        valid_image_paths = []

        for i, (text, path) in enumerate(zip(texts, image_paths)):
            if text and str(path).strip() and Path(path).exists():
                valid_indices.append(i)
                valid_texts.append(text)
                valid_image_paths.append(path)
        
        if not valid_texts:
            return self._apply_dimension_reduction(final_embeddings)

        batch_size = kwargs.get('batch_size', 4)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_texts), batch_size), desc="编码多模态"):
                batch_texts = valid_texts[i:i + len(valid_texts)]
                batch_paths = valid_image_paths[i:i + len(valid_texts)]
                
                messages_batch = []
                for text, path in zip(batch_texts, batch_paths):
                    messages_batch.append([{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{path}"},
                            {"type": "text", "text": text}
                        ]
                    }])
                
                texts_formatted = [self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False) for msgs in messages_batch]
                image_inputs, _ = process_vision_info(messages_batch)
                
                inputs = self.processor(text=texts_formatted, images=image_inputs, padding=True, return_tensors="pt").to(self.device)
                
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                batch_embeddings = self._extract_embeddings_from_hidden_states(hidden_states, inputs.attention_mask)
                
                original_indices_in_full_list = [valid_indices[j] for j in range(i, i + len(batch_texts))]
                final_embeddings[original_indices_in_full_list, :] = batch_embeddings

        return self._apply_dimension_reduction(final_embeddings)
    
    def get_native_embedding_dim(self) -> int:
        # 对于维度变换编码器，返回目标维度
        return self.target_dimension
    
    def get_model_native_embedding_dim(self) -> int:
        # 返回模型真实的原生维度
        return super().get_native_embedding_dim()