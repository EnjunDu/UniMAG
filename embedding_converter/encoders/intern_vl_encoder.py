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

@EncoderFactory.register("intern_vl")
class InternVL3Encoder(BaseEncoder):
    """
    InternVL3-1B 多模态编码器实现。
    这个版本返回模型原生的嵌入维度。
    """
    def __init__(self, model_name: str = "OpenGVLab/InternVL3-1B-hf", cache_dir: Optional[str] = None, device: Optional[str] = None, torch_dtype: str = "auto", attn_implementation: str = "sdpa", **kwargs):
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        super().__init__(model_name, cache_dir, device, **kwargs)
    
    def _load_model(self, **kwargs) -> None:
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            # 启用新的、更快的图像处理器以提升性能
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir, 
                trust_remote_code=True, 
                use_fast=True
            )
            
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
                logger.info("在CUDA设备上，自动设置torch_dtype=torch.bfloat16")
                model_kwargs["torch_dtype"] = torch.bfloat16
            
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
        # 根据config, InternVL3-1B的文本部分隐藏层大小为896
        return self.model.config.text_config.hidden_size
    
    def _extract_embeddings_from_hidden_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).to(hidden_states.device)
        masked_embeddings = hidden_states * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings.detach().cpu().numpy()
    
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        # 修复: 直接从模型配置获取真实的原生维度，避免被子类重写的方法影响
        native_dim = self.model.config.text_config.hidden_size
        if not texts: return np.empty((0, native_dim), dtype=np.float32)
        final_embeddings = np.zeros((len(texts), native_dim), dtype=np.float32)
        
        non_empty_texts, non_empty_indices = [], []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        if not non_empty_texts: return final_embeddings
        batch_size = kwargs.get('batch_size', 8)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(non_empty_texts), batch_size), desc="编码文本 (InternVL)"):
                batch_texts = non_empty_texts[i:i + batch_size]
                inputs = self.processor(text=batch_texts, padding=True, return_tensors="pt").to(self.model.device)
                outputs = self.model.language_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
                batch_embeddings = self._extract_embeddings_from_hidden_states(outputs.hidden_states[-1], inputs.attention_mask)
                original_indices = [non_empty_indices[j] for j in range(i, i + len(batch_texts))]
                final_embeddings[original_indices, :] = batch_embeddings
        return final_embeddings
    
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        native_dim = self.model.config.text_config.hidden_size
        if not image_paths: return np.empty((0, native_dim), dtype=np.float32)
        final_embeddings = np.zeros((len(image_paths), native_dim), dtype=np.float32)

        valid_paths, valid_indices = [], []
        for i, path in enumerate(image_paths):
            if path and str(path).strip() and Path(path).exists():
                valid_paths.append(path)
                valid_indices.append(i)

        if not valid_paths: return final_embeddings
        batch_size = kwargs.get('batch_size', 4)

        with torch.no_grad():
            for i in tqdm(range(0, len(valid_paths), batch_size), desc="编码图像 (InternVL)"):
                batch_paths = valid_paths[i:i + batch_size]
                try:
                    images = [Image.open(path).convert("RGB") for path in batch_paths]
                    # 对于纯图像，我们仍然需要一个虚拟的文本输入
                    dummy_text = "Please describe the image in detail."
                    inputs = self.processor(text=dummy_text, images=images, padding=True, return_tensors="pt").to(self.model.device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    batch_embeddings = self._extract_embeddings_from_hidden_states(outputs.hidden_states[-1], inputs.attention_mask)
                    original_indices = [valid_indices[j] for j in range(i, i + len(batch_paths))]
                    final_embeddings[original_indices, :] = batch_embeddings
                except Exception as e:
                    logger.error(f"处理批次时发生错误 (从索引 {i} 开始): {e}")
                    continue
        return final_embeddings

    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        native_dim = self.model.config.text_config.hidden_size
        if len(texts) != len(image_paths): raise ValueError("文本和图像列表的长度必须相等")
        if not texts: return np.empty((0, native_dim), dtype=np.float32)
        final_embeddings = np.zeros((len(texts), native_dim), dtype=np.float32)

        valid_indices, valid_texts, valid_image_paths = [], [], []
        for i, (text, path) in enumerate(zip(texts, image_paths)):
            if text and str(path).strip() and Path(path).exists():
                valid_indices.append(i); valid_texts.append(text); valid_image_paths.append(path)
        
        if not valid_texts: return final_embeddings
        batch_size = kwargs.get('batch_size', 4)
        
        with torch.no_grad():
            for i in tqdm(range(0, len(valid_texts), batch_size), desc="编码多模态 (InternVL)"):
                batch_texts = valid_texts[i:i + batch_size]
                batch_paths = valid_image_paths[i:i + batch_size]
                try:
                    images = [Image.open(path).convert("RGB") for path in batch_paths]
                    inputs = self.processor(text=batch_texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
                    outputs = self.model(**inputs, output_hidden_states=True)
                    batch_embeddings = self._extract_embeddings_from_hidden_states(outputs.hidden_states[-1], inputs.attention_mask)
                    original_indices = [valid_indices[j] for j in range(i, i + len(batch_texts))]
                    final_embeddings[original_indices, :] = batch_embeddings
                except Exception as e:
                    logger.error(f"处理多模态批次时发生错误 (从索引 {i} 开始): {e}")
                    continue
        return final_embeddings

@EncoderFactory.register("intern_vl_with_dim")
class InternVL3EncoderWithDimension(InternVL3Encoder):
    """
    带维度变换的InternVL3编码器。
    """
    def __init__(self, target_dimension: int, reduction_method: Literal["linear", "pca"] = "linear", **kwargs):
        super().__init__(**kwargs)
        self.target_dimension = target_dimension
        model_native_dim = super().get_native_embedding_dim()
        self.dimension_reducer = DimensionReducer(input_dim=model_native_dim, output_dim=target_dimension, method=reduction_method)
    
    def _apply_dimension_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        return self.dimension_reducer.transform(embeddings)

    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        native_embeddings = super().encode_text(texts, **kwargs)
        return self._apply_dimension_reduction(native_embeddings)
    
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        native_embeddings = super().encode_image(image_paths, **kwargs)
        return self._apply_dimension_reduction(native_embeddings)
    
    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        native_embeddings = super().encode_multimodal(texts, image_paths, **kwargs)
        return self._apply_dimension_reduction(native_embeddings)
    
    def get_native_embedding_dim(self) -> int:
        return self.target_dimension