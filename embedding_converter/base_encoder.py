import abc
import torch
import numpy as np
from typing import Dict, List, Union, Optional, Any
from enum import Enum
from pathlib import Path

class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

class BaseEncoder(abc.ABC):
    """抽象编码器基类"""
    
    def __init__(self, 
                 model_name: str,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or self._get_default_device()
        self.model = None
        self.processor = None
        self._load_model(**kwargs)
    
    def _get_default_device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @abc.abstractmethod
    def _load_model(self, **kwargs) -> None:
        """加载模型和处理器"""
        pass
    
    @abc.abstractmethod
    def get_native_embedding_dim(self) -> int:
        """获取模型原生嵌入维度"""
        pass
    
    @abc.abstractmethod
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码文本"""
        pass
    
    @abc.abstractmethod
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """编码图像"""
        pass
    
    @abc.abstractmethod
    def encode_multimodal(self, 
                         texts: List[str], 
                         image_paths: List[str], 
                         **kwargs) -> np.ndarray:
        """编码多模态数据"""
        pass
    
    def get_supported_modalities(self) -> List[ModalityType]:
        return [ModalityType.TEXT, ModalityType.IMAGE, ModalityType.MULTIMODAL]
    
    def encode(self, 
               modality: ModalityType,
               texts: Optional[List[str]] = None,
               image_paths: Optional[List[str]] = None,
               **kwargs) -> np.ndarray:
        """统一编码接口"""
        if modality == ModalityType.TEXT:
            if texts is None:
                raise ValueError("texts不能为空")
            return self.encode_text(texts, **kwargs)
        elif modality == ModalityType.IMAGE:
            if image_paths is None:
                raise ValueError("image_paths不能为空")
            return self.encode_image(image_paths, **kwargs)
        elif modality == ModalityType.MULTIMODAL:
            if texts is None or image_paths is None:
                raise ValueError("texts和image_paths都不能为空")
            return self.encode_multimodal(texts, image_paths, **kwargs)
        else:
            raise ValueError(f"不支持的模态类型: {modality}")

class EncoderConfig:
    """编码器配置类"""
    
    def __init__(self,
                 model_name: str,
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 batch_size: int = 32,
                 max_length: int = 512,
                 **model_kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model_kwargs = model_kwargs