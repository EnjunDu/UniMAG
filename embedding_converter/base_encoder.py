import abc
import torch
import numpy as np
from typing import List, Optional
from enum import Enum

class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"

class BaseEncoder(abc.ABC):
    """
    所有编码器的抽象基类。
    定义了编码器的标准接口，确保所有编码器都可以被特征提取管道统一调用。
    """
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None, device: Optional[str] = None, **kwargs):
        """
        初始化编码器。

        Args:
            model_name (str): Hugging Face上的模型名称。
            cache_dir (Optional[str]): 模型缓存目录。如果为None，则使用Hugging Face默认缓存。
            device (Optional[str]): 计算设备 (例如 "cuda:0", "cpu")。如果为None，则自动检测。
            **kwargs: 传递给模型加载的其他参数。
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device or self._get_default_device()
        self.model = None
        self.processor = None
        self._load_model(**kwargs)
    
    def _get_default_device(self) -> str:
        """获取默认的计算设备。"""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @abc.abstractmethod
    def _load_model(self, **kwargs) -> None:
        """
        加载模型和处理器/分词器。
        子类必须实现此方法。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_native_embedding_dim(self) -> int:
        """
        获取模型原生的嵌入维度。
        子类必须实现此方法。
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        将一批文本编码为嵌入向量。
        子类必须实现此方法。
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """
        将一批图像编码为嵌入向量。
        子类必须实现此方法。
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        """
        将一批图文对编码为多模态嵌入向量。
        子类必须实现此方法。
        """
        raise NotImplementedError
    
    def encode(self, modality: ModalityType, texts: Optional[List[str]] = None, image_paths: Optional[List[str]] = None, **kwargs) -> np.ndarray:
        """
        统一的编码调度接口。
        根据指定的模态调用相应的编码方法。
        """
        if modality == ModalityType.TEXT:
            if texts is None:
                raise ValueError("进行文本编码时，'texts'参数不能为空。")
            return self.encode_text(texts, **kwargs)
        elif modality == ModalityType.IMAGE:
            if image_paths is None:
                raise ValueError("进行图像编码时，'image_paths'参数不能为空。")
            return self.encode_image(image_paths, **kwargs)
        elif modality == ModalityType.MULTIMODAL:
            if texts is None or image_paths is None:
                raise ValueError("进行多模态编码时，'texts'和'image_paths'参数均不能为空。")
            return self.encode_multimodal(texts, image_paths, **kwargs)
        else:
            raise ValueError(f"不支持的模态类型: {modality}")