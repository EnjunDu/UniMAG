from typing import Optional

from .base_encoder import BaseEncoder, ModalityType, EncoderConfig
from .dimension_reducer import DimensionReducer
from .qwen_vl_encoder import QwenVLEncoder, QwenVLEncoderWithDimension
from .encoder_factory import EncoderFactory, EncoderManager, encoder_manager
from .quality_checker import QualityChecker
from .config_loader import load_embedding_config, ConfigLoader

__version__ = "0.2.0"

__all__ = [
    # 基础类
    "BaseEncoder",
    "ModalityType",
    "EncoderConfig",

    # 维度缩减
    "DimensionReducer",

    # Qwen编码器
    "QwenVLEncoder",
    "QwenVLEncoderWithDimension",

    # 工厂和管理器
    "EncoderFactory",
    "EncoderManager",
    "encoder_manager",

    # 质量检查
    "QualityChecker",

    # 配置加载
    "ConfigLoader",
    "load_embedding_config",
]

def get_available_encoders():
    """获取所有可用的编码器类型"""
    return EncoderFactory.get_available_encoders()