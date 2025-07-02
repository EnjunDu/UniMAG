from typing import Dict, Type, Optional, Literal, Any
import logging

from .base_encoder import BaseEncoder, EncoderConfig
from .qwen_vl_encoder import QwenVLEncoder, QwenVLEncoderWithDimension

logger = logging.getLogger(__name__)

class EncoderFactory:
    """
    编码器工厂类
    
    负责创建和管理不同类型的编码器实例
    """
    
    # 注册的编码器类型
    _encoder_registry: Dict[str, Type[BaseEncoder]] = {
        "qwen_vl": QwenVLEncoder,
        "qwen_vl_with_dim": QwenVLEncoderWithDimension,
    }
    
    @classmethod
    def register_encoder(cls, name: str, encoder_class: Type[BaseEncoder]) -> None:
        """
        注册新的编码器类型
        
        Args:
            name: 编码器名称
            encoder_class: 编码器类
        """
        cls._encoder_registry[name] = encoder_class
        logger.info(f"注册编码器: {name}")
    
    @classmethod
    def get_available_encoders(cls) -> list[str]:
        """获取所有可用的编码器名称"""
        return list(cls._encoder_registry.keys())
    
    @classmethod
    def create_encoder(cls, 
                      encoder_type: str,
                      config: Optional[EncoderConfig] = None,
                      **kwargs) -> BaseEncoder:
        """
        创建编码器实例
        
        Args:
            encoder_type: 编码器类型
            config: 编码器配置
            **kwargs: 额外参数
            
        Returns:
            编码器实例
        """
        if encoder_type not in cls._encoder_registry:
            raise ValueError(f"未知的编码器类型: {encoder_type}. "
                           f"可用类型: {list(cls._encoder_registry.keys())}")
        
        encoder_class = cls._encoder_registry[encoder_type]
        
        # 合并配置和额外参数
        init_kwargs = {}
        if config:
            init_kwargs.update({
                'model_name': config.model_name,
                'cache_dir': config.cache_dir,
                'device': config.device,
                'batch_size': config.batch_size,
                'max_length': config.max_length,
                **config.model_kwargs
            })
        init_kwargs.update(kwargs)
        
        logger.info(f"创建编码器: {encoder_type}")
        return encoder_class(**init_kwargs)
    
    @classmethod
    def create_qwen_vl_encoder(cls,
                              model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                              cache_dir: Optional[str] = None,
                              device: Optional[str] = None,
                              target_dimension: Optional[int] = None,
                              reduction_method: Literal["linear", "pca"] = "linear",
                              **kwargs) -> BaseEncoder:
        """
        便捷方法：创建Qwen VL编码器
        
        Args:
            model_name: 模型名称
            cache_dir: 缓存目录
            device: 运行设备
            target_dimension: 目标维度（如果指定则使用带维度变换的版本）
            reduction_method: 降维方法
            **kwargs: 额外参数
            
        Returns:
            Qwen VL编码器实例
        """
        if target_dimension is not None:
            return cls.create_encoder(
                "qwen_vl_with_dim",
                model_name=model_name,
                cache_dir=cache_dir,
                device=device,
                target_dimension=target_dimension,
                reduction_method=reduction_method,
                **kwargs
            )
        else:
            return cls.create_encoder(
                "qwen_vl",
                model_name=model_name,
                cache_dir=cache_dir,
                device=device,
                **kwargs
            )


class EncoderManager:
    """
    编码器管理器
    
    负责管理编码器的生命周期和资源
    """
    
    def __init__(self):
        self._active_encoders: Dict[str, BaseEncoder] = {}
    
    def get_encoder(self, 
                   encoder_id: str,
                   encoder_type: str,
                   config: Optional[EncoderConfig] = None,
                   **kwargs) -> BaseEncoder:
        """
        获取编码器实例（支持复用）
        
        Args:
            encoder_id: 编码器标识符
            encoder_type: 编码器类型
            config: 编码器配置
            **kwargs: 额外参数
            
        Returns:
            编码器实例
        """
        if encoder_id in self._active_encoders:
            logger.info(f"复用已存在的编码器: {encoder_id}")
            return self._active_encoders[encoder_id]
        
        # 创建新的编码器
        encoder = EncoderFactory.create_encoder(encoder_type, config, **kwargs)
        self._active_encoders[encoder_id] = encoder
        logger.info(f"创建并缓存编码器: {encoder_id}")
        
        return encoder
    
    def remove_encoder(self, encoder_id: str) -> None:
        """
        移除编码器实例
        
        Args:
            encoder_id: 编码器标识符
        """
        if encoder_id in self._active_encoders:
            del self._active_encoders[encoder_id]
            logger.info(f"移除编码器: {encoder_id}")
    
    def clear_all_encoders(self) -> None:
        """清除所有编码器实例"""
        self._active_encoders.clear()
        logger.info("清除所有编码器")
    
    def get_active_encoders(self) -> Dict[str, BaseEncoder]:
        """获取所有活跃的编码器"""
        return self._active_encoders.copy()


# 全局编码器管理器实例
encoder_manager = EncoderManager()