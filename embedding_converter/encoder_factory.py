from typing import Dict, Type, Any
import logging
from .base_encoder import BaseEncoder

logger = logging.getLogger(__name__)

class EncoderFactory:
    """
    编码器工厂，采用装饰器模式实现自动注册和动态创建。
    
    使用方法:
    1. 定义一个新的编码器类，继承自BaseEncoder。
    2. 在类定义上方加上装饰器 @EncoderFactory.register("your_encoder_name")。
    3. 即可通过 EncoderFactory.create_encoder("your_encoder_name", **kwargs) 来创建实例。
    """
    _encoder_registry: Dict[str, Type[BaseEncoder]] = {}

    @classmethod
    def register(cls, name: str):
        """
        一个用于注册编码器类的装饰器。

        Args:
            name (str): 用于在工厂中引用该编码器的唯一名称。
        """
        def decorator(encoder_class: Type[BaseEncoder]) -> Type[BaseEncoder]:
            if name in cls._encoder_registry:
                logger.warning(f"编码器名称 '{name}' 已被注册，将被覆盖。")
            cls._encoder_registry[name] = encoder_class
            logger.info(f"成功注册编码器: '{name}' -> {encoder_class.__name__}")
            return encoder_class
        return decorator

    @classmethod
    def create_encoder(cls, name: str, **kwargs: Any) -> BaseEncoder:
        """
        根据名称和参数创建编码器实例。

        Args:
            name (str): 已注册的编码器名称。
            **kwargs: 传递给编码器构造函数的参数。

        Returns:
            BaseEncoder: 所请求的编码器类的实例。
            
        Raises:
            ValueError: 如果请求的编码器名称未被注册。
        """
        if name not in cls._encoder_registry:
            available_encoders = ", ".join(cls._encoder_registry.keys())
            raise ValueError(f"未知的编码器类型: '{name}'. 可用类型: [{available_encoders}]")
        
        encoder_class = cls._encoder_registry[name]
        logger.info(f"正在从工厂创建编码器实例: '{name}'")
        return encoder_class(**kwargs)

    @classmethod
    def get_available_encoders(cls) -> list[str]:
        """获取所有已注册的编码器名称列表。"""
        return list(cls._encoder_registry.keys())