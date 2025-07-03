import numpy as np
import logging
from typing import List

from ..encoder_factory import EncoderFactory
from ..base_encoder import BaseEncoder

logger = logging.getLogger(__name__)

@EncoderFactory.register("vit")
class ViTEncoder(BaseEncoder):
    """
    纯视觉ViT编码器（骨架实现）。
    这是一个如何集成一个单模态（纯视觉）编码器的范例。
    """
    def _load_model(self, **kwargs) -> None:
        """加载ViT模型和Processor的逻辑"""
        # 完整的实现需要取消以下代码的注释并安装transformers
        # from transformers import ViTModel, ViTImageProcessor
        # logger.info(f"正在加载ViT模型: {self.model_name}")
        # self.processor = ViTImageProcessor.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        # self.model = ViTModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        # self.model.to(self.device)
        # self.model.eval()
        logger.info(f"骨架加载(演示): {self.model_name}")
        # 模拟模型加载的配置，以便get_native_embedding_dim可以工作
        self.model = type('obj', (object,), {'config': type('obj', (object,), {'hidden_size': 768})})()

    def get_native_embedding_dim(self) -> int:
        return self.model.config.hidden_size

    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        raise NotImplementedError("ViTEncoder是一个纯视觉编码器，不支持文本编码。")

    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """实现纯视觉编码逻辑"""
        logger.info(f"骨架编码图像: {len(image_paths)}个样本 (返回零向量)")
        # 实际的编码逻辑会在这里
        # from PIL import Image
        # images = [Image.open(path).convert("RGB") for path in image_paths]
        # with torch.no_grad():
        #     inputs = self.processor(images, return_tensors="pt").to(self.device)
        #     outputs = self.model(**inputs)
        #     return outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return np.zeros((len(image_paths), self.get_native_embedding_dim()))

    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        raise NotImplementedError("ViTEncoder是一个纯视觉编码器，不支持多模态编码。")