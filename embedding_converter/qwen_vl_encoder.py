import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Union, Literal
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import logging
from tqdm import tqdm

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
                        # 提取图像路径
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
    
    支持文本、图像和多模态编码
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 cache_dir: Optional[str] = None,
                 device: Optional[str] = None,
                 torch_dtype: str = "auto",
                 attn_implementation: str = "flash_attention_2",
                 **kwargs):
        """
        初始化Qwen2.5-VL编码器
        
        Args:
            model_name: 模型名称
            cache_dir: 缓存目录
            device: 运行设备
            torch_dtype: PyTorch数据类型
            attn_implementation: 注意力实现方式
        """
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        super().__init__(model_name, cache_dir, device, **kwargs)
    
    def _load_model(self, **kwargs) -> None:
        """加载Qwen2.5-VL模型和处理器"""
        try:
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 加载处理器
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # 设置模型加载参数
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True
            }
            
            # 设置数据类型
            if self.torch_dtype != "auto":
                model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)
            
            # 设置注意力实现
            if self.attn_implementation and self.device == "cuda":
                model_kwargs["attn_implementation"] = self.attn_implementation
            
            # 加载模型
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                **model_kwargs
            )
            
            # 如果没有使用device_map，手动移动到设备
            if self.device and (self.device != "cuda" or "device_map" not in model_kwargs):
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def get_native_embedding_dim(self) -> int:
        """获取模型原生嵌入维度"""
        # Qwen2.5-VL-7B的隐层维度为4096
        return self.model.config.hidden_size
    
    def _extract_embeddings_from_hidden_states(self, 
                                             hidden_states: torch.Tensor,
                                             attention_mask: torch.Tensor) -> np.ndarray:
        """
        从隐层状态中提取特征向量
        
        Args:
            hidden_states: 模型输出的隐层状态 [batch_size, seq_len, hidden_dim]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            提取的特征向量 [batch_size, hidden_dim]
        """
        # 使用注意力掩码进行平均池化
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        masked_embeddings = hidden_states * mask_expanded
        sum_embeddings = masked_embeddings.sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1)
        
        # 避免除零
        mean_embeddings = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        
        return mean_embeddings.detach().cpu().numpy()
    
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        编码纯文本
        
        Args:
            texts: 文本列表
            
        Returns:
            文本嵌入矩阵 [num_texts, hidden_dim]
        """
        if not texts:
            return np.empty((0, self.get_native_embedding_dim()))
        
        batch_size = kwargs.get('batch_size', 8)
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
                batch_texts = texts[i:i + batch_size]
                
                # 构建消息格式
                messages_batch = []
                for text in batch_texts:
                    messages_batch.append([{
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                    }])
                
                # 应用对话模板
                texts_formatted = [
                    self.processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    ) for msgs in messages_batch
                ]
                
                # 处理视觉信息（纯文本时为空）
                image_inputs, video_inputs = process_vision_info(messages_batch)
                
                # 编码
                inputs = self.processor(
                    text=texts_formatted,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # 前向传播获取隐层状态
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # 最后一层
                
                # 提取嵌入
                embeddings = self._extract_embeddings_from_hidden_states(
                    hidden_states, inputs.attention_mask
                )
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """
        编码纯图像
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            图像嵌入矩阵 [num_images, hidden_dim]
        """
        if not image_paths:
            return np.empty((0, self.get_native_embedding_dim()))
        
        batch_size = kwargs.get('batch_size', 4)  # 图像批次小一些
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(image_paths), batch_size), desc="编码图像"):
                batch_paths = image_paths[i:i + batch_size]
                
                # 构建消息格式
                messages_batch = []
                for path in batch_paths:
                    messages_batch.append([{
                        "role": "user", 
                        "content": [
                            {"type": "image", "image": f"file://{path}"},
                            {"type": "text", "text": "描述这张图片"}
                        ]
                    }])
                
                # 应用对话模板
                texts_formatted = [
                    self.processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    ) for msgs in messages_batch
                ]
                
                # 处理视觉信息
                image_inputs, video_inputs = process_vision_info(messages_batch)
                
                # 编码
                inputs = self.processor(
                    text=texts_formatted,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # 前向传播获取隐层状态
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                # 提取嵌入
                embeddings = self._extract_embeddings_from_hidden_states(
                    hidden_states, inputs.attention_mask
                )
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def encode_multimodal(self, 
                         texts: List[str], 
                         image_paths: List[str], 
                         **kwargs) -> np.ndarray:
        """
        编码多模态数据（文本+图像）
        
        Args:
            texts: 文本列表
            image_paths: 图像路径列表
            
        Returns:
            多模态嵌入矩阵 [num_pairs, hidden_dim]
        """
        if len(texts) != len(image_paths):
            raise ValueError("文本和图像数量必须相等")
        
        if not texts:
            return np.empty((0, self.get_native_embedding_dim()))
        
        batch_size = kwargs.get('batch_size', 4)
        all_embeddings = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="编码多模态"):
                batch_texts = texts[i:i + batch_size]
                batch_paths = image_paths[i:i + batch_size]
                
                # 构建消息格式
                messages_batch = []
                for text, path in zip(batch_texts, batch_paths):
                    messages_batch.append([{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{path}"},
                            {"type": "text", "text": text}
                        ]
                    }])
                
                # 应用对话模板
                texts_formatted = [
                    self.processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False
                    ) for msgs in messages_batch
                ]
                
                # 处理视觉信息
                image_inputs, video_inputs = process_vision_info(messages_batch)
                
                # 编码
                inputs = self.processor(
                    text=texts_formatted,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # 前向传播获取隐层状态
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                
                # 提取嵌入
                embeddings = self._extract_embeddings_from_hidden_states(
                    hidden_states, inputs.attention_mask
                )
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)

class QwenVLEncoderWithDimension(QwenVLEncoder):
    """
    带维度变换的Qwen2.5-VL编码器
    
    支持将原生嵌入变换到指定维度
    """
    
    def __init__(self,
                 target_dimension: int,
                 reduction_method: Literal["linear", "pca"] = "linear",
                 **kwargs):
        """
        初始化带维度变换的编码器
        
        Args:
            target_dimension: 目标嵌入维度
            reduction_method: 降维方法 ('linear', 'pca')
        """
        super().__init__(**kwargs)
        self.target_dimension = target_dimension
        self.dimension_reducer = DimensionReducer(
            input_dim=self.get_native_embedding_dim(),
            output_dim=target_dimension,
            method=reduction_method
        )
    
    def _apply_dimension_reduction(self, embeddings: np.ndarray) -> np.ndarray:
        """应用维度变换"""
        return self.dimension_reducer.transform(embeddings)
    
    def encode_text(self, texts: List[str], **kwargs) -> np.ndarray:
        """编码文本并应用维度变换"""
        embeddings = super().encode_text(texts, **kwargs)
        return self._apply_dimension_reduction(embeddings)
    
    def encode_image(self, image_paths: List[str], **kwargs) -> np.ndarray:
        """编码图像并应用维度变换"""
        embeddings = super().encode_image(image_paths, **kwargs)
        return self._apply_dimension_reduction(embeddings)
    
    def encode_multimodal(self, texts: List[str], image_paths: List[str], **kwargs) -> np.ndarray:
        """编码多模态数据并应用维度变换"""
        embeddings = super().encode_multimodal(texts, image_paths, **kwargs)
        return self._apply_dimension_reduction(embeddings)
    
    def get_native_embedding_dim(self) -> int:
        """返回目标维度而不是原生维度"""
        return self.target_dimension