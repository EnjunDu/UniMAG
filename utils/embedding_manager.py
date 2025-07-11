"""
嵌入管理器 (EmbeddingManager)

这是为整个UniMAG项目提供的、用于访问和管理所有预生成嵌入向量的API。
它的目标是为下游任务（如节点分类、链接预测）的开发者提供一个极其简洁的接口，
让他们可以轻松获取所需特征，而无需关心特征是如何生成、存储或命名的。
"""

import sys
from pathlib import Path
import numpy as np
from typing import Optional, Union, List, Any
from PIL.Image import Image

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 导入位于子模块中的底层工具
from embedding_converter.utils.storage_manager import StorageManager
from embedding_converter.utils.view_npy_file import view_npy as view_npy_file_content
from embedding_converter.encoder_factory import EncoderFactory
from embedding_converter.base_encoder import ModalityType
# 导入此包以触发所有编码器的自动注册
from embedding_converter import encoders

class EmbeddingManager:
    """
    一个高级API，用于获取和查看预先生成的嵌入向量。
    """
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        初始化嵌入管理器。

        Args:
            base_path (Optional[Union[str, Path]]):
                数据集的根目录路径。如果未提供，
                将使用StorageManager中的默认路径 (例如 /home/ai/MMAG)。
        """
        self._storage = StorageManager(base_path=base_path)
        self._encoders: Dict[str, Any] = {}

    def get_embedding(
        self,
        dataset_name: str,
        modality: str,
        encoder_name: str,
        dimension: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        获取指定参数的嵌入向量。

        Args:
            dataset_name (str): 数据集名称 (例如 "books-nc-50")。
            modality (str): 模态 ("text", "image", 或 "multimodal")。
            encoder_name (str): 编码器模型的Hugging Face名称 (例如 "Qwen/Qwen2.5-VL-3B-Instruct")。
            dimension (Optional[int]): 特征维度。如果为None，则获取原生维度特征。

        Returns:
            np.ndarray: 加载的嵌入向量Numpy数组。如果文件不存在则返回None。
        """
        feature_path = self._storage.get_feature_path(
            dataset_name=dataset_name,
            modality=modality,
            encoder_name=encoder_name,
            dimension=dimension
        )

        if not feature_path.exists():
            print(f"警告: 未找到对应的嵌入文件: {feature_path}")
            return None
        
        return self._storage.load_features(feature_path)

    def generate_embedding(
        self,
        data: Union[List[str], List[Image]],
        modality: str,
        encoder_name: str,
        dimension: Optional[int] = None,
        output_feature_map: bool = False,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        即时生成嵌入向量或特征图。

        Args:
            data (Union[List[str], List[Image]]): 原始数据，可以是文本字符串列表或PIL图像列表。
            modality (str): 模态 ("text", "image", 或 "multimodal")。
            encoder_name (str): 编码器模型的Hugging Face名称。
            dimension (Optional[int]): 目标特征维度。如果为None，则使用原生维度。
            output_feature_map (bool): 如果为True，则返回特征图而不是嵌入向量。
            **kwargs: 传递给编码器的其他参数 (例如, cache_dir, device)。

        Returns:
            np.ndarray: 生成的嵌入向量或特征图。
        """
        try:
            modality_enum = ModalityType(modality)
        except ValueError:
            print(f"错误: 不支持的模态 '{modality}'")
            return None

        encoder_key = f"{encoder_name}_{dimension or 'native'}"
        if encoder_key not in self._encoders:
            # TODO: 这里的实现是一个简化的假设。一个更健壮的实现需要一个机制
            # 来从encoder_name（例如 "Qwen/Qwen2.5-VL-7B-Instruct"）动态推断出
            # 应该在EncoderFactory中使用的基础encoder_type（例如 "qwen_vl"）。
            # 目前，它硬编码为 "qwen_vl"，这对于其他类型的编码器会失败。
            encoder_type = "qwen_vl"
            if dimension:
                encoder_type += "_with_dim"
            
            encoder_kwargs = {
                'model_name': encoder_name,
                'cache_dir': kwargs.get('cache_dir'),
                'device': kwargs.get('device')
            }
            if dimension:
                encoder_kwargs['target_dimension'] = dimension

            self._encoders[encoder_key] = EncoderFactory.create_encoder(encoder_type, **encoder_kwargs)
        
        encoder = self._encoders[encoder_key]

        if modality_enum == ModalityType.TEXT:
            return encoder.encode_text(data, output_feature_map=output_feature_map, **kwargs)
        elif modality_enum == ModalityType.IMAGE:
            # TODO: 当前BaseEncoder中的encode_image方法期望接收文件路径列表，
            # 而不是原始的PIL图像对象列表。这是一个设计上的不一致性，因为
            # 即时生成功能的核心理念是处理内存中的原始数据。
            # 需要重构BaseEncoder及其子类，使其能够直接处理PIL.Image对象。
            # 目前，调用者需要传入一个文件路径列表才能使其工作。
            return encoder.encode_image(data, output_feature_map=output_feature_map, **kwargs)
        elif modality_enum == ModalityType.MULTIMODAL:
            # TODO: 多模态的即时生成需要一个更清晰的输入格式。当前实现假设
            # 调用者会传入一个 (texts, image_paths) 的元组，这可能不够直观。
            # 未来的版本可以考虑接受一个更结构化的输入，例如一个字典列表：
            # [{"text": "...", "image": <PIL.Image>}, ...]
            texts, image_paths = data
            return encoder.encode_multimodal(texts, image_paths, output_feature_map=output_feature_map, **kwargs)
        
        return None

    def view_embedding(
        self,
        dataset_name: str,
        modality: str,
        encoder_name: str,
        dimension: Optional[int] = None,
        head: int = 5,
        tail: int = 0
    ):
        """
        加载并以表格形式显示嵌入向量的信息和预览。

        Args:
            dataset_name (str): 数据集名称。
            modality (str): 模态。
            encoder_name (str): 编码器模型名称。
            dimension (Optional[int]): 特征维度。
            head (int): 显示头部的行数。
            tail (int): 显示尾部的行数。
        """
        feature_path = self._storage.get_feature_path(
            dataset_name=dataset_name,
            modality=modality,
            encoder_name=encoder_name,
            dimension=dimension
        )
        
        if not feature_path.exists():
            print(f"错误: 无法查看，文件不存在: {feature_path}")
            return

        view_npy_file_content(file_path=feature_path, head=head, tail=tail)

if __name__ == '__main__':
    # === 使用示例 ===
    print("=== EmbeddingManager 使用示例 ===")
    
    # 1. 初始化管理器
    # 默认情况下，它会使用StorageManager的默认路径
    manager = EmbeddingManager()
    
    # 或者，你可以提供一个自定义路径
    # local_manager = EmbeddingManager(base_path="./hugging_face")

    # 2. 获取嵌入向量
    print("\n--- 示例1: 获取 'books-nc-50' 的文本特征 ---")
    text_embeddings = manager.get_embedding(
        dataset_name="books-nc-50",
        modality="text",
        encoder_name="Qwen/Qwen2.5-VL-3B-Instruct",
        dimension=768
    )
    if text_embeddings is not None:
        print(f"成功获取文本特征，形状: {text_embeddings.shape}, 类型: {text_embeddings.dtype}")

    # 3. 查看嵌入向量的详细信息和预览
    print("\n--- 示例2: 查看 'books-nc-50' 的多模态特征 ---")
    manager.view_embedding(
        dataset_name="books-nc-50",
        modality="multimodal",
        encoder_name="Qwen/Qwen2.5-VL-3B-Instruct",
        dimension=None  # 查看原生维度
    )

    # 4. 即时生成嵌入
    print("\n--- 示例3: 即时生成文本嵌入 ---")
    custom_texts = ["这是一个测试。", "这是另一个测试。"]
    on_the_fly_embeddings = manager.generate_embedding(
        data=custom_texts,
        modality="text",
        encoder_name="Qwen/Qwen2.5-VL-7B-Instruct" # 使用一个具体的模型
    )
    if on_the_fly_embeddings is not None:
        print(f"成功即时生成文本嵌入，形状: {on_the_fly_embeddings.shape}")