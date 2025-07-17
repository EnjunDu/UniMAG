"""
嵌入管理器 (EmbeddingManager)

这是为整个UniMAG项目提供的、用于访问和管理所有预生成嵌入向量的API。
它的目标是为下游任务（如节点分类、链接预测）的开发者提供一个极其简洁的接口，
让他们可以轻松获取所需特征，而无需关心特征是如何生成、存储或命名的。
"""

import sys
from pathlib import Path
import numpy as np
from typing import Optional, Union, List, Any, Dict
from PIL.Image import Image
import json
import glob

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
    一个高级API，用于获取和查看预先生成的嵌入向量以及原始数据。
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
            dataset_name (str): 数据集名称 (例如 "books-nc")。
            modality (str): 模态 ("text", "image", 或 "multimodal")。
            encoder_name (str): 编码器模型的Hugging Face名称 (例如 "Qwen/Qwen2.5-VL-3B-Instruct")。
            dimension (Optional[int]): 特征维度。如果为None，则获取原生维度特征。

        Returns:
            Optional[np.ndarray]: 加载的嵌入向量Numpy数组。如果文件不存在则返回None。
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

    def get_raw_data_by_index(self, dataset_name: str, node_index: int) -> Optional[Dict[str, str]]:
        """
        根据节点在特征文件中的索引，获取其原始文本和图像路径。

        Args:
            dataset_name (str): 数据集名称。
            node_index (int): 节点的索引。

        Returns:
            Optional[Dict[str, str]]: 包含 'text' 和 'image_path' 的字典，如果找不到则返回None。
        """
        try:
            node_ids = self._storage.load_node_ids(dataset_name)
            if node_index >= len(node_ids):
                print(f"错误: 索引 {node_index} 超出范围。")
                return None
            target_id = node_ids[node_index]

            text_map = self._storage.load_raw_text_map(dataset_name)
            text = text_map.get(target_id, "")

            image_map = self._storage.load_image_path_map(dataset_name)
            image_path = image_map.get(target_id, "")
            
            if not text and not image_path:
                print(f"警告: 未能为ID '{target_id}' 找到文本或图像。")
                return None

            return {"text": text, "image_path": str(image_path)}

        except FileNotFoundError as e:
            print(f"错误: {e}")
            return None
        except Exception as e:
            print(f"获取原始数据时发生未知错误: {e}")
            return None

    def generate_embedding(
        self,
        data: Union[List[str], List[Image], tuple],
        modality: str,
        encoder_type: str,
        encoder_name: str,
        dimension: Optional[int] = None,
        output_feature_map: bool = False,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        即时生成嵌入向量或特征图。

        Args:
            data (Union[List[str], List[Image], tuple]): 原始数据。
                - 对于 "text": 文本字符串列表 (List[str])。
                - 对于 "image": PIL图像对象或图像路径字符串的列表 (List[Union[Image, str]])。
                - 对于 "multimodal": (texts, images) 的元组，其中 images 是 List[Union[Image, str]]。
            modality (str): 模态 ("text", "image", 或 "multimodal")。
            encoder_type (str): 在EncoderFactory中注册的编码器类型 (例如, "qwen_vl", "bert")。
            encoder_name (str): 编码器模型的Hugging Face名称。
            dimension (Optional[int]): 目标特征维度。如果为None，则使用原生维度。
            output_feature_map (bool): 如果为True，则返回特征图而不是嵌入向量。
            **kwargs: 传递给编码器的其他参数 (例如, cache_dir, device)。

        Returns:
            Optional[np.ndarray]: 生成的嵌入向量或特征图。
        """
        try:
            modality_enum = ModalityType(modality)
        except ValueError:
            print(f"错误: 不支持的模态 '{modality}'")
            return None

        encoder_instance_key = f"{encoder_type}_{encoder_name}_{dimension or 'native'}"
        if encoder_instance_key not in self._encoders:
            factory_encoder_type = f"{encoder_type}_with_dim" if dimension else encoder_type
            
            encoder_kwargs = {
                'model_name': encoder_name,
                'cache_dir': kwargs.get('cache_dir'),
                'device': kwargs.get('device')
            }
            if dimension:
                encoder_kwargs['target_dimension'] = dimension
            self._encoders[encoder_instance_key] = EncoderFactory.create_encoder(factory_encoder_type, **encoder_kwargs)
        
        encoder = self._encoders[encoder_instance_key]

        if modality_enum == ModalityType.TEXT:
            return encoder.encode_text(data, output_feature_map=output_feature_map, **kwargs)
        elif modality_enum == ModalityType.IMAGE:
            return encoder.encode_image(data, output_feature_map=output_feature_map, **kwargs)
        elif modality_enum == ModalityType.MULTIMODAL:
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
    print("\n--- 示例1: 获取 'books-nc' 的文本特征 ---")
    text_embeddings = manager.get_embedding(
        dataset_name="books-nc",
        modality="text",
        encoder_name="Qwen/Qwen2.5-VL-3B-Instruct",
        dimension=768
    )
    if text_embeddings is not None:
        print(f"成功获取文本特征，形状: {text_embeddings.shape}, 类型: {text_embeddings.dtype}")

    # 3. 查看嵌入向量的详细信息和预览
    print("\n--- 示例2: 查看 'books-nc' 的多模态特征 ---")
    manager.view_embedding(
        dataset_name="books-nc",
        modality="multimodal",
        encoder_name="Qwen/Qwen2.5-VL-3B-Instruct",
        dimension=None
    )

    # 4. 即时生成嵌入
    print("\n--- 示例3: 即时生成文本嵌入 ---")
    custom_texts = ["这是一个测试。", "这是另一个测试。"]
    on_the_fly_embeddings = manager.generate_embedding(
        data=custom_texts,
        modality="text",
        encoder_type="qwen_vl",
        encoder_name="Qwen/Qwen2.5-VL-7B-Instruct"
    )
    if on_the_fly_embeddings is not None:
        print(f"成功即时生成文本嵌入，形状: {on_the_fly_embeddings.shape}")

    # 5. 根据索引获取原始数据
    print("\n--- 示例4: 获取 'Grocery' 数据集索引为 2 的节点的原始数据 ---")
    raw_data = manager.get_raw_data_by_index("Grocery", 2)
    if raw_data:
        print(f"  文本: {raw_data['text'][:100]}...")
        print(f"  图像路径: {raw_data['image_path']}")

    print("\n--- 示例5: 获取 'sports-copurchase' 数据集索引为 10 的节点的原始数据 ---")
    raw_data_sports = manager.get_raw_data_by_index("sports-copurchase", 10)
    if raw_data_sports:
        print(f"  文本: {raw_data_sports['text'][:100]}...")
        print(f"  图像路径: {raw_data_sports['image_path']}")