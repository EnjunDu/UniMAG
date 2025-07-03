"""
嵌入管理器 (EmbeddingManager)

这是为整个UniMAG项目提供的、用于访问和管理所有预生成嵌入向量的官方高级API。
它的目标是为下游任务（如节点分类、链接预测）的开发者提供一个极其简洁的接口，
让他们可以轻松获取所需特征，而无需关心特征是如何生成、存储或命名的。
"""

import sys
from pathlib import Path
import numpy as np
from typing import Optional

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 导入位于子模块中的底层工具
from embedding_converter.utils.storage_manager import StorageManager
from embedding_converter.utils.view_npy_file import view_npy as view_npy_file_content

class EmbeddingManager:
    """
    一个高级API，用于获取和查看预先生成的嵌入向量。
    """
    def __init__(self):
        """
        初始化嵌入管理器。
        使用默认的服务器路径配置来初始化底层的存储管理器。
        """
        self._storage = StorageManager()

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
    
    # 1. 初始化管理器 (通常在你的项目代码的某个地方进行)
    manager = EmbeddingManager()

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