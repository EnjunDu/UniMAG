# -*- coding: utf-8 -*-
"""
模态匹配 (Modality Matching)

此模块实现了在 "Tasks_and_arrangements.md" 中定义的两种模态匹配方法：
1. 传统方法: 计算任意一对图像和文本嵌入之间的无上下文 CLIP-score。
2. MAG 特定方法: 计算一个节点在图上下文环境下的增强 CLIP-score。
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, Union
from torch_geometric.nn import GCNConv

# 将项目根目录添加到Python路径中，以方便模块导入
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager

def calculate_clip_score(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray
) -> float:
    """
    计算并返回一对图像和文本嵌入之间的 CLIP-score (传统方法)。

    Args:
        image_embedding (np.ndarray): 单个图像的嵌入向量 (1D array)。
        text_embedding (np.ndarray): 单个文本的嵌入向量 (1D array)。

    Returns:
        float: 计算出的 CLIP-score，范围在 -100 到 100 之间。
    """
    image_embedding = np.asarray(image_embedding)
    text_embedding = np.asarray(text_embedding)

    image_embedding_norm = image_embedding / np.linalg.norm(image_embedding)
    text_embedding_norm = text_embedding / np.linalg.norm(text_embedding)

    similarity = np.dot(image_embedding_norm, text_embedding_norm)
    clip_score = 100.0 * similarity

    return clip_score


class MAGModalityMatcher:
    """
    实现 MAG 特定的、考虑图上下文的模态匹配方法。
    """
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        初始化 MAG 特定的模态匹配器。

        Args:
            base_path (Optional[Union[str, Path]]): 数据集的根目录路径。
        """
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        self.gcn_layer = None
        # TODO: 需要一个图数据加载器来获取图结构和邻接信息。
        # self.graph_loader = GraphLoader()

    def _initialize_gcn_layer(self, in_dim: int, out_dim: int):
        """如果需要，则初始化GCN层。"""
        if self.gcn_layer is None:
            self.gcn_layer = GCNConv(in_dim, out_dim)

    def _get_graph_structure(self, dataset_name: str) -> Optional[torch.Tensor]:
        """
        一个用于获取图结构的占位符方法。
        
        TODO: 实际实现需要调用一个图加载器来返回一个 PyG 格式的 edge_index 张量。
              这个加载器是运行此模块的先决条件。
        """
        print(f"警告: 正在使用伪造的图结构 (edge_index) 用于 '{dataset_name}'。")
        # 返回一个伪造的 edge_index [2, num_edges]
        return torch.tensor([
            [0, 1, 1, 2, 3, 0],
            [1, 0, 2, 1, 0, 3]
        ], dtype=torch.long)

    def get_neighbor_enhanced_embedding(
        self,
        dataset_name: str,
        encoder_name: str,
        dimension: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        使用GCN层计算整个图的邻域增强嵌入。

        Args:
            dataset_name (str): 数据集名称。
            encoder_name (str): 编码器模型的Hugging Face名称。
            dimension (Optional[int]): 特征维度。

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]: 
                一个包含 (所有节点的增强图像嵌入, 所有节点的增强文本嵌入) 的元组。
                如果无法获取任何所需嵌入，则返回 None。
        """
        image_embeddings = self.embedding_manager.get_embedding(
            dataset_name, "image", encoder_name, dimension
        )
        text_embeddings = self.embedding_manager.get_embedding(
            dataset_name, "text", encoder_name, dimension
        )

        if image_embeddings is None or text_embeddings is None:
            print("错误: 无法加载基础嵌入，无法进行邻域增强。")
            return None

        edge_index = self._get_graph_structure(dataset_name)
        if edge_index is None:
            print("错误: 无法加载图结构。")
            return None

        # 将numpy嵌入转换为torch张量
        image_features_tensor = torch.from_numpy(image_embeddings).float()
        text_features_tensor = torch.from_numpy(text_embeddings).float()
        
        # 初始化GCN层
        feature_dim = image_features_tensor.shape[1]
        self._initialize_gcn_layer(feature_dim, feature_dim)

        # 使用GCN层进行前向传播
        enhanced_image_embeddings = self.gcn_layer(image_features_tensor, edge_index).detach().numpy()
        enhanced_text_embeddings = self.gcn_layer(text_features_tensor, edge_index).detach().numpy()

        return enhanced_image_embeddings, enhanced_text_embeddings

    def calculate_mag_clip_score(
        self,
        dataset_name: str,
        encoder_name: str,
        dimension: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        计算图中所有节点的上下文感知 CLIP-score。

        Args:
            dataset_name (str): 数据集名称。
            encoder_name (str): 编码器模型的Hugging Face名称。
            dimension (Optional[int]): 特征维度。

        Returns:
            Optional[np.ndarray]: 一个包含所有节点CLIP-score的Numpy数组。
        """
        all_enhanced_embeddings = self.get_neighbor_enhanced_embedding(
            dataset_name, encoder_name, dimension
        )

        if all_enhanced_embeddings is None:
            return None

        enhanced_image_embeddings, enhanced_text_embeddings = all_enhanced_embeddings
        
        num_nodes = enhanced_image_embeddings.shape[0]
        scores = np.zeros(num_nodes)

        for i in range(num_nodes):
            scores[i] = calculate_clip_score(
                enhanced_image_embeddings[i],
                enhanced_text_embeddings[i]
            )
        
        return scores


if __name__ == '__main__':
    print("=== 模态匹配模块使用示例 ===")

    # --- 示例 1: 传统 CLIP-score 计算 ---
    print("\n--- 示例 1: 传统 CLIP-score ---")
    img_emb = np.random.rand(768).astype(np.float32)
    txt_emb = img_emb + 0.1 * np.random.rand(768).astype(np.float32)
    
    traditional_score = calculate_clip_score(img_emb, txt_emb)
    print(f"计算出的传统 CLIP-score: {traditional_score:.2f}")

    # --- 示例 2: MAG 特定 CLIP-score 计算 ---
    print("\n--- 示例 2: MAG 特定 CLIP-score ---")
    # 此示例使用预先生成的真实嵌入进行计算。
    # 确保 'Grocery' 数据集的嵌入文件存在。
    
    DATASET = "Grocery"
    ENCODER = "Qwen/Qwen2.5-VL-3B-Instruct"
    DIMENSION = 768
    TARGET_NODE = 5

    # 初始化匹配器
    matcher = MAGModalityMatcher()
    
    print(f"\n正在为数据集 '{DATASET}' 计算所有节点的 MAG 特定分数...")
    
    mag_scores = matcher.calculate_mag_clip_score(
        dataset_name=DATASET,
        encoder_name=ENCODER,
        dimension=DIMENSION
    )

    if mag_scores is not None:
        print(f"成功计算出 {len(mag_scores)} 个节点的分数。")
        print(f"平均 MAG 特定 CLIP-score: {np.mean(mag_scores):.2f}")
        print(f"前5个节点的分数: {np.round(mag_scores[:5], 2)}")
    else:
        print("无法计算 MAG 特定分数。请检查嵌入文件是否存在并且图结构是否正确。")