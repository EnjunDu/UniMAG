# -*- coding: utf-8 -*-
"""
模态检索 (Modality Retrieval)

此模块实现了在 "Tasks_and_arrangements.md" 中定义的两种模态检索方法：
1. 传统方法: 使用一个查询嵌入从一个候选池中检索最相关的项目。
2. MAG 特定方法: 使用一个节点的邻域增强嵌入在图中检索最相关的节点。
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, Union, Dict, Any, List
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# 将项目根目录添加到Python路径中，以方便模块导入
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader

def retrieve_traditional(
    query_embedding: np.ndarray,
    candidate_pool: np.ndarray,
    top_k: int = 10
) -> np.ndarray:
    """
    计算查询嵌入与候选池中所有嵌入的余弦相似度，并返回top_k个最相似的候选索引。

    Args:
        query_embedding (np.ndarray): 查询嵌入 (1D array)。
        candidate_pool (np.ndarray): 候选嵌入池 (2D array, N x D)。
        top_k (int): 要返回的最相似候选的数量。

    Returns:
        np.ndarray: 排序后的top_k个候选索引。
    """
    query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0)
    pool_tensor = torch.from_numpy(candidate_pool).float()

    # 标准化嵌入
    query_tensor = F.normalize(query_tensor, p=2, dim=1)
    pool_tensor = F.normalize(pool_tensor, p=2, dim=1)

    # 计算余弦相似度
    similarity_scores = torch.matmul(query_tensor, pool_tensor.t()).squeeze(0)

    # 获取 top_k 结果
    _, top_k_indices = torch.topk(similarity_scores, k=top_k)

    return top_k_indices.numpy()


class MAGModalityRetriever:
    """
    实现 MAG 特定的、考虑图上下文的模态检索方法。
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 MAG 特定的模态检索器。

        Args:
            config (Optional[Dict[str, Any]]): 包含任务和数据集配置的字典。
        """
        self.config = config
        base_path = self.config.get('dataset', {}).get('data_root') if self.config else None
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        self.graph_loader = GraphLoader(config=self.config)
        self.gcn_layer = None

    def _initialize_gcn_layer(self, in_dim: int, out_dim: int):
        """如果需要，则初始化GCN层。"""
        if self.gcn_layer is None:
            self.gcn_layer = GCNConv(in_dim, out_dim)

    def _get_graph_structure(self, dataset_name: str) -> Optional[torch.Tensor]:
        """
        使用 GraphLoader 获取真实的图结构。
        """
        try:
            graph_data = self.graph_loader.load_graph(dataset_name)
            return graph_data.edge_index
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: 无法为数据集 '{dataset_name}' 加载图结构: {e}")
            return None

    def retrieve_mag_specific(
        self,
        dataset_name: str,
        query_node_id: int,
        encoder_name: str,
        modality: str,
        dimension: Optional[int] = None,
        top_k: int = 10
    ) -> Optional[np.ndarray]:
        """
        使用GCN增强的查询嵌入在图中检索top_k个最相似的节点。

        Args:
            dataset_name (str): 数据集名称。
            query_node_id (int): 查询节点的ID。
            encoder_name (str): 编码器模型的Hugging Face名称。
            modality (str): 要使用的嵌入类型 ('image', 'text', or 'multimodal')。
            dimension (Optional[int]): 特征维度。
            top_k (int): 要返回的最相似节点的数量。

        Returns:
            Optional[np.ndarray]: 排序后的top_k个节点ID。
        """
        # 1. 加载所有节点的嵌入
        all_embeddings = self.embedding_manager.get_embedding(
            dataset_name, modality, encoder_name, dimension
        )
        if all_embeddings is None:
            print(f"错误: 无法加载 '{modality}' 嵌入。")
            return None

        # 2. 加载图结构
        edge_index = self._get_graph_structure(dataset_name)
        if edge_index is None:
            return None

        # 3. 创建邻域增强的查询嵌入
        features_tensor = torch.from_numpy(all_embeddings).float()
        feature_dim = features_tensor.shape[1]
        self._initialize_gcn_layer(feature_dim, feature_dim)
        
        enhanced_features = self.gcn_layer(features_tensor, edge_index).detach()
        
        query_embedding_enhanced = enhanced_features[query_node_id].numpy()

        # 4. 计算相似度并检索
        # 从候选池中移除查询节点本身
        candidate_indices = np.arange(all_embeddings.shape[0])
        candidate_indices = np.delete(candidate_indices, query_node_id)
        candidate_pool = all_embeddings[candidate_indices]

        top_k_indices_in_candidates = retrieve_traditional(
            query_embedding_enhanced,
            candidate_pool,
            top_k=top_k
        )
        
        # 将候选索引映射回原始节点ID
        top_k_node_ids = candidate_indices[top_k_indices_in_candidates]

        return top_k_node_ids

if __name__ == '__main__':
    print("=== 模态检索模块使用示例 ===")

    # --- 固定配置 ---
    DATASET_NAME = "Grocery"
    ENCODER_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    DIMENSION = 768
    QUERY_NODE = 5
    MODALITY = "text"  # 'image' or 'text'
    TOP_K = 5
    # 服务器上的固定数据根目录
    DATA_ROOT = "/home/ai/MMAG"

    # --- 初始化 ---
    config = {
        "dataset": {
            "name": DATASET_NAME,
            "data_root": DATA_ROOT
        }
    }
    retriever = MAGModalityRetriever(config=config)
    
    # --- 加载数据 ---
    print(f"正在从数据集 '{DATASET_NAME}' 加载 '{MODALITY}' 嵌入...")
    all_embeddings = retriever.embedding_manager.get_embedding(
        DATASET_NAME, MODALITY, ENCODER_NAME, DIMENSION
    )

    if all_embeddings is None:
        print(f"错误: 无法加载 '{DATASET_NAME}' 的 '{MODALITY}' 嵌入。")
        print(f"请确保文件存在: {DATA_ROOT}/{DATASET_NAME}/{MODALITY}_features/{DATASET_NAME}_{MODALITY}_{ENCODER_NAME}_{DIMENSION}d.npy")
        sys.exit(1)
        
    if QUERY_NODE >= len(all_embeddings):
        print(f"错误: 查询节点 {QUERY_NODE} 超出范围 (0-{len(all_embeddings)-1})。")
        sys.exit(1)

    # --- 示例 1: 传统检索 ---
    print(f"\n--- 示例 1: 传统检索 (使用来自 '{DATASET_NAME}' 的真实数据) ---")
    query_embedding = all_embeddings[QUERY_NODE]
    
    # 从候选池中移除查询节点本身
    candidate_indices = np.arange(all_embeddings.shape[0])
    candidate_indices = np.delete(candidate_indices, QUERY_NODE)
    candidate_pool = all_embeddings[candidate_indices]

    top_k_indices_in_candidates = retrieve_traditional(
        query_embedding,
        candidate_pool,
        top_k=TOP_K
    )
    
    # 将候选索引映射回原始节点ID
    top_k_node_ids_traditional = candidate_indices[top_k_indices_in_candidates]
    print(f"传统检索 Top-{TOP_K} 节点 ID (对于节点 {QUERY_NODE}): {top_k_node_ids_traditional}")

    # --- 示例 2: MAG 特定检索 ---
    print(f"\n--- 示例 2: MAG 特定检索 ---")
    print(f"正在为数据集 '{DATASET_NAME}' 的节点 {QUERY_NODE} 进行 MAG 特定检索...")
    
    top_nodes_mag = retriever.retrieve_mag_specific(
        dataset_name=DATASET_NAME,
        query_node_id=QUERY_NODE,
        encoder_name=ENCODER_NAME,
        modality=MODALITY,
        dimension=DIMENSION,
        top_k=TOP_K
    )

    if top_nodes_mag is not None:
        print(f"成功检索到 Top-{TOP_K} 节点: {top_nodes_mag}")
        # 比较两种方法的结果
        print(f"\n--- 对比分析 (节点 {QUERY_NODE}) ---")
        print(f"传统检索结果:   {top_k_node_ids_traditional}")
        print(f"MAG 增强检索结果: {top_nodes_mag}")
    else:
        print("无法执行 MAG 特定检索。请检查嵌入文件和图结构。")