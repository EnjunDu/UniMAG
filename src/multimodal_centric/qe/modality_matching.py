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
from typing import Optional, Tuple, Union, Dict, Any
from torch_geometric.nn import GCNConv

# 将项目根目录添加到Python路径中，以方便模块导入
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader
from src.model.models import GCN, GAT, GraphSAGE

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
    def __init__(self, gnn_model: torch.nn.Module, config: Optional[Dict[str, Any]] = None):
        """
        初始化 MAG 特定的模态匹配器。

        Args:
            gnn_model (torch.nn.Module): 一个已经实例化的GNN模型 (例如 GCN, GAT)。
            config (Optional[Dict[str, Any]]): 包含任务和数据集配置的字典。
        """
        self.config = config
        # 从配置中获取嵌入的基础路径，如果存在的话
        base_path = self.config.get('dataset', {}).get('data_root') if self.config else None
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        self.graph_loader = GraphLoader(config=self.config)
        self.gnn_model = gnn_model

    def _get_graph_structure(self, dataset_name: str) -> Optional[torch.Tensor]:
        """
        使用 GraphLoader 获取真实的图结构。

        Args:
            dataset_name (str): 数据集名称。

        Returns:
            Optional[torch.Tensor]: 图的 edge_index 张量，如果加载失败则返回 None。
        """
        try:
            graph_data = self.graph_loader.load_graph(dataset_name)
            return graph_data.edge_index
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: 无法为数据集 '{dataset_name}' 加载图结构: {e}")
            return None

    def get_neighbor_enhanced_embedding(
        self,
        dataset_name: str,
        encoder_name: str,
        dimension: Optional[int] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        使用传入的GNN模型计算整个图的邻域增强嵌入。

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

        # 将图像和文本嵌入拼接成GNN的输入
        multimodal_features = np.concatenate((image_embeddings, text_embeddings), axis=1)
        features_tensor = torch.from_numpy(multimodal_features).float()

        # 使用传入的GNN模型进行前向传播
        # model(x, edge_index) 返回 (out, out_v, out_t)
        _ , enhanced_image_embeddings, enhanced_text_embeddings = self.gnn_model(features_tensor, edge_index)

        return enhanced_image_embeddings.detach().numpy(), enhanced_text_embeddings.detach().numpy()

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

    # --- 固定配置 ---
    DATASET_NAME = "Grocery"
    ENCODER_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    DIMENSION = 768
    TARGET_NODE = 1
    # 服务器上的固定数据根目录
    DATA_ROOT = "/home/ai/MMAG"

    # --- 初始化 ---
    config = {
        "dataset": {
            "name": DATASET_NAME,
            "data_root": DATA_ROOT
        }
    }
    
    # 1. 在外部实例化你想要的GNN模型
    print("正在初始化GAT模型...")
    # 注意：这里的in_dim需要是拼接后特征的维度
    gat_model = GAT(
        in_dim=DIMENSION * 2,
        hidden_dim=128,       # GAT的隐藏维度可以自定义
        num_layers=2,
        heads=4,
        dropout=0.5
    )

    # 2. 将模型实例注入到Matcher中
    matcher = MAGModalityMatcher(gnn_model=gat_model, config=config)

    # --- 加载数据 ---
    print(f"正在从数据集 '{DATASET_NAME}' 加载嵌入...")
    image_embeds = matcher.embedding_manager.get_embedding(
        DATASET_NAME, "image", ENCODER_NAME, DIMENSION
    )
    text_embeds = matcher.embedding_manager.get_embedding(
        DATASET_NAME, "text", ENCODER_NAME, DIMENSION
    )

    if image_embeds is None or text_embeds is None:
        print(f"错误: 无法加载 '{DATASET_NAME}' 数据集的嵌入。")
        print("请确保以下文件存在:")
        print(f"- {DATA_ROOT}/{DATASET_NAME}/image_features/{DATASET_NAME}_image_{ENCODER_NAME}_{DIMENSION}d.npy")
        print(f"- {DATA_ROOT}/{DATASET_NAME}/text_features/{DATASET_NAME}_text_{ENCODER_NAME}_{DIMENSION}d.npy")
        sys.exit(1)

    print(f"成功加载 {len(image_embeds)} 个节点的嵌入。")

    # --- 示例 1: 传统 CLIP-score 计算 (单个节点，无图上下文) ---
    print("\n--- 示例 1: 传统 CLIP-score (节点自身模态对齐) ---")
    
    traditional_score = None
    if TARGET_NODE < len(image_embeds):
        node_img_emb = image_embeds[TARGET_NODE]
        node_txt_emb = text_embeds[TARGET_NODE]
        
        traditional_score = calculate_clip_score(node_img_emb, node_txt_emb)
        print(f"节点 {TARGET_NODE} 的传统 CLIP-score (无图上下文): {traditional_score:.2f}")
    else:
        print(f"错误: 目标节点 {TARGET_NODE} 超出范围 (0-{len(image_embeds)-1})。")

    # --- 示例 2: MAG 特定 CLIP-score 计算 (图上下文增强) ---
    print("\n--- 示例 2: MAG 特定 CLIP-score (图上下文增强) ---")
    
    print("正在计算所有节点的 MAG 特定分数...")
    mag_scores = matcher.calculate_mag_clip_score(
        dataset_name=DATASET_NAME,
        encoder_name=ENCODER_NAME,
        dimension=DIMENSION
    )

    if mag_scores is not None:
        print(f"成功计算出 {len(mag_scores)} 个节点的 MAG 特定分数。")
        print(f"平均 MAG 特定 CLIP-score: {np.mean(mag_scores):.2f}")
        print(f"前5个节点的分数: {np.round(mag_scores[:5], 2)}")
        
        # 对比分析
        if traditional_score is not None and TARGET_NODE < len(mag_scores):
            enhanced_score = mag_scores[TARGET_NODE]
            print(f"\n--- 对比分析 (节点 {TARGET_NODE}) ---")
            print(f"传统方法分数: {traditional_score:.2f}")
            print(f"MAG 增强分数: {enhanced_score:.2f}")
            print(f"图上下文增益: {enhanced_score - traditional_score:.2f}")
    else:
        print("无法计算 MAG 特定分数。请检查图结构加载是否正确。")