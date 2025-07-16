# -*- coding: utf-8 -*-
"""
模态匹配 (Modality Matching) 评估器

此模块定义了 MatchingEvaluator 类，用于执行模态匹配评估任务。
它遵循 "运行-训练-评估" 架构，并被 `run.py` 调用。
核心功能是计算图增强后的平均 CLIP-score。
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader

def calculate_clip_score(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray
) -> float:
    """
    计算并返回一对图像和文本嵌入之间的 CLIP-score。

    Args:
        image_embedding (np.ndarray): 单个图像的嵌入向量 (1D array)。
        text_embedding (np.ndarray): 单个文本的嵌入向量 (1D array)。

    Returns:
        float: 计算出的 CLIP-score，范围在 -100 到 100 之间。
    """
    image_embedding = np.asarray(image_embedding)
    text_embedding = np.asarray(text_embedding)

    # 归一化
    image_embedding_norm = image_embedding / np.linalg.norm(image_embedding)
    text_embedding_norm = text_embedding / np.linalg.norm(text_embedding)

    similarity = np.dot(image_embedding_norm, text_embedding_norm)
    clip_score = 100.0 * similarity

    return clip_score


class MatchingEvaluator:
    """
    负责执行模态匹配评估任务。
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        """
        初始化模态匹配评估器。

        Args:
            config (Dict[str, Any]): 包含任务和数据集配置的字典。
            gnn_model (torch.nn.Module): 一个已经训练好的、可用于评估的GNN模型。
        """
        self.config = config
        self.gnn_model = gnn_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_model.to(self.device)

        # 从配置中获取嵌入的基础路径
        base_path = self.config.get('dataset', {}).get('data_root')
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        self.graph_loader = GraphLoader(config=self.config)

    def _get_enhanced_embeddings(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        使用GNN模型计算整个图的邻域增强嵌入。

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]:
                一个包含 (所有节点的增强图像嵌入, 所有节点的增强文本嵌入) 的元组。
                如果无法获取任何所需数据，则返回 None。
        """
        dataset_name = self.config['dataset']['name']
        encoder_name = self.config['embedding']['encoder_name']
        dimension = self.config['embedding']['dimension']

        # 1. 加载原始嵌入
        image_embeddings = self.embedding_manager.get_embedding(
            dataset_name, "image", encoder_name, dimension
        )
        text_embeddings = self.embedding_manager.get_embedding(
            dataset_name, "text", encoder_name, dimension
        )

        if image_embeddings is None or text_embeddings is None:
            print("错误: 无法加载基础嵌入，无法进行邻域增强。")
            return None

        # 2. 加载图结构
        try:
            graph_data = self.graph_loader.load_graph(dataset_name)
            edge_index = graph_data.edge_index.to(self.device)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: 无法为数据集 '{dataset_name}' 加载图结构: {e}")
            return None

        # 3. 准备GNN输入
        # 注意：这里的拼接方式需要与训练时保持一致
        multimodal_features = np.concatenate((text_embeddings, image_embeddings), axis=1)
        features_tensor = torch.from_numpy(multimodal_features).float().to(self.device)

        # 4. 使用GNN模型进行前向传播
        self.gnn_model.eval() # 确保模型处于评估模式
        with torch.no_grad():
            # 假设模型返回 (out, out_v, out_t)
            _, enhanced_image_embeddings, enhanced_text_embeddings = self.gnn_model(features_tensor, edge_index)

        return enhanced_image_embeddings.cpu().numpy(), enhanced_text_embeddings.cpu().numpy()

    def evaluate(self) -> Dict[str, float]:
        """
        执行完整的评估流程。

        Returns:
            Dict[str, float]: 包含评估结果的字典，例如 {'mean_clip_score': 75.4}。
        """
        print("--- 开始模态匹配评估 ---")
        
        enhanced_embeddings = self._get_enhanced_embeddings()
        if enhanced_embeddings is None:
            print("评估失败，无法获取增强嵌入。")
            return {"error": "Failed to get enhanced embeddings."}

        enhanced_image_embeddings, enhanced_text_embeddings = enhanced_embeddings
        
        num_nodes = enhanced_image_embeddings.shape[0]
        scores = np.zeros(num_nodes)

        print(f"正在为 {num_nodes} 个节点计算 CLIP-score...")
        for i in range(num_nodes):
            scores[i] = calculate_clip_score(
                enhanced_image_embeddings[i],
                enhanced_text_embeddings[i]
            )
        
        mean_score = np.mean(scores)
        print(f"评估完成。平均 CLIP-score: {mean_score:.4f}")
        
        return {"mean_clip_score": float(mean_score)}