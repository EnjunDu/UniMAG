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
) -> Optional[float]:
    """
    计算并返回一对图像和文本嵌入之间的 CLIP-score。
    如果任一嵌入为零向量，则返回 None。
    """
    img_norm = np.linalg.norm(image_embedding)
    txt_norm = np.linalg.norm(text_embedding)

    if img_norm == 0 or txt_norm == 0:
        return None

    image_embedding = image_embedding / img_norm
    text_embedding = text_embedding / txt_norm
    return 100.0 * np.dot(image_embedding, text_embedding)


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

        base_path = self.config.get('dataset', {}).get('data_root')
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        self.graph_loader = GraphLoader(config=self.config)

    def _get_enhanced_embeddings(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        使用GNN模型计算整个图的邻域增强嵌入。
        """
        dataset_name = self.config['dataset']['name']
        encoder_name = self.config['embedding']['encoder_name']
        dimension = self.config['embedding']['dimension']

        image_embeddings = self.embedding_manager.get_embedding(
            dataset_name, "image", encoder_name, dimension
        )
        text_embeddings = self.embedding_manager.get_embedding(
            dataset_name, "text", encoder_name, dimension
        )

        if image_embeddings is None or text_embeddings is None:
            print("错误: 无法加载基础嵌入。")
            return None

        try:
            graph_data = self.graph_loader.load_graph(dataset_name)
            edge_index = graph_data.edge_index.to(self.device)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: 无法加载图结构: {e}")
            return None

        multimodal_features = np.concatenate((text_embeddings, image_embeddings), axis=1)
        features_tensor = torch.from_numpy(multimodal_features).float().to(self.device)

        self.gnn_model.eval()
        with torch.no_grad():
            _, enhanced_image_embeddings, enhanced_text_embeddings = self.gnn_model(features_tensor, edge_index)

        return enhanced_image_embeddings.cpu().numpy(), enhanced_text_embeddings.cpu().numpy()

    def evaluate(self) -> Dict[str, float]:
        """
        执行完整的评估流程。
        """
        print("--- 开始模态匹配评估 ---")
        
        enhanced_embeddings = self._get_enhanced_embeddings()
        if enhanced_embeddings is None:
            print("评估失败，无法获取增强嵌入。")
            return {"error": "Failed to get enhanced embeddings."}

        enhanced_image_embeddings, enhanced_text_embeddings = enhanced_embeddings
        
        scores = [
            calculate_clip_score(img, txt) 
            for img, txt in zip(enhanced_image_embeddings, enhanced_text_embeddings)
        ]
        
        valid_scores = [s for s in scores if s is not None]
        num_invalid = len(scores) - len(valid_scores)
        
        if not valid_scores:
            print("错误: 没有任何有效的样本可以计算分数。")
            return {"error": "No valid samples to score."}

        mean_score = np.mean(valid_scores)
        
        print(f"评估完成。在 {len(valid_scores)} 个有效样本上计算的平均 CLIP-score: {mean_score:.4f}")
        if num_invalid > 0:
            print(f"  (警告: 在评估中忽略了 {num_invalid} 个零向量样本)")
        
        return {"mean_clip_score": float(mean_score)}