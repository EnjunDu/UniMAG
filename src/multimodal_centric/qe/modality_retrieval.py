# -*- coding: utf-8 -*-
"""
模态检索 (Modality Retrieval) 评估器

此模块定义了 RetrievalEvaluator 类，用于执行模态检索评估任务。
它遵循 "运行-训练-评估" 架构，并被 `run.py` 调用。
核心功能是评估图增强嵌入在检索任务上的性能。
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader

class RetrievalEvaluator:
    """
    负责执行模态检索评估任务。
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        """
        初始化模态检索评估器。

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
        
        # 从配置中获取评估参数
        self.top_k_list = self.config.get('evaluation', {}).get('top_k', [1, 5, 10])

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
        执行完整的图到文和文到图检索评估。

        Returns:
            Dict[str, float]: 包含评估指标的字典 (MRR, Hits@K)。
        """
        print("--- 开始模态检索评估 ---")
        
        enhanced_embeddings = self._get_enhanced_embeddings()
        if enhanced_embeddings is None:
            print("评估失败，无法获取增强嵌入。")
            return {"error": "Failed to get enhanced embeddings."}

        image_embeds, text_embeds = enhanced_embeddings
        
        # 计算文到图检索指标
        print("正在执行文到图 (Text-to-Image) 检索...")
        t2i_metrics = self._calculate_retrieval_metrics(text_embeds, image_embeds)
        
        # 计算图到文检索指标
        print("正在执行图到文 (Image-to-Text) 检索...")
        i2t_metrics = self._calculate_retrieval_metrics(image_embeds, text_embeds)
        
        results = {
            "text_to_image": t2i_metrics,
            "image_to_text": i2t_metrics
        }
        
        print("--- 模态检索评估完成 ---")
        return results

    def _calculate_retrieval_metrics(self, queries: np.ndarray, candidates: np.ndarray) -> Dict[str, float]:
        """
        计算检索任务的核心指标 (MRR, Hits@K)。

        Args:
            queries (np.ndarray): 查询嵌入 (N x D)。
            candidates (np.ndarray): 候选嵌入池 (N x D)。

        Returns:
            Dict[str, float]: 包含计算出的指标的字典。
        """
        num_queries = queries.shape[0]
        ranks = np.zeros(num_queries)
        
        # 计算所有查询与所有候选之间的相似度矩阵
        query_tensor = F.normalize(torch.from_numpy(queries).float(), p=2, dim=1)
        candidate_tensor = F.normalize(torch.from_numpy(candidates).float(), p=2, dim=1)
        sim_matrix = torch.matmul(query_tensor, candidate_tensor.t()).cpu().numpy()

        for i in range(num_queries):
            # 对相似度进行排序，找到真实匹配项的排名
            # 真实匹配项是索引为 i 的候选
            sorted_indices = np.argsort(-sim_matrix[i, :])
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks[i] = rank
            
        # 计算 MRR
        mrr = np.mean(1.0 / ranks)
        
        # 计算 Hits@K
        hits_at_k = {}
        for k in self.top_k_list:
            hits = np.sum(ranks <= k)
            hits_at_k[f"Hits@{k}"] = hits / num_queries
            
        metrics = {"MRR": mrr, **hits_at_k}
        return {k: float(v) for k, v in metrics.items()}