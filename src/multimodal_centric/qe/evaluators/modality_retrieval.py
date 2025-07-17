# -*- coding: utf-8 -*-
"""
模态检索 (Modality Retrieval) 评估器

此模块定义了 RetrievalEvaluator 类，用于执行模态检索评估任务。
它继承自 BaseEvaluator，并实现了具体的评估逻辑。
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any
from collections import OrderedDict

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.evaluators.base_evaluator import BaseEvaluator
from src.multimodal_centric.qe.models.retrieval_model import TwoTowerModel

class RetrievalEvaluator(BaseEvaluator):
    """
    负责执行模态检索评估任务。
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module, retrieval_model: TwoTowerModel):
        """
        初始化模态检索评估器。

        Args:
            config (Dict[str, Any]): 包含任务和数据集配置的字典。
            gnn_model (torch.nn.Module): 第一阶段训练好的GNN模型。
            retrieval_model (TwoTowerModel): 第二阶段训练好的双塔检索模型。
        """
        super().__init__(config, gnn_model)
        self.retrieval_model = retrieval_model.to(self.device)
        self.retrieval_model.eval()
        self.top_k_list = self.config.get('evaluation', {}).get('top_k', [1, 5, 10])

    def evaluate(self) -> Dict[str, Any]:
        """
        执行完整的图到文和文到图检索评估。
        """
        print("--- 开始模态检索评估 ---")
        
        # 1. 获取第一阶段的增强嵌入
        enhanced_embeddings = self._get_enhanced_embeddings()
        if enhanced_embeddings is None:
            print("评估失败，无法获取GNN增强嵌入。")
            return {"error": "Failed to get GNN enhanced embeddings."}
        enhanced_text_embeds, enhanced_image_embeds = enhanced_embeddings
        
        enhanced_text_embeds = torch.from_numpy(enhanced_text_embeds).float().to(self.device)
        enhanced_image_embeds = torch.from_numpy(enhanced_image_embeds).float().to(self.device)

        # 2. 使用第二阶段的双塔模型获取最终的检索嵌入
        with torch.no_grad():
            final_text_embeds = self.retrieval_model.encode_text(enhanced_text_embeds).cpu().numpy()
            final_image_embeds = self.retrieval_model.encode_image(enhanced_image_embeds).cpu().numpy()

        print("正在执行文到图 (Text-to-Image) 检索...")
        t2i_metrics = self._calculate_retrieval_metrics(final_text_embeds, final_image_embeds)
        
        print("正在执行图到文 (Image-to-Text) 检索...")
        i2t_metrics = self._calculate_retrieval_metrics(final_image_embeds, final_text_embeds)
        
        results = {
            "text_to_image": t2i_metrics,
            "image_to_text": i2t_metrics
        }
        
        print("--- 模态检索评估完成 ---")
        return results

    def _calculate_retrieval_metrics(self, queries: np.ndarray, candidates: np.ndarray) -> Dict[str, float]:
        """
        计算检索任务的核心指标 (MRR, Hits@K)。
        """
        num_queries = queries.shape[0]
        ranks = np.zeros(num_queries)
        
        query_tensor = F.normalize(torch.from_numpy(queries).float(), p=2, dim=1)
        candidate_tensor = F.normalize(torch.from_numpy(candidates).float(), p=2, dim=1)
        sim_matrix = torch.matmul(query_tensor, candidate_tensor.t()).cpu().numpy()

        for i in range(num_queries):
            sorted_indices = np.argsort(-sim_matrix[i, :])
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks[i] = rank
            
        mrr = np.mean(1.0 / ranks)
        
        metrics = OrderedDict()
        metrics["MRR"] = float(mrr)
        
        for k in self.top_k_list:
            hits = np.sum(ranks <= k)
            metrics[f"Hits@{k}"] = float(hits / num_queries)
            
        return metrics