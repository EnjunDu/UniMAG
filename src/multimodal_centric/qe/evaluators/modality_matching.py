# -*- coding: utf-8 -*-
"""
模态匹配 (Modality Matching) 评估器

此模块定义了 MatchingEvaluator 类，用于执行模态匹配评估任务。
它继承自 BaseEvaluator，并实现了具体的评估逻辑。
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Dict, Any

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.evaluators.base_evaluator import BaseEvaluator

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


class MatchingEvaluator(BaseEvaluator):
    """
    负责执行模态匹配评估任务。
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        """
        初始化模态匹配评估器。
        """
        super().__init__(config, gnn_model)

    def evaluate(self) -> Dict[str, float]:
        """
        执行完整的评估流程。
        """
        enhanced_embeddings = self._get_enhanced_embeddings()
        if enhanced_embeddings is None:
            print("Error: Failed to get enhanced embeddings.")
            return {"error": "Failed to get enhanced embeddings."}

        enhanced_image_embeddings, enhanced_text_embeddings = enhanced_embeddings
        
        scores = [
            calculate_clip_score(img, txt) 
            for img, txt in zip(enhanced_image_embeddings, enhanced_text_embeddings)
        ]
        
        valid_scores = [s for s in scores if s is not None]
        num_invalid = len(scores) - len(valid_scores)
        
        if not valid_scores:
            print("Error: No valid samples to score.")
            return {"error": "No valid samples to score."}

        mean_score = np.mean(valid_scores)
        
        print(f"Evaluation completed. The average CLIP-score on {len(valid_scores)} valid samples is: {mean_score:.4f}")
        if num_invalid > 0:
            print(f"  (Warning: {num_invalid} zero-vector samples were ignored during evaluation)")
        
        return {"mean_clip_score": float(mean_score)}