# -*- coding: utf-8 -*-
"""
模态对齐 (Modality Alignment) 评估器
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Any

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.evaluators.base_evaluator import BaseEvaluator
from src.multimodal_centric.qe.evaluators.modality_matching import calculate_clip_score

class AlignmentEvaluator(BaseEvaluator):
    """
    负责执行细粒度的模态对齐评估任务。
    该评估器依赖于一个预处理好的数据文件，该文件包含了所有必要的
    预计算特征（短语嵌入、区域嵌入等）。
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        super().__init__(config, gnn_model)
        self.preprocessed_data_dir = Path(__file__).resolve().parent.parent / "scripts" / "ground_truth"
        self.preprocessed_data_path = self.preprocessed_data_dir / f"{self.config.dataset.name}_alignment_preprocessed.pt"

    def evaluate(self) -> Dict[str, float]:
        """
        执行完整的细粒度对齐评估流程。
        """
        print("--- Start Modality Alignment Evaluation ---")

        if not self.preprocessed_data_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data file not found: {self.preprocessed_data_path}.\n"
                f"Please run 'prepare_alignment_data.py' script with '--stage all' or '--stage 2' to generate this file."
            )

        print(f"正在从 '{self.preprocessed_data_path}' 加载预处理数据...")
        preprocessed_data = torch.load(self.preprocessed_data_path)
        if not preprocessed_data:
            print("Error: Preprocessed data file is empty.")
            return {"error": "Preprocessed data file is empty."}

        print("Loading global features for GNN enhancement...")
        global_image_embeds = self.embedding_manager.get_embedding(self.config.dataset.name, "image", self.config.embedding.encoder_name, self.config.embedding.dimension)
        global_text_embeds = self.embedding_manager.get_embedding(self.config.dataset.name, "text", self.config.embedding.encoder_name, self.config.embedding.dimension)
        if global_image_embeds is None or global_text_embeds is None:
            raise ValueError("Failed to load global embeddings for evaluation.")
        
        global_features = np.concatenate((global_text_embeds, global_image_embeds), axis=1)
        global_features_tensor = torch.from_numpy(global_features).float().to(self.device)
        
        edge_index = self.graph_loader.load_graph(self.config.dataset.name).edge_index.to(self.device)

        all_scores = []
        invalid_pairs_count = 0
        
        for item in tqdm(preprocessed_data, desc="Evaluating Alignment Scores"):
            node_index = item["node_index"]
            phrase_embedding = item["phrase_embedding"]
            region_embedding = item["region_embedding"]

            # 使用 torch.isnan 和 torch.isinf 进行数值检查
            if torch.isnan(phrase_embedding).any() or torch.isinf(phrase_embedding).any() or \
               torch.isnan(region_embedding).any() or torch.isinf(region_embedding).any():
                invalid_pairs_count += 1
                continue

            phrase_embedding = phrase_embedding.to(self.device)
            region_embedding = region_embedding.to(self.device)

            # 1. 构建GNN输入
            if phrase_embedding.shape != region_embedding.shape:
                min_dim = min(phrase_embedding.shape[0], region_embedding.shape[0])
                phrase_embedding = phrase_embedding[:min_dim]
                region_embedding = region_embedding[:min_dim]

            local_feature_pair = torch.cat([phrase_embedding, region_embedding]).unsqueeze(0)
            
            expected_dim = global_features_tensor.shape[1]
            if local_feature_pair.shape[1] != expected_dim:
                print(f"Warning: Local feature dimension {local_feature_pair.shape[1]} does not match global feature dimension {expected_dim}. Skipping this pair.")
                invalid_pairs_count += 1
                continue

            # 2. 替换全局特征中的对应行
            temp_features = global_features_tensor.clone()
            temp_features[node_index, :] = local_feature_pair
            
            # 3. GNN 增强
            with torch.no_grad():
                _, enhanced_img, enhanced_txt = self.gnn_model(temp_features, edge_index)
            
            # 4. 计算分数
            enhanced_local_img = enhanced_img[node_index].cpu().numpy()
            enhanced_local_txt = enhanced_txt[node_index].cpu().numpy()
            score = calculate_clip_score(enhanced_local_img, enhanced_local_txt)
            if score is not None:
                all_scores.append(score)
            else:
                invalid_pairs_count += 1

        if invalid_pairs_count > 0:
            print(f"Warning: During evaluation, {invalid_pairs_count} feature pairs were skipped due to invalid values or dimension mismatches.")

        if not all_scores:
            print("Error: Failed to calculate any valid alignment scores.")
            return {"error": "No valid alignment scores could be calculated."}

        mean_alignment_score = np.nanmean(all_scores)
        print(f"Evaluation completed. The average alignment score on {len(all_scores)} valid local feature pairs is: {mean_alignment_score:.4f}")
        
        return {"mean_alignment_score": float(mean_alignment_score)}