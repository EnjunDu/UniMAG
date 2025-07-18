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
from collections import defaultdict

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
        self.preprocessed_data_path = self.preprocessed_data_dir / f"{self.config['dataset']['name']}_alignment_preprocessed.pt"

    def evaluate(self) -> Dict[str, float]:
        """
        执行完整的细粒度对齐评估流程。
        """
        print("--- 开始模态对齐评估 ---")

        if not self.preprocessed_data_path.exists():
            raise FileNotFoundError(
                f"未找到预处理数据文件: {self.preprocessed_data_path}。\n"
                f"请先运行 'prepare_alignment_data.py' 脚本并使用 '--stage all' 或 '--stage 2' 来生成该文件。"
            )

        print(f"正在从 '{self.preprocessed_data_path}' 加载预处理数据...")
        preprocessed_data = torch.load(self.preprocessed_data_path)
        if not preprocessed_data:
            print("错误: 预处理数据文件为空。")
            return {"error": "Preprocessed data file is empty."}

        print("加载全局特征以用于GNN增强...")
        global_image_embeds = self.embedding_manager.get_embedding(self.config['dataset']['name'], "image", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        global_text_embeds = self.embedding_manager.get_embedding(self.config['dataset']['name'], "text", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        if global_image_embeds is None or global_text_embeds is None:
            raise ValueError("无法加载全局嵌入进行评估。")
        
        global_features = np.concatenate((global_text_embeds, global_image_embeds), axis=1)
        global_features_tensor = torch.from_numpy(global_features).float().to(self.device)
        
        edge_index = self.graph_loader.load_graph(self.config['dataset']['name']).edge_index.to(self.device)

        all_scores = []
        
        for item in tqdm(preprocessed_data, desc="评估对齐分数"):
            node_index = item["node_index"]
            phrase_embedding = item["phrase_embedding"].to(self.device)
            region_embedding = item["region_embedding"].to(self.device)

            # 1. 构建GNN输入
            # 确保维度匹配
            if phrase_embedding.shape != region_embedding.shape:
                min_dim = min(phrase_embedding.shape[0], region_embedding.shape[0])
                phrase_embedding = phrase_embedding[:min_dim]
                region_embedding = region_embedding[:min_dim]

            local_feature_pair = torch.cat([phrase_embedding, region_embedding]).unsqueeze(0)
            
            expected_dim = global_features_tensor.shape[1]
            if local_feature_pair.shape[1] != expected_dim:
                print(f"警告: 局部特征维度 {local_feature_pair.shape[1]} 与全局特征维度 {expected_dim} 不匹配。跳过此对。")
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

        if not all_scores:
            print("错误: 未能计算任何有效的对齐分数。")
            return {"error": "No valid alignment scores could be calculated."}

        mean_alignment_score = np.mean(all_scores)
        print(f"评估完成。在 {len(all_scores)} 个局部特征对上计算的平均对齐分数: {mean_alignment_score:.4f}")
        
        return {"mean_alignment_score": float(mean_alignment_score)}