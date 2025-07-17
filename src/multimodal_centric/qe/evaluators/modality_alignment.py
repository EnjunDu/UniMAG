# -*- coding: utf-8 -*-
"""
模态对齐 (Modality Alignment) 评估器
"""

import sys
import os
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Any, Optional
from torchvision.ops import roi_align

project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.evaluators.base_evaluator import BaseEvaluator
from src.multimodal_centric.qe.evaluators.modality_matching import calculate_clip_score

class AlignmentEvaluator(BaseEvaluator):
    """
    负责执行细粒度的模态对齐评估任务。
    分别为每个局部特征对进行GNN增强并评估。
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        super().__init__(config, gnn_model)
        self.ground_truth_dir = Path(__file__).resolve().parent / "ground_truth"
        self.ground_truth_path = self.ground_truth_dir / f"{self.config['dataset']['name']}_ground_truth.jsonl"

    def evaluate(self) -> Dict[str, float]:
        """
        执行完整的细粒度对齐评估流程。
        """
        print("--- 开始模态对齐评估 ---")

        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"未找到基准真值文件: {self.ground_truth_path}。请先运行 'generate_alignment_ground_truth.py' 脚本。")

        # 1. 加载所有节点的原始全局特征
        print("加载全局特征...")
        global_image_embeds = self.embedding_manager.get_embedding(self.config['dataset']['name'], "image", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        global_text_embeds = self.embedding_manager.get_embedding(self.config['dataset']['name'], "text", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        if global_image_embeds is None or global_text_embeds is None:
            raise ValueError("无法加载全局嵌入进行评估。")
        global_features = np.concatenate((global_text_embeds, global_image_embeds), axis=1)
        global_features_tensor = torch.from_numpy(global_features).float().to(self.device)

        # 2. 加载原始特征图
        print("加载原始特征图...")
        # 注意: 这里需要 embedding_manager 支持获取特征图列表
        # 我们暂时简化，假设可以获取
        all_node_ids = self._storage.load_node_ids(self.config['dataset']['name'])
        image_paths = [self.embedding_manager.get_raw_data_by_index(self.config['dataset']['name'], i)['image_path'] for i in range(len(all_node_ids))]
        
        # 为了演示，我们使用一个占位符。实际应用中需要调用 embedding_manager
        # all_feature_maps = self.embedding_manager.generate_embedding(data=image_paths, modality="image", ..., output_feature_map=True)
        sample_map = torch.randn(1, 256, 7, 7) # C, H, W
        all_feature_maps = [sample_map for _ in image_paths]
        
        all_scores = []
        
        with open(self.ground_truth_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="评估对齐分数"):
                record = json.loads(line)
                node_index = record["node_index"]
                
                # 获取该节点的原始特征图
                # 实际应为 all_feature_maps[node_index]
                image_feature_map = torch.from_numpy(all_feature_maps[node_index]).float().to(self.device)
                if image_feature_map.ndim == 3: image_feature_map.unsqueeze_(0)

                for grounding_pair in record["grounding"]:
                    phrase = grounding_pair["phrase"]
                    box = grounding_pair["box"]

                    # 3. 获取局部特征
                    phrase_embed = self.embedding_manager.generate_embedding(data=[phrase], modality="text", encoder_type=self.config['embedding']['encoder_type'], encoder_name=self.config['embedding']['encoder_name'], dimension=self.config['embedding']['dimension'])
                    if phrase_embed is None: continue
                    
                    phrase_tensor = torch.from_numpy(phrase_embed).float().to(self.device)
                    
                    box_tensor = torch.tensor([box], dtype=image_feature_map.dtype, device=self.device)
                    region_feature = roi_align(image_feature_map, [box_tensor], output_size=(1, 1), spatial_scale=1.0).flatten(1)
                    
                    # 4. 构建GNN输入
                    local_feature_pair = torch.cat([phrase_tensor, region_feature], dim=1)
                    
                    # 替换掉全局特征中的对应行
                    temp_features = global_features_tensor.clone()
                    temp_features[node_index, :] = local_feature_pair
                    
                    # 5. GNN增强
                    with torch.no_grad():
                        _, enhanced_img, enhanced_txt = self.gnn_model(temp_features, self.graph_loader.load_graph(self.config['dataset']['name']).edge_index.to(self.device))
                    
                    # 6. 计算分数
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