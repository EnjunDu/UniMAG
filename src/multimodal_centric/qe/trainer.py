# -*- coding: utf-8 -*-
"""
GNN 模型训练器

该模块负责：
1.  第一阶段通用训练：使用InfoNCE损失进行对比学习。
2.  第二阶段微调：为特定下游任务（如检索）使用专门的损失函数进行微调。
3.  根据任务类型，自动选择合适的训练或加载策略。
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import sys
import os
import yaml
import numpy as np
from tqdm import tqdm
from typing import Optional

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader
from src.model.models import GCN, GraphSAGE, GAT, MLP
from src.model.MMGCN import Net as MMGCN
from src.model.MGAT import MGAT as MGAT_model
from src.model.REVGAT import RevGAT

def calculate_clip_score(image_embedding: np.ndarray, text_embedding: np.ndarray) -> Optional[float]:
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

class GNNTrainer:
    """
    负责训练、保存和加载用于QE任务的GNN模型。
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dataset_name = self.config['dataset']['name']
        self.gnn_model_name = self.config['model']['name']
        
        # 通用训练参数
        train_params = self.config.get('training', {})
        self.epochs = train_params.get('epochs', 50)
        self.lr = train_params.get('lr', 0.001)
        self.patience = train_params.get('patience', 10)
        self.val_ratio = train_params.get('val_ratio', 0.1)
        self.tau = train_params.get('tau', 0.07)
        
        # 微调专用参数
        finetune_params = self.config.get('finetune', {})
        self.ft_epochs = finetune_params.get('epochs', 20)
        self.ft_lr = finetune_params.get('lr', 1e-4)
        self.ft_margin = finetune_params.get('margin', 0.1)

        self.base_dir = Path(__file__).resolve().parent
        self.model_save_dir = self.base_dir / "trained_models" / self.dataset_name / self.gnn_model_name
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        base_path = self.config.get('dataset', {}).get('data_root')
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        self.graph_loader = GraphLoader(config=self.config)

    def _init_model(self) -> torch.nn.Module:
        model_params = self.config['model']['params']
        encoder_name = self.config['embedding']['encoder_name']
        dimension = self.config['embedding']['dimension']
        sample_text_embed = self.embedding_manager.get_embedding(self.dataset_name, "text", encoder_name, dimension)
        if sample_text_embed is None: raise ValueError("无法加载嵌入以确定模型输入维度。")
        in_dim = sample_text_embed.shape[1] * 2
        
        model_params_with_in_dim = {'in_dim': in_dim, **model_params}

        if self.gnn_model_name == 'GCN': model = GCN(**model_params_with_in_dim)
        elif self.gnn_model_name == 'GAT': model = GAT(**model_params_with_in_dim)
        elif self.gnn_model_name == 'GraphSAGE': model = GraphSAGE(**model_params_with_in_dim)
        elif self.gnn_model_name == 'MLP': model = MLP(**model_params_with_in_dim)
        elif self.gnn_model_name == "RevGAT":
            model = RevGAT(in_feats=in_dim, **model_params)
        elif self.gnn_model_name == "MMGCN":
            v_dim = sample_text_embed.shape[1]
            num_nodes = self.graph_loader.load_graph(self.dataset_name).num_nodes()
            model = MMGCN(v_feat_dim=in_dim, t_feat_dim=in_dim, num_nodes=num_nodes, v_dim=v_dim, **model_params)
        elif self.gnn_model_name == "MGAT":
            v_dim = sample_text_embed.shape[1]
            num_nodes = self.graph_loader.load_graph(self.dataset_name).num_nodes()
            model = MGAT_model(v_feat_dim=in_dim, t_feat_dim=in_dim, num_nodes=num_nodes, v_dim=v_dim, **model_params)
        else:
            raise ValueError(f"未知的GNN模型: {self.gnn_model_name}")
        return model

    def _calculate_infonce_loss(self, query, positive_key, all_keys) -> torch.Tensor:
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)
        all_keys = F.normalize(all_keys, p=2, dim=1)
        logits = torch.matmul(query, all_keys.T)
        logits /= self.tau
        labels = torch.arange(len(query), device=self.device)
        return F.cross_entropy(logits, labels)

    def _get_data_splits(self, edge_index):
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)
        val_size = int(num_edges * self.val_ratio)
        val_edges = edge_index[:, perm[:val_size]]
        train_edges = edge_index[:, perm[val_size:]]
        return train_edges, val_edges

    def _train_generic(self, model, save_path):
        print(f"开始为数据集 '{self.dataset_name}' 进行第一阶段通用训练...")
        
        graph = self.graph_loader.load_graph(self.dataset_name)
        train_edge_index, val_edge_index = self._get_data_splits(graph.edge_index)
        train_edge_index = train_edge_index.to(self.device)
        val_edge_index = val_edge_index.to(self.device)

        encoder_name = self.config['embedding']['encoder_name']
        dimension = self.config['embedding']['dimension']
        image_embeds = self.embedding_manager.get_embedding(self.dataset_name, "image", encoder_name, dimension)
        text_embeds = self.embedding_manager.get_embedding(self.dataset_name, "text", encoder_name, dimension)
        
        scores = [calculate_clip_score(img, txt) for img, txt in zip(image_embeds, text_embeds)]
        valid_scores = [s for s in scores if s is not None]
        num_invalid = len(scores) - len(valid_scores)
        
        print(f"基线 CLIP-Score (无GNN增强): {np.mean(valid_scores):.4f}")
        if num_invalid > 0:
            print(f"  (警告: 在计算基线时忽略了 {num_invalid} 个零向量样本)")

        features = torch.from_numpy(np.concatenate((text_embeds, image_embeds), axis=1)).float().to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(self.epochs), desc="GNN Generic Training"):
            model.train()
            optimizer.zero_grad()
            _, enhanced_img, enhanced_txt = model(features, train_edge_index)
            loss = self._calculate_infonce_loss(enhanced_img, enhanced_txt, enhanced_txt)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                _, val_img, val_txt = model(features, val_edge_index)
                val_loss = self._calculate_infonce_loss(val_img, val_txt, val_txt)

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                tqdm.write(f"验证损失连续 {self.patience} 个轮次没有改善，提前停止训练。")
                break
        
        print("第一阶段训练完成。加载性能最佳的模型。")
        model.load_state_dict(torch.load(save_path))

    def train_or_load_generic_model(self) -> torch.nn.Module:
        generic_model_path = self.model_save_dir / "model_generic.pt"
        model = self._init_model()
        if generic_model_path.exists():
            print(f"加载通用预训练模型: {generic_model_path}")
            model.load_state_dict(torch.load(generic_model_path, map_location=self.device))
        else:
            print("未找到通用预训练模型，开始第一阶段训练...")
            self._train_generic(model, generic_model_path)
        return model.to(self.device)

    def _fine_tune_for_retrieval(self, base_model: torch.nn.Module) -> torch.nn.Module:
        print("开始第二阶段微调：检索任务")
        
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=self.ft_margin)
        
        image_embeds = self.embedding_manager.get_embedding(self.dataset_name, "image", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        text_embeds = self.embedding_manager.get_embedding(self.dataset_name, "text", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        features = torch.from_numpy(np.concatenate((text_embeds, image_embeds), axis=1)).float().to(self.device)
        graph = self.graph_loader.load_graph(self.dataset_name)
        edge_index = graph.edge_index.to(self.device)

        optimizer = optim.Adam(base_model.parameters(), lr=self.ft_lr)

        for epoch in tqdm(range(self.ft_epochs), desc="Finetuning for Retrieval"):
            base_model.train()
            optimizer.zero_grad()

            _, enhanced_img, enhanced_txt = base_model(features, edge_index)
            
            # 难负样本挖掘
            sim_matrix = torch.matmul(F.normalize(enhanced_txt), F.normalize(enhanced_img).T)
            sim_matrix.fill_diagonal_(-1e9)
            hard_negative_indices = sim_matrix.argmax(dim=1)
            hard_negative_images = enhanced_img[hard_negative_indices]

            loss = triplet_loss_fn(enhanced_txt, enhanced_img, hard_negative_images)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 5 == 0:
                tqdm.write(f"Finetune Epoch {epoch+1}/{self.ft_epochs}, Triplet Loss: {loss.item():.4f}")

        return base_model

    def _fine_tune_for_alignment(self, base_model: torch.nn.Module) -> torch.nn.Module:
        print("警告: Alignment 任务的微调尚未实现。返回基础模型。")
        return base_model

    def train_or_load_model(self, task_name: str) -> torch.nn.Module:
        """
        主分发器方法。根据任务名称决定训练策略。
        """
        if task_name == 'modality_matching':
            return self.train_or_load_generic_model()

        finetuned_model_path = self.model_save_dir / f"model_{task_name}.pt"
        
        if finetuned_model_path.exists():
            print(f"加载微调过的模型: {finetuned_model_path}")
            model = self._init_model()
            model.load_state_dict(torch.load(finetuned_model_path, map_location=self.device))
            return model.to(self.device)
        else:
            print(f"未找到为 '{task_name}' 微调过的模型。开始两阶段训练...")
            base_model = self.train_or_load_generic_model()
            
            if task_name == 'modality_retrieval':
                finetuned_model = self._fine_tune_for_retrieval(base_model)
            elif task_name == 'modality_alignment':
                finetuned_model = self._fine_tune_for_alignment(base_model)
            else:
                raise ValueError(f"未知的微调任务: {task_name}")
            
            print(f"保存微调后的模型到: {finetuned_model_path}")
            torch.save(finetuned_model.state_dict(), finetuned_model_path)
            return finetuned_model