# -*- coding: utf-8 -*-
"""
GNN 模型训练器

该模块负责：
1.  根据配置初始化一个GNN模型。
2.  检查是否存在已经为特定数据集和GNN模型训练好的权重。
3.  如果权重存在，则加载；如果不存在，则执行训练循环。
4.  训练使用InfoNCE对比学习目标。
5.  在验证集上实施Early Stopping以防止过拟合。
6.  保存训练好的模型权重。
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
        
        self.dataset_name = self.config.dataset.name
        self.gnn_model_name = self.config.model.name
        
        train_params = self.config.training
        self.epochs = train_params.epochs
        self.lr = train_params.lr
        self.patience = train_params.patience
        self.val_ratio = train_params.val_ratio
        self.tau = train_params.tau

        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_save_dir = self.base_dir / "trained_models" / self.dataset_name / self.gnn_model_name
        self.model_save_path = self.model_save_dir / "model.pt"
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        self.embedding_manager = EmbeddingManager(base_path=self.config.dataset.data_root)
        self.graph_loader = GraphLoader(config=self.config)

        self.model = self._init_model().to(self.device)

    def _init_model(self) -> torch.nn.Module:
        model_params = self.config.model.params
        encoder_name = self.config.embedding.encoder_name
        dimension = self.config.embedding.dimension
        sample_text_embed = self.embedding_manager.get_embedding(self.dataset_name, "text", encoder_name, dimension)
        if sample_text_embed is None: raise ValueError("Cannot load embedding to determine model input dimension.")
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
            raise ValueError(f"Unknown GNN model: {self.gnn_model_name}")
        return model

    def _calculate_infonce_loss(self, query, positive_key, all_keys) -> torch.Tensor:
        query = F.normalize(query, p=2, dim=1)
        positive_key = F.normalize(positive_key, p=2, dim=1)
        all_keys = F.normalize(all_keys, p=2, dim=1)
        l_pos = (query * positive_key).sum(dim=-1)
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

    def train(self):
        print(f"Start training GNN model '{self.gnn_model_name}' for dataset '{self.dataset_name}'...")
        
        graph = self.graph_loader.load_graph(self.dataset_name)
        train_edge_index, val_edge_index = self._get_data_splits(graph.edge_index)
        train_edge_index = train_edge_index.to(self.device)
        val_edge_index = val_edge_index.to(self.device)

        encoder_name = self.config.embedding.encoder_name
        dimension = self.config.embedding.dimension
        image_embeds = self.embedding_manager.get_embedding(self.dataset_name, "image", encoder_name, dimension)
        text_embeds = self.embedding_manager.get_embedding(self.dataset_name, "text", encoder_name, dimension)
        
        scores = [calculate_clip_score(img, txt) for img, txt in zip(image_embeds, text_embeds)]
        valid_scores = [s for s in scores if s is not None]
        num_invalid = len(scores) - len(valid_scores)
        
        print(f"Baseline CLIP-Score (no GNN enhancement): {np.mean(valid_scores):.4f}")
        if num_invalid > 0:
            print(f"  (Warning: Ignored {num_invalid} zero-vector samples when calculating baseline)")

        features = torch.from_numpy(np.concatenate((text_embeds, image_embeds), axis=1)).float().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(self.epochs), desc="GNN Training"):
            self.model.train()
            optimizer.zero_grad()
            _, enhanced_img, enhanced_txt = self.model(features, train_edge_index)
            loss = self._calculate_infonce_loss(enhanced_img, enhanced_txt, enhanced_txt)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                _, val_img, val_txt = self.model(features, val_edge_index)
                val_loss = self._calculate_infonce_loss(val_img, val_txt, val_txt)

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                tqdm.write(f"Validation loss has not improved for {self.patience} consecutive epochs. Early stopping.")
                break
        
        print("Training completed. Loading best performing model.")
        self.model.load_state_dict(torch.load(self.model_save_path))

    def train_or_load_model(self) -> torch.nn.Module:
        if self.model_save_path.exists():
            print(f"Found pre-trained model at '{self.model_save_path}'. Loading...")
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        else:
            print(f"No pre-trained model found at '{self.model_save_path}'.")
            self.train()
        
        self.model.eval()
        return self.model