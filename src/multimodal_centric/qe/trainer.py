# -*- coding: utf-8 -*-
"""
GNN 模型训练器

该模块负责：
1.  根据配置初始化一个GNN模型。
2.  检查是否存在已经为特定数据集和GNN模型训练好的权重。
3.  如果权重存在，则加载；如果不存在，则执行训练循环。
4.  训练使用对比学习目标，最大化同节点图文嵌入的相似度。
5.  保存训练好的模型权重。
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

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader
from src.model.models import GCN, GraphSAGE, GAT, MLP
from src.model.MMGCN import Net as MMGCN
from src.model.MGAT import MGAT as MGAT_model
from src.model.REVGAT import RevGAT

class GNNTrainer:
    """
    负责训练、保存和加载用于QE任务的GNN模型。
    """
    def __init__(self, config: dict):
        """
        初始化训练器。

        Args:
            config (dict): 包含所有必要配置的字典。
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 从配置中解析关键信息
        self.dataset_name = self.config['dataset']['name']
        self.gnn_model_name = self.config['model']['name']
        self.epochs = self.config['training']['epochs']
        self.lr = self.config['training']['lr']
        
        # 定义模型和日志的存储路径
        self.base_dir = Path(__file__).resolve().parent
        self.model_save_dir = self.base_dir / "trained_models" / self.dataset_name / self.gnn_model_name
        self.model_save_path = self.model_save_dir / "model.pt"
        self.log_dir = self.base_dir / "logs" / self.dataset_name
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化管理器
        base_path = self.config.get('dataset', {}).get('data_root')
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        self.graph_loader = GraphLoader(config=self.config)

        # 初始化GNN模型
        self.model = self._init_model().to(self.device)

    def _init_model(self) -> torch.nn.Module:
        """根据配置初始化GNN模型。"""
        model_params = self.config['model']
        
        encoder_name = self.config['embedding']['encoder_name']
        dimension = self.config['embedding']['dimension']
        sample_text_embed = self.embedding_manager.get_embedding(self.dataset_name, "text", encoder_name, dimension)
        
        if sample_text_embed is None:
            raise ValueError("无法加载嵌入以确定模型输入维度。")
            
        in_dim = sample_text_embed.shape[1] * 2
        
        print(f"根据配置 '{self.gnn_model_name}' 初始化GNN模型，输入维度为 {in_dim}。")
        
        if self.gnn_model_name == 'GCN':
            model = GCN(in_dim=in_dim, **model_params)
        elif self.gnn_model_name == 'GAT':
            model = GAT(in_dim=in_dim, **model_params)
        elif self.gnn_model_name == 'GraphSAGE':
            model = GraphSAGE(in_dim=in_dim, **model_params)
        elif self.gnn_model_name == 'MLP':
            model = MLP(in_dim=in_dim, **model_params)
        elif self.gnn_model_name == "RevGAT":
            model = RevGAT(in_feats=in_dim, **model_params)
        elif self.gnn_model_name == "MMGCN":
            # MMGCN 和 MGAT 有不同的参数签名，需要特殊处理
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

    def _calculate_loss(self, enhanced_image_embeds, enhanced_text_embeds) -> torch.Tensor:
        """
        计算对比损失。
        """
        enhanced_image_embeds = F.normalize(enhanced_image_embeds, p=2, dim=1)
        enhanced_text_embeds = F.normalize(enhanced_text_embeds, p=2, dim=1)
        cosine_sim = F.cosine_similarity(enhanced_image_embeds, enhanced_text_embeds, dim=1)
        loss = -cosine_sim.mean()
        return loss

    def train(self):
        """执行完整的训练循环。"""
        print(f"开始为数据集 '{self.dataset_name}' 训练GNN模型 '{self.gnn_model_name}'...")
        
        graph = self.graph_loader.load_graph(self.dataset_name)
        edge_index = graph.edge_index.to(self.device)
        
        encoder_name = self.config['embedding']['encoder_name']
        dimension = self.config['embedding']['dimension']
        image_embeds = self.embedding_manager.get_embedding(self.dataset_name, "image", encoder_name, dimension)
        text_embeds = self.embedding_manager.get_embedding(self.dataset_name, "text", encoder_name, dimension)
        
        if image_embeds is None or text_embeds is None:
            raise ValueError("无法加载训练所需的嵌入向量。")

        features = np.concatenate((text_embeds, image_embeds), axis=1)
        features = torch.from_numpy(features).float().to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        print(f"将在 {self.device} 上训练 {self.epochs} 个轮次。")
        for epoch in tqdm(range(self.epochs), desc="GNN Training"):
            self.model.train()
            optimizer.zero_grad()
            
            _, enhanced_img, enhanced_txt = self.model(features, edge_index)
            loss = self._calculate_loss(enhanced_img, enhanced_txt)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
        
        print("训练完成。")
        
        print(f"正在将模型保存到: {self.model_save_path}")
        torch.save(self.model.state_dict(), self.model_save_path)
        print("模型保存成功。")

    def train_or_load_model(self) -> torch.nn.Module:
        """
        检查模型是否存在。如果存在，则加载；否则，进行训练。
        """
        if self.model_save_path.exists():
            print(f"找到了预训练模型，正在从 '{self.model_save_path}' 加载...")
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            self.model.eval()
            print("模型加载成功。")
        else:
            print(f"未找到预训练模型于 '{self.model_save_path}'。")
            self.train()
            self.model.eval()
        
        return self.model