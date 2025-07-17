# -*- coding: utf-8 -*-
"""
用于模态检索任务的双塔模型训练器 (第二阶段训练)。
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import sys
import os
import numpy as np
from tqdm import tqdm

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.trainers.gnn_trainer import GNNTrainer
from src.multimodal_centric.qe.models.retrieval_model import TwoTowerModel

class RetrievalTrainer:
    """
    负责检索任务第二阶段的训练：训练双塔模型。
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 解析配置
        self.dataset_name = config['dataset']['name']
        self.gnn_model_name = config['model']['name']
        retrieval_params = config['retrieval_training']
        self.epochs = retrieval_params['epochs']
        self.lr = retrieval_params['lr']
        self.patience = retrieval_params.get('patience', 10)
        self.margin = retrieval_params.get('margin', 0.5) # Triplet loss 的 margin

        # 定义模型保存路径
        self.base_dir = Path(__file__).resolve().parent.parent
        self.model_save_dir = self.base_dir / "models" / "trained" / self.dataset_name
        self.model_save_path = self.model_save_dir / f"{self.gnn_model_name}_retrieval_model.pt"
        os.makedirs(self.model_save_dir, exist_ok=True)

        # 初始化第一阶段的GNN训练器以获取增强嵌入
        self.gnn_trainer = GNNTrainer(config)

    def _get_enhanced_embeddings(self) -> (torch.Tensor, torch.Tensor):
        """
        获取第一阶段GNN增强后的嵌入。
        """
        print("--- 第一阶段: 获取GNN增强嵌入 ---")
        gnn_model = self.gnn_trainer.train_or_load_model()
        
        evaluator = self.gnn_trainer # 借用其属性
        image_embeds = evaluator.embedding_manager.get_embedding(evaluator.dataset_name, "image", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        text_embeds = evaluator.embedding_manager.get_embedding(evaluator.dataset_name, "text", self.config['embedding']['encoder_name'], self.config['embedding']['dimension'])
        
        features = np.concatenate((text_embeds, image_embeds), axis=1)
        features = torch.from_numpy(features).float().to(self.device)
        
        graph = evaluator.graph_loader.load_graph(evaluator.dataset_name)
        edge_index = graph.edge_index.to(self.device)
        
        gnn_model.eval()
        with torch.no_grad():
            _, enhanced_img, enhanced_txt = gnn_model(features, edge_index)
        
        print("成功获取GNN增强嵌入。")
        return enhanced_txt, enhanced_img

    def train(self):
        """
        执行第二阶段的训练循环。
        """
        enhanced_text_embeds, enhanced_image_embeds = self._get_enhanced_embeddings()
        
        # 初始化双塔模型
        retrieval_model_params = self.config['retrieval_model']
        retrieval_model_params['input_dim'] = enhanced_text_embeds.shape[1]
        retrieval_model = TwoTowerModel(**retrieval_model_params).to(self.device)
        
        optimizer = optim.Adam(retrieval_model.parameters(), lr=self.lr)
        loss_fn = nn.TripletMarginLoss(margin=self.margin)

        print(f"--- 第二阶段: 开始训练双塔检索模型 ---")
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in tqdm(range(self.epochs), desc="Retrieval Training"):
            retrieval_model.train()
            optimizer.zero_grad()

            # 获取投影后的嵌入
            proj_text, proj_image = retrieval_model(enhanced_text_embeds, enhanced_image_embeds)
            
            # 构造三元组 (anchor, positive, negative)
            # 这里我们使用一个简单的策略：对于每个文本锚点，其对应的图像是正样本，
            # 随机采样的其他图像是负样本。
            anchor = proj_text
            positive = proj_image
            
            # 随机采样负样本
            negative_indices = torch.randperm(len(proj_image))
            negative = proj_image[negative_indices]

            loss = loss_fn(anchor, positive, negative)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                tqdm.write(f"Epoch {epoch+1}/{self.epochs}, Triplet Loss: {loss.item():.4f}")
            
            # Early stopping (简化版，实际应在验证集上计算)
            if loss.item() < best_val_loss:
                best_val_loss = loss.item()
                patience_counter = 0
                torch.save(retrieval_model.state_dict(), self.model_save_path)
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                tqdm.write(f"训练损失连续 {self.patience} 个轮次没有改善，提前停止训练。")
                break
        
        print("训练完成。")
        return retrieval_model

    def train_or_load_model(self) -> TwoTowerModel:
        """
        检查模型是否存在。如果存在，则加载；否则，进行训练。
        """
        # 初始化一个空的模型以加载状态或用于训练
        retrieval_model_params = self.config['retrieval_model']
        # 需要一个虚拟的input_dim来初始化
        retrieval_model_params['input_dim'] = self.config['model']['params']['hidden_dim']
        retrieval_model = TwoTowerModel(**retrieval_model_params).to(self.device)

        if self.model_save_path.exists():
            print(f"找到了预训练的检索模型，正在从 '{self.model_save_path}' 加载...")
            retrieval_model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        else:
            print(f"未找到预训练的检索模型于 '{self.model_save_path}'。")
            retrieval_model = self.train()
        
        retrieval_model.eval()
        return retrieval_model