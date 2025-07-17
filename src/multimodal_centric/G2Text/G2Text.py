# -*- coding: utf-8 -*-
"""
多模态文本生成器 (优化版)

改进点：
1. 直接输入原始多模态embedding，不手动分离图像/文本
2. 利用GCN自动输出分离模态的特征 (out_v, out_t)
3. 更简洁的流程
"""

import torch
import numpy as np
from typing import List, Optional,Dict
from pathlib import Path
import json
import requests
import sys
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
# 导入项目内模块
from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader
from src.model.models import GCN

class EfficientDescGenerator:
    def __init__(
        self,
        base_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        # 设置设备为传入的device，如果没有传入则检查CUDA是否可用，如果可用则使用cuda，否则使用cpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化嵌入管理器
        self.embedding_manager = EmbeddingManager(base_path=base_path)
        # 初始化图加载器
        self.graph_loader = GraphLoader(data_root = base_path)
        # 打印初始化完成信息
        print(f"生成器初始化完成，设备: {self.device}")

    def get_multimodal_embeddings(
        self,
        dataset_name: str,
        encoder_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        dimension: int = 768
    ) -> torch.Tensor:
        """直接获取多模态embedding（不区分图像/文本）"""
        # 获取原始多模态embedding
        embeddings = self.embedding_manager.get_embedding(
            dataset_name=dataset_name,
            modality="multimodal",
            encoder_name=encoder_name,
            dimension=dimension
        )
        print("embedding获取完毕")
        return torch.from_numpy(embeddings).to(self.device)

    def process_with_gcn(
        self,
        embeddings: torch.Tensor,
        dataset_name: str,
        gcn_hidden_dim: int = 64,
        gcn_num_layers: int = 3,
        gcn_dropout: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """用GCN处理embedding并自动分离模态"""
        # 加载图结构
        graph_data = self.graph_loader.load_graph(dataset_name)
        edge_index = graph_data.edge_index.to(self.device)
        
        # 初始化GCN（输入维度自动匹配）
        gcn = GCN(
            in_dim=embeddings.size(1),
            hidden_dim=gcn_hidden_dim,
            num_layers=gcn_num_layers,
            dropout=gcn_dropout
        ).to(self.device)
        # 前向传播（自动分离模态）
        out, out_v, out_t = gcn(embeddings, edge_index)
        return {
            "image": out_v,  # 图像模态特征
            "text": out_t    # 文本模态特征
        }
    def build_prompt(self,image_feat: np.ndarray, text_feat: np.ndarray, top_k: int):
    # 仅展示前K维特征以便语言模型理解
        img_vec = ", ".join([f"{x:.3f}" for x in image_feat[:top_k]])
        txt_vec = ", ".join([f"{x:.3f}" for x in text_feat[:top_k]])
        
        prompt = f"""
            你正在观察一个多模态属性图中的节点。以下是该节点的模态特征（已聚合其邻居信息）：
            
            图像模态特征前{top_k}维: [{img_vec}]
            文本模态特征前{top_k}维: [{txt_vec}]
            
            请结合这些特征，推测该节点可能代表的内容，并用一段文本去描述目标节点。无需给出推理过程。
            """
        return prompt.strip()
    def generate_description(
        self,
        features: Dict[str, torch.Tensor],
        target_index: int,
        api_key: str,
        model_name: str = "qwen-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_k: int = 768
    ) -> str:
        """调用Qwen API生成节点描述"""
        image_feature = features["image"][target_index].cpu().detach().numpy()
        text_feature = features["text"][target_index].cpu().detach().numpy()
    
        prompt = self.build_prompt(image_feat=image_feature, text_feat=text_feature, top_k=top_k)
    
        data = {
            "model": model_name,
            "input": {
                "prompt": prompt
            },
            "parameters": {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        }
    
        try:
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=data
            )
            response.raise_for_status()
            return response.json()["output"]["text"]
        except Exception as e:
            raise RuntimeError(f"API调用失败: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 配置
    DATASET_NAME = "Grocery"
    TARGET_NODE_ID = 0
    QWEN_API_KEY = "your-api-key"
    base_path = "your-datapath"
    # 初始化
    generator = EfficientDescGenerator(base_path = base_path)
    
    try:
        # 1. 获取多模态embedding
        print("加载embedding...")
        all_node_ids = [TARGET_NODE_ID]
        embeddings = generator.get_multimodal_embeddings(
            dataset_name=DATASET_NAME,
        )
        
        # 2. GCN处理并分离模态
        print("GCN处理...")
        features = generator.process_with_gcn(
            embeddings=embeddings,
            dataset_name=DATASET_NAME
        )
        
        # 3. 生成描述
        print("生成描述...")
        desc = generator.generate_description(
            features=features,
            target_index=TARGET_NODE_ID,  # 目标节点是batch中的第一个
            api_key=QWEN_API_KEY
        )
        
        print("\n生成结果：")
        print(desc)
        
    except Exception as e:
        print(f"错误: {str(e)}")