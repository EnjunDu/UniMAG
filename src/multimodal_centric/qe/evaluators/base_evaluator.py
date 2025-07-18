# -*- coding: utf-8 -*-
"""
评估器基类

该模块定义了 BaseEvaluator，这是一个抽象基类，为所有具体的评估器提供了
共享的功能，包括：
-   统一的初始化流程，处理配置和GNN模型。
-   一个通用的 `_get_enhanced_embeddings` 方法，用于获取图增强的嵌入。
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader

class BaseEvaluator(ABC):
    """
    所有QE评估器的抽象基类。
    """
    def __init__(self, config: Dict[str, Any], gnn_model: torch.nn.Module):
        """
        初始化评估器。

        Args:
            config (Dict[str, Any]): 包含任务和数据集配置的字典。
            gnn_model (torch.nn.Module): 一个已经训练好的、可用于评估的GNN模型。
        """
        self.config = config
        self.gnn_model = gnn_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_model.to(self.device)

        self.embedding_manager = EmbeddingManager(base_path=self.config.dataset.data_root)
        self.graph_loader = GraphLoader(config=self.config)

    def _get_enhanced_embeddings(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        使用GNN模型计算整个图的邻域增强嵌入。
        这是一个通用方法，所有子类都可以使用。

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]:
                    一个包含 (所有节点的增强图像嵌入, 所有节点的增强文本嵌入) 的元组。
                    如果无法获取任何所需数据，则返回 None。
            """
            dataset_name = self.config.dataset.name
            encoder_name = self.config.embedding.encoder_name
            dimension = self.config.embedding.dimension
    
            image_embeddings = self.embedding_manager.get_embedding(
                dataset_name, "image", encoder_name, dimension
        )
        text_embeddings = self.embedding_manager.get_embedding(
            dataset_name, "text", encoder_name, dimension
        )

        if image_embeddings is None or text_embeddings is None:
            print("错误: 无法加载基础嵌入。")
            return None

        try:
            graph_data = self.graph_loader.load_graph(dataset_name)
            edge_index = graph_data.edge_index.to(self.device)
        except (FileNotFoundError, ValueError) as e:
            print(f"错误: 无法加载图结构: {e}")
            return None

        multimodal_features = np.concatenate((text_embeddings, image_embeddings), axis=1)
        features_tensor = torch.from_numpy(multimodal_features).float().to(self.device)

        self.gnn_model.eval()
        with torch.no_grad():
            _, enhanced_image_embeddings, enhanced_text_embeddings = self.gnn_model(features_tensor, edge_index)

        return enhanced_image_embeddings.cpu().numpy(), enhanced_text_embeddings.cpu().numpy()

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """
        执行评估的抽象方法。
        每个子类必须实现自己的评估逻辑。
        """
        pass