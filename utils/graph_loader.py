# -*- coding: utf-8 -*-
"""
通用图加载器 (Graph Loader)

该模块提供一个通用的 `GraphLoader` 类，用于根据数据集名称加载和提供标准的图结构数据。
"""

import os
import torch
import dgl
from torch_geometric.data import Data
from typing import Dict, Any, Optional

class GraphLoader:
    """
    一个通用的图加载器，用于从不同的数据集中加载图结构。
    """
    def __init__(self, data_root: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化 GraphLoader。

        Args:
            data_root (Optional[str]): 数据集的根目录路径。如果提供，则优先使用此路径。
            config (Optional[Dict[str, Any]]): 包含数据集配置的字典。
                                               如果 'data_root' 未提供，则会尝试从这里获取。
        """
        if data_root:
            self.data_root = data_root
        elif config and 'data_root' in config.get('dataset', {}):
            self.data_root = config['dataset']['data_root']
        else:
            self.data_root = "/home/ai/MMAG"  # 默认服务器路径
        
        print(f"Info: GraphLoader is initialized. Using data root: {self.data_root}")

    def _get_graph_path(self, dataset_name: str) -> str:
        """
        根据数据集名称获取图文件的相对路径。

        Args:
            dataset_name (str): 数据集的名称。

        Returns:
            str: 图文件的相对路径。
        
        Raises:
            ValueError: 如果数据集名称不被支持。
        """
        # MAGB 数据集的图文件通常以 "Graph.pt" 结尾
        if dataset_name in ["Grocery", "Movies", "Reddit-M", "Reddit-S", "Toys"]:
            return f"{dataset_name}/{dataset_name}Graph.pt"
        
        # mm-graph-benchmark 数据集的图文件路径
        elif dataset_name == "books-lp":
            return "books-lp/lp-edge-split.pt"
        elif dataset_name == "books-nc":
            return "books-nc/nc_edges-nodeid.pt"
        elif dataset_name == "cloth-copurchase":
            return "cloth-copurchase/lp-edge-split.pt"
        elif dataset_name == "ele-fashion":
            return "ele-fashion/nc_edges-nodeid.pt"
        elif dataset_name == "sports-copurchase":
            return "sports-copurchase/lp-edge-split.pt"
        
        # mm-codex 数据集
        elif dataset_name in ["mm-codex-m", "mm-codex-s"]:
             return f"{dataset_name}/Graph.pt"

        else:
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

    def load_graph(self, dataset_name: str) -> Data:
        """
        根据数据集名称加载图数据。

        Args:
            dataset_name (str): 数据集的名称。

        Returns:
            Data: 一个包含图结构信息的 `torch_geometric.data.Data` 对象。
        
        Raises:
            FileNotFoundError: 如果图文件不存在。
        """
        relative_path = self._get_graph_path(dataset_name)
        full_path = os.path.join(self.data_root, relative_path)

        print(f"Info: Attempting to load graph from: {full_path}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Graph file not found at: {full_path}")

        # 加载图数据
        if "lp-edge-split.pt" in full_path or "nc_edges-nodeid.pt" in full_path:
            # mm-graph-benchmark 的部分数据集直接是 edge_index
            if "nc_edges-nodeid.pt" in full_path:
                edge_index = torch.load(full_path)
            else: # lp-edge-split.pt
                edge_split = torch.load(full_path)
                edge_index = edge_split['train']['edge']
        
        elif "Graph.pt" in full_path:
             # mm-codex 和 MAGB 数据集是 DGL Graph 对象
            graph_data = dgl.load_graphs(full_path)
            if isinstance(graph_data, tuple) and len(graph_data) > 0:
                graph = graph_data[0][0] # DGL-style
                src, dst = graph.edges()
                edge_index = torch.stack([src, dst], dim=0)
            else:
                raise TypeError(f"Unsupported graph data format in {full_path}")
        else:
            raise ValueError(f"Unknown graph file format for {dataset_name}")

        print(f"Info: Successfully loaded graph for '{dataset_name}'. Edge index shape: {edge_index.shape}")
        
        return Data(edge_index=edge_index)

if __name__ == '__main__':
    print("=== GraphLoader 使用示例 ===")
    
    loader = GraphLoader() # 使用默认或配置的路径

    # --- 示例 1: 加载 MAGB 数据集 ---
    try:
        print("\n--- 示例 1: 加载 MAGB 数据集 (Grocery) ---")
        grocery_graph_data = loader.load_graph("Grocery")
        print(f"成功加载 'Grocery' 图。Edge index shape: {grocery_graph_data.edge_index.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")

    # --- 示例 2: 加载 mm-graph 数据集 ---
    try:
        print("\n--- 示例 2: 加载 mm-graph 数据集 (books-nc) ---")
        books_nc_data = loader.load_graph("books-nc")
        print(f"成功加载 'books-nc' 图。Edge index shape: {books_nc_data.edge_index.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")

    # --- 示例 3: 尝试加载不支持的数据集 ---
    try:
        print("\n--- 示例 3: 尝试加载不支持的数据集 ---")
        loader_for_test = GraphLoader()
        loader_for_test.load_graph("non_existent_dataset")
    except ValueError as e:
        print(f"成功捕获到预期的错误: {e}")