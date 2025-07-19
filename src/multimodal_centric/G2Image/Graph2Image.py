# -*- coding: utf-8 -*-
"""
G2Image.py

功能：
1. 调用项目内模块加载并处理多模态 embedding
2. 构建 prompt，并添加生成提示词
3. 调用 DashScope 文生图 API 生成图片
"""

import os
import time
import requests
import torch
import numpy as np
from typing import Dict

# 引入项目内模块
from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader
from src.model.models import GCN  # Graph Convolutional Network

def load_and_process_embeddings(dataset_name: str,
                                base_path: str = './data',
                                encoder_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
                                dimension: int = 768,
                                gcn_hidden_dim: int = 64,
                                gcn_num_layers: int = 3,
                                gcn_dropout: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    使用项目内的 EmbeddingManager 和 GraphLoader 加载原始多模态 embedding 并用 GCN 自动分离模态特征
    返回 dict: {'image': Tensor, 'text': Tensor}
    """
    # 1. 初始化管理器和加载器
    em = EmbeddingManager(base_path=base_path)
    gl = GraphLoader(data_root=base_path)

    # 2. 获取原始多模态 embedding
    embeddings_np = em.get_embedding(
        dataset_name=dataset_name,
        modality="multimodal",
        encoder_name=encoder_name,
        dimension=dimension
    )  # 返回 numpy array
    embeddings = torch.from_numpy(embeddings_np).float().to(em.device)

    # 3. 加载图结构
    graph_data = gl.load_graph(dataset_name)
    edge_index = graph_data.edge_index.to(em.device)

    # 4. 初始化并执行 GCN
    gcn = GCN(
        in_dim=embeddings.size(1),
        hidden_dim=gcn_hidden_dim,
        num_layers=gcn_num_layers,
        dropout=gcn_dropout
    ).to(em.device)
    _, out_v, out_t = gcn(embeddings, edge_index)
    return {'image': out_v, 'text': out_t}


def build_prompt(image_feat: np.ndarray,
                 text_feat: np.ndarray,
                 top_k: int = 10) -> str:
    """
    从 image/text 特征中各选 top_k 维，生成文本 prompt
    返回完整 prompt
    """
    img_idx = np.argsort(-image_feat)[:top_k]
    txt_idx = np.argsort(-text_feat)[:top_k]
    parts = []
    for idx in img_idx:
        parts.append(f"图像特征{idx}")
    for idx in txt_idx:
        parts.append(f"文本特征{idx}")
    feat_str = '，'.join(parts)
    # 添加提示词
    return f"请基于以下多模态节点特征生成场景图像：{feat_str}。"

# 文生图 API 函数
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")
SERVICE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
TASK_URL    = "https://dashscope.aliyuncs.com/api/v1/tasks/{}"

def create_text2image_task(prompt: str,
                           model: str = "wanx2.1-t2i-turbo",
                           size: str = "1024*1024",
                           n: int = 1,
                           seed: int = None,
                           prompt_extend: bool = True,
                           watermark: bool = False) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": model,
        "input": {"prompt": prompt},
        "parameters": {"size": size, "n": n, "prompt_extend": prompt_extend, "watermark": watermark}
    }
    if seed is not None:
        payload["parameters"]["seed"] = seed
    resp = requests.post(SERVICE_URL, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["output"]["task_id"]

def get_task_result(task_id: str, interval: int = 5, timeout: int = 300) -> list:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    start = time.time()
    while True:
        resp = requests.get(TASK_URL.format(task_id), headers=headers)
        resp.raise_for_status()
        out = resp.json()["output"]
        status = out["task_status"]
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 状态：{status}")
        if status == "SUCCEEDED":
            return [i.get("url") for i in out.get("results", [])]
        if status in ("FAILED","CANCELED"):
            raise RuntimeError(f"任务失败：{out}")
        if time.time()-start>timeout:
            raise TimeoutError("超时")
        time.sleep(interval)

if __name__ == "__main__":
    # 参数区
    DATASET_NAME   = "Grocery"
    TARGET_NODE_ID = 0
    BASE_PATH      = "./data"
    TOP_K          = 10

    # 加载并处理 embedding
    feats = load_and_process_embeddings(DATASET_NAME, BASE_PATH)

    # 构建 prompt
    img_feat = feats['image'][TARGET_NODE_ID].cpu().numpy()
    txt_feat = feats['text'][TARGET_NODE_ID].cpu().numpy()
    prompt = build_prompt(img_feat, txt_feat, TOP_K)
    print("生成的 prompt: ", prompt)

    # 提交并获取图片
    task_id = create_text2image_task(prompt)
    print("task_id=", task_id)
    urls = get_task_result(task_id)
    for i, url in enumerate(urls, 1):
        print(f"第{i}张图: {url}")
