#!/usr/bin/env python
import os
import sys
import json
import random
import pandas as pd
from pathlib import Path
import argparse

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
print("Script starting - testing output...")
# 一些参数
parser = argparse.ArgumentParser(description="process.py 参数.")
parser.add_argument(
        "--train_ratio", 
        type=float, 
        # required=True, 
        default=0.8,
        help="划分训练集和测试集中，训练集占比"
    )
parser.add_argument(
        "--read_data_path", 
        type=str, 
        default="./data/Movies/MoviesGraph.pt",
        help="读取的dgl.pt文件路径"
    )
parser.add_argument(
        "--csv_path", 
        type=str, 
        default="./data/Movies/Movies.csv",
        help="读取的节点文本信息的CSV文件路径"
    )
parser.add_argument(
        "--save_data_path", 
        type=str, 
        default="./data/Movies",
        help="保存的.jsonl文件路径(会自动在该文件下创建train和test文件夹)"
    )
args = parser.parse_args()

try:
    import dgl
    import torch
    import numpy as np
    print("成功导入所需库")

    # 加载图数据
    print(f"从 {args.read_data_path} 加载图...")
    graph_list, _ = dgl.load_graphs(args.read_data_path)
    g = graph_list[0]

    # 输出图的基本信息
    print(f"图加载成功")
    print(f"图信息: {g}")
    print(f"节点数量: {g.num_nodes()}")
    print(f"边数量: {g.num_edges()}")
    print(f"节点特征: {g.ndata.keys()}")
    print(f"边特征: {g.edata.keys()}")

    # 加载CSV文件以获取节点的文本信息
    print(f"从 {args.csv_path} 加载节点文本信息...")
    df = pd.read_csv(args.csv_path)
    print(f"CSV文件加载成功，共 {len(df)} 行")
    
    # 检查CSV文件的列
    print(f"CSV文件列: {df.columns.tolist()}")
    
    # 创建节点ID到文本的映射
    node_to_text = {}
    for i, row in df.iterrows():
        node_id = row['id']  # CSV中的id列对应于图中的节点ID
        # 优先使用text列，如果text为空则使用title列，如果都为空则使用默认值
        text = row['text'] if pd.notna(row['text']) else row['title'] if pd.notna(row['title']) else f"Node {node_id}"
        node_to_text[node_id] = text
    
    print(f"创建了 {len(node_to_text)} 个节点的文本映射")

    # 获取所有节点ID
    all_nodes = list(range(g.num_nodes()))

    # 随机打乱节点ID
    random.seed(42)  # 为了可重复性
    random.shuffle(all_nodes)

    # 按指定比例划分训练集和测试集
    split_idx = int(len(all_nodes) * args.train_ratio)
    train_nodes = all_nodes[:split_idx]
    test_nodes = all_nodes[split_idx:]

    print(f"训练集大小: {len(train_nodes)}")
    print(f"测试集大小: {len(test_nodes)}")

    # 创建输出目录（如果不存在）
    train_dir = Path(args.save_data_path + "/train")
    test_dir = Path(args.save_data_path + "/test")
    
    print(f"创建目录: {train_dir} 和 {test_dir}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"检查目录是否创建成功:")
    print(f"训练目录存在: {os.path.exists(train_dir)}")
    print(f"测试目录存在: {os.path.exists(test_dir)}")

    # 创建节点元数据并写入JSONL文件的函数
    def write_nodes_to_jsonl(nodes, output_path):
        print(f"正在将 {len(nodes)} 个节点写入 {output_path}...")
        try:
            with open(output_path, 'w') as f:
                for node_id in nodes:
                    # 获取节点特征
                    node_data = {}
                    
                    # 将节点ID设置为center字段
                    node_data["center"] = str(node_id)
                    
                    # 从映射中获取节点的文本信息
                    if node_id in node_to_text:
                        node_data["text"] = node_to_text[node_id]
                    else:
                        # 如果在CSV中找不到对应的文本，使用默认值
                        node_data["text"] = f"Node {node_id}"
                    
                    # 获取相邻节点（使用有向图的出边）
                    neighbors = g.predecessors(node_id).tolist()
                    node_data["neighbors"] = [str(n) for n in neighbors]
                    
                    # 写入JSONL文件
                    f.write(json.dumps(node_data) + "\n")
            print(f"完成写入 {output_path}")
            return True
        except Exception as e:
            print(f"写入 {output_path} 时出错: {e}")
            return False

    # 写入训练集和测试集节点到JSONL文件
    train_file = os.path.join(train_dir, "metadata.jsonl")
    test_file = os.path.join(test_dir, "metadata.jsonl")
    
    train_success = write_nodes_to_jsonl(train_nodes, train_file)
    test_success = write_nodes_to_jsonl(test_nodes, test_file)

    print(f"训练文件写入成功: {train_success}")
    print(f"测试文件写入成功: {test_success}")
    
    # 验证文件是否创建成功
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("两个文件均创建成功")
        print(f"训练文件大小: {os.path.getsize(train_file)} 字节")
        print(f"测试文件大小: {os.path.getsize(test_file)} 字节")
    else:
        print(f"训练文件存在: {os.path.exists(train_file)}")
        print(f"测试文件存在: {os.path.exists(test_file)}")

    print("脚本执行完成")

except Exception as e:
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)