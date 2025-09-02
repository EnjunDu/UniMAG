#!/usr/bin/env python3
"""
预生成负样本脚本
避免每次运行都重新计算负样本，提高运行效率
"""

import os
import pickle
import numpy as np
import torch as t
from scipy.sparse import csr_matrix, coo_matrix
import argparse
from tqdm import tqdm

def generate_and_save_negative_samples(data_dir, dataset_name, num_neg=100):
    """为指定数据集生成并保存负样本"""
    
    print(f"开始为数据集 {dataset_name} 生成负样本...")
    
    # 数据文件路径
    edge_split_file = os.path.join(data_dir, dataset_name, 'lp-edge-split.pt')
    neg_samples_file = os.path.join(data_dir, dataset_name, f'neg_samples_{num_neg}.pkl')
    
    if not os.path.exists(edge_split_file):
        print(f"错误：找不到边分割文件 {edge_split_file}")
        return False
    
    # 加载边分割数据
    print("加载边分割数据...")
    edge_split = t.load(edge_split_file, map_location='cpu')
    
    # 获取训练边（用于构建邻接矩阵）
    train_edges = edge_split['train']
    source_nodes = train_edges['source_node'].numpy()
    target_nodes = train_edges['target_node'].numpy()
    
    # 获取验证和测试边
    val_edges = edge_split['valid']
    tst_edges = edge_split['test']
    
    # 计算节点数量
    max_node = max(np.max(source_nodes), np.max(target_nodes)) + 1
    print(f"数据集信息：节点数={max_node}, 训练边数={len(source_nodes)}")
    
    # 构建训练邻接矩阵（用于检查边是否存在）
    print("构建训练邻接矩阵...")
    edge_values = np.ones(len(source_nodes), dtype=np.float32)
    trn_mat = coo_matrix((edge_values, (source_nodes, target_nodes)), shape=(max_node, max_node))
    trn_mat_csr = trn_mat.tocsr()
    
    # 生成验证集负样本
    print(f"生成验证集负样本（{len(val_edges['source_node'])} 个用户，每个 {num_neg} 个负样本）...")
    val_neg_samples = generate_negative_samples_batch(
        val_edges['source_node'].numpy(), 
        val_edges['target_node'].numpy(), 
        trn_mat_csr, 
        max_node, 
        num_neg
    )
    
    # 生成测试集负样本
    print(f"生成测试集负样本（{len(tst_edges['source_node'])} 个用户，每个 {num_neg} 个负样本）...")
    tst_neg_samples = generate_negative_samples_batch(
        tst_edges['source_node'].numpy(), 
        tst_edges['target_node'].numpy(), 
        trn_mat_csr, 
        max_node, 
        num_neg
    )
    
    # 保存负样本
    print("保存负样本到文件...")
    neg_samples_data = {
        'val_neg_samples': val_neg_samples,
        'tst_neg_samples': tst_neg_samples,
        'num_neg': num_neg,
        'max_node': max_node
    }
    
    with open(neg_samples_file, 'wb') as f:
        pickle.dump(neg_samples_data, f)
    
    print(f"负样本生成完成！")
    print(f"验证集：{len(val_neg_samples)} 个用户，每个 {num_neg} 个负样本")
    print(f"测试集：{len(tst_neg_samples)} 个用户，每个 {num_neg} 个负样本")
    print(f"保存到：{neg_samples_file}")
    
    return True

def generate_negative_samples_batch(source_nodes, target_nodes, trn_mat_csr, num_nodes, num_neg):
    """批量生成负样本"""
    neg_samples = []
    
    for i in tqdm(range(len(source_nodes)), desc="生成负样本"):
        src = source_nodes[i]
        pos_target = target_nodes[i]
        
        # 生成负样本
        neg_targets = []
        attempts = 0
        max_attempts = num_neg * 10
        
        while len(neg_targets) < num_neg and attempts < max_attempts:
            neg_target = np.random.randint(0, num_nodes)
            attempts += 1
            
            # 检查是否可用
            if (neg_target != pos_target and 
                neg_target not in neg_targets and
                not is_edge_exists_fast(src, neg_target, trn_mat_csr)):
                neg_targets.append(neg_target)
        
        # 如果生成的负样本不足，用随机样本填充
        while len(neg_targets) < num_neg:
            neg_target = np.random.randint(0, num_nodes)
            if neg_target != pos_target:
                neg_targets.append(neg_target)
        
        neg_samples.append(neg_targets[:num_neg])
    
    return np.array(neg_samples)

def is_edge_exists_fast(src, dst, trn_mat_csr):
    """快速检查边是否存在"""
    return trn_mat_csr[src, dst] != 0

def main():
    parser = argparse.ArgumentParser(description='预生成负样本')
    parser.add_argument('--data_dir', type=str, required=True, help='数据根目录')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称')
    parser.add_argument('--num_neg', type=int, default=100, help='每个正样本的负样本数量')
    
    args = parser.parse_args()
    
    print(f"参数：数据目录={args.data_dir}, 数据集={args.dataset}, 负样本数={args.num_neg}")
    
    # 生成负样本
    success = generate_and_save_negative_samples(args.data_dir, args.dataset, args.num_neg)
    
    if success:
        print("✅ 负样本生成成功！")
    else:
        print("❌ 负样本生成失败！")

if __name__ == "__main__":
    main()
