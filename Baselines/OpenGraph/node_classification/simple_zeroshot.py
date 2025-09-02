#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简单的Zero-shot推理脚本 - 直接使用预训练模型进行推理
避免重新训练和SVD计算，节省内存和时间
"""

import torch as t
import numpy as np
import os
import pickle
from scipy.sparse import coo_matrix
import torch.utils.data as data

# 简单的评估函数
def accuracy_score(y_true, y_pred):
    """简单的准确率计算"""
    return (y_true == y_pred).mean()

def f1_score(y_true, y_pred, average='macro'):
    """简单的F1分数计算"""
    from collections import Counter
    import numpy as np
    
    def calculate_f1(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    # 获取所有类别
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    if average == 'macro':
        f1_scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_scores.append(calculate_f1(precision, recall))
        
        return np.mean(f1_scores)
    elif average == 'micro':
        # 微平均F1
        tp_total = fp_total = fn_total = 0
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            tp_total += tp
            fp_total += fp
            fn_total += fn
        
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        return calculate_f1(precision, recall)
    else:
        raise ValueError(f"Average method '{average}' not supported")

class NodeDataset(data.Dataset):
    """节点数据集"""
    def __init__(self, nodes, labels):
        self.nodes = nodes
        self.labels = labels
    
    def __len__(self):
        return len(self.nodes)
    
    def __getitem__(self, idx):
        return t.tensor(self.nodes[idx], dtype=t.long), t.tensor(self.labels[idx], dtype=t.long)

def main():
    # 配置参数
    model_path = '../Models/pretrn_gen1.mod'
    data_name = 'books-nc'
    data_dir = '../datasets'
    batch_size = 64
    
    print(f"使用参数: model_path={model_path}, data_name={data_name}, batch_size={batch_size}")
    
    # 设置设备
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载预训练模型
    print(f"加载预训练模型: {model_path}")
    try:
        # 修复PyTorch 2.6的weights_only问题
        try:
            checkpoint = t.load(model_path, weights_only=False)
        except:
            from torch.serialization import add_safe_globals
            # 动态导入模型类
            import sys
            sys.path.append('.')
            from model import OpenGraph
            add_safe_globals([OpenGraph])
            checkpoint = t.load(model_path, weights_only=False)
        
        model = checkpoint['model']
        model = model.to(device)
        model.eval()
        print("预训练模型加载成功")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("尝试创建新模型...")
        import sys
        sys.path.append('.')
        from model import OpenGraph
        model = OpenGraph()
        model = model.to(device)
        model.eval()
    
    # 加载数据
    print(f"加载数据集: {data_name}")
    data_dir_path = os.path.join(data_dir, data_name)
    edge_file = os.path.join(data_dir_path, 'nc_edges-nodeid.pt')
    label_file = os.path.join(data_dir_path, 'labels-w-missing.pt')
    split_file = os.path.join(data_dir_path, 'split.pt')
    
    # 加载边数据
    edge_list = t.load(edge_file, weights_only=True)
    edges = np.asarray(edge_list, dtype=np.int64)
    num_nodes = int(edges.max()) + 1
    
    # 创建邻接矩阵
    adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), 
                    shape=(num_nodes, num_nodes), dtype=np.float32)
    adj = adj + adj.T  # 对称化
    adj = (adj != 0).astype(np.float32)  # 二值化
    
    # 加载标签和分割
    labels = t.load(label_file, weights_only=True)
    labels = np.asarray(labels, dtype=np.int64)
    if labels.min() != 0:
        labels = labels - labels.min()
    
    split = t.load(split_file, weights_only=True)
    train_idx = np.asarray(split['train_idx'], dtype=np.int64)
    val_idx = np.asarray(split['val_idx'], dtype=np.int64)
    test_idx = np.asarray(split['test_idx'], dtype=np.int64)
    
    # 创建掩码
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    print(f"数据加载完成: 节点数={num_nodes}, 边数={adj.nnz}, 类别数={int(labels.max()) + 1}")
    
    # 创建初始投影器（使用随机初始化避免SVD）
    print("创建初始投影器...")
    node_num = adj.shape[0]
    latdim = 1024  # 使用预训练模型的维度
    projection = t.randn(node_num, latdim, device=device)
    projection = t.nn.functional.normalize(projection, p=2, dim=1)
    
    # 创建PyTorch稀疏邻接矩阵
    print("创建PyTorch邻接矩阵...")
    adj_coo = adj.tocoo()
    indices = t.from_numpy(np.vstack([adj_coo.row, adj_coo.col]).astype(np.int64))
    values = t.from_numpy(adj_coo.data.astype(np.float32))
    shape = t.Size(adj_coo.shape)
    torch_adj = t.sparse_coo_tensor(indices, values, shape, device=device)
    
    def inference(mask, mask_name):
        """进行推理"""
        print(f"\n=== {mask_name}推理 ===")
        print(f"节点数: {mask.sum()}")
        
        # 获取需要推理的节点
        test_nodes = np.where(mask)[0]
        
        # 创建数据加载器
        test_dataset = NodeDataset(test_nodes, labels[test_nodes])
        test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []
        
        with t.no_grad():
            for i, (nodes, labels_batch) in enumerate(test_loader):
                nodes = nodes.to(device)
                labels_batch = labels_batch.to(device)
                
                # 进行推理
                preds = model.pred_for_node_test(
                    nodes, torch_adj, lambda: projection, 
                    rerun_embed=False if i != 0 else True
                )
                
                all_preds.append(preds.cpu())
                all_labels.append(labels_batch.cpu())
                
                if i % 10 == 0:
                    print(f"推理进度: {i+1}/{len(test_loader)}")
        
        # 合并结果
        all_preds = t.cat(all_preds, dim=0)
        all_labels = t.cat(all_labels, dim=0)
        
        # 评估结果
        preds_np = all_preds.numpy()
        labels_np = all_labels.numpy()
        
        accuracy = accuracy_score(labels_np, preds_np)
        f1_macro = f1_score(labels_np, preds_np, average='macro')
        f1_micro = f1_score(labels_np, preds_np, average='micro')
        
        print(f"{mask_name}结果: Accuracy={accuracy:.4f}, "
              f"F1-Macro={f1_macro:.4f}, "
              f"F1-Micro={f1_micro:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'preds': preds_np,
            'labels': labels_np
        }
    
    # 运行推理
    print("开始Zero-shot推理...")
    
    # 验证集推理
    val_results = inference(val_mask, "验证集")
    
    # 测试集推理
    test_results = inference(test_mask, "测试集")
    
    # 保存结果
    results = {
        'val': val_results,
        'test': test_results
    }
    
    with open('zeroshot_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n=== 最终结果 ===")
    print(f"验证集: Accuracy={val_results['accuracy']:.4f}")
    print(f"测试集: Accuracy={test_results['accuracy']:.4f}")
    print("\n结果已保存到 zeroshot_results.pkl")

if __name__ == '__main__':
    main()
