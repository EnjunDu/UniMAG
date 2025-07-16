import dgl
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
from model.models import GCN, GraphSAGE, GAT, MLP
from model.MMGCN import Net
from model.MGAT import MGAT
from model.REVGAT import RevGAT
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.datasets import TUDataset

def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def convert_to_data_format(dataset, config):
    data_list = []
    
    # 计算划分比例
    train_ratio = config.task.train_ratio
    val_ratio = config.task.val_ratio
    test_ratio = 1 - train_ratio - val_ratio  # 测试集比例
    
    # 随机打乱数据集索引
    num_graphs = len(dataset)
    indices = torch.randperm(num_graphs)  # 打乱图的索引
    print(torch.randperm(5)) 
    
    # 计算每个划分集的大小
    train_size = int(train_ratio * num_graphs)
    val_size = int(val_ratio * num_graphs)
    test_size = num_graphs - train_size - val_size  # 剩下的就是测试集大小

    # 划分数据集
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 将图数据分配到训练集、验证集和测试集中
    train_data, val_data, test_data = [], [], []
    for i, graph in enumerate(dataset):
        # 提取图中的节点特征、边索引和标签
        x = graph.x  # 节点特征
        edge_index = graph.edge_index  # 边索引
        y = graph.y  # 图标签
        
        # 按照图级别划分数据集
        if i in train_indices:
            # 所有节点都用于训练
            train_data.append(Data(x=x, edge_index=edge_index, y=y))
        elif i in val_indices:
            # 所有节点都用于验证
            val_data.append(Data(x=x, edge_index=edge_index, y=y))
        else:
            # 所有节点都用于测试
            test_data.append(Data(x=x, edge_index=edge_index, y=y))
    
    # 返回字典形式的划分数据
    return {'train': train_data, 'val': val_data, 'test': test_data}

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
    
    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)

class GNNModel(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index, batch):
        node_emb, out_v, out_t  = self.encoder(x, edge_index)
        graph_emb = global_max_pool(node_emb, batch)
        out = self.classifier(graph_emb)
        return out, out_v, out_t

def calculate_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """
    计算Macro-F1分数
    Args:
        preds: 预测标签 [N]
        labels: 真实标签 [N]
        num_classes: 类别数量
    Returns:
        macro_f1: 平均F1分数
    """
    # 初始化混淆矩阵 [num_classes, num_classes]
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float32)
    
    # 填充混淆矩阵
    for p, l in zip(preds, labels):
        confusion_matrix[l, p] += 1
    
    # 计算每个类别的统计量
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        # True Positive
        tp = confusion_matrix[cls, cls]
        
        # False Positive (其他类预测为本类)
        fp = confusion_matrix[:, cls].sum() - tp
        
        # False Negative (本类预测为其他类)
        fn = confusion_matrix[cls, :].sum() - tp
        
        # 避免除零
        precision[cls] = tp / (tp + fp + 1e-10)
        recall[cls] = tp / (tp + fn + 1e-10)
        
        # F1计算
        f1[cls] = 2 * (precision[cls] * recall[cls]) / (precision[cls] + recall[cls] + 1e-10)
    
    # Macro-F1 (各类别平均)
    macro_f1 = f1.mean().item()
    return macro_f1

@torch.no_grad()
def evaluate(model, data_list, num_classes, config):
    model.eval()
    for data in data_list:
        data = data.to(config.device)
    loader = DataLoader(data_list, batch_size=config.task.batch_size, shuffle=True)
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for data in loader:
        out, out_v, out_t = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        y_true = data.y

        all_preds.append(pred.cpu())
        all_labels.append(y_true.cpu())

        correct += (pred == y_true).sum().item()
        total += y_true.size(0)

    # 计算指标
    accuracy = correct / total if total > 0 else 0.0
    
    # 计算F1分数

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    f1 = calculate_f1(all_preds, all_labels, num_classes)
    

    return f1, accuracy

def train_and_eval(config, model, data_all, run_id=0):
    model.reset_parameters() if hasattr(model, 'reset_parameters') else None
    model = model.to(config.device)
    data_list = data_all["train"]
    for data in data_list:
        data = data.to(config.device)

    optimizer = optim.Adam(model.parameters(), lr=config.task.lr, weight_decay=config.task.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    best_val_f1 = 0
    final_test_acc = 0
    final_test_f1 = 0
    loader = DataLoader(data_list, batch_size=config.task.batch_size, shuffle=True)
    for epoch in range(config.task.n_epochs):
        model.train()
        total_loss = 0
        for data in loader:
            out, out_v, out_t = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(total_loss)
        val_f1, val_acc = evaluate(model, data_all["val"], config.task.num_classes, config)
        test_f1, test_acc = evaluate(model, data_all["test"], config.task.num_classes, config)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_test_f1 = test_f1
        # if val_f1 > best_val_f1:
        #     best_val_acc = val_acc
        #     final_test_acc = test_acc
        #     final_test_f1 = test_f1

        if epoch % 10 == 0 or epoch == config.task.n_epochs - 1:
            print(f"[Run {run_id+1}] Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")

    return best_val_acc, best_val_f1, final_test_acc, final_test_f1

def run_gc(config):
    print(config)
    # np.random.seed(config.seed)
    set_seed(config.seed)
    # 1.加载数据
    dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')
    data_all = convert_to_data_format(dataset, config)
    data = data_all["train"][0]
    # 2.构建模型
    num_classes = config.task.num_classes
    if config.model.name =="MLP":
        encoder = MLP(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GAT":
        encoder = GAT(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, heads=config.model.heads, dropout=config.model.dropout, att_dropout=config.model.att_dropout)
    elif config.model.name =="GCN":
        encoder = GCN(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name =="GraphSAGE":
        encoder = GCN(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, dropout=config.model.dropout)
    elif config.model.name == "RevGAT":
        encoder = RevGAT(in_feats=data.x.size(1),n_hidden=config.model.hidden_dim, n_layers=config.model.num_layers, n_heads=config.model.heads, activation=F.relu, dropout=config.model.dropout)
    elif config.model.name == "MMGCN":
        encoder = Net(
            v_feat_dim=data.x.size(1),
            t_feat_dim=data.x.size(1),
            num_nodes=data.x.size(0),
            aggr_mode='mean',
            concate=False,
            num_layer=config.model.num_layers,
            has_id=True,
            dim_x=config.model.hidden_dim,
            v_dim=data.v_dim
        )
    elif config.model.name == "MGAT":
        encoder = MGAT(
            v_feat_dim=data.x.size(1),
            t_feat_dim=data.x.size(1),
            num_nodes=data.x.size(0),
            num_layers=config.model.num_layers,
            dim_x=config.model.hidden_dim,
            v_dim=data.v_dim
        )
        config.model.hidden_dim = config.model.hidden_dim * config.model.num_layers
    classifier = GraphClassifier(in_dim=config.model.hidden_dim, num_classes=num_classes)
    model = GNNModel(encoder, classifier)
    # 3.训练 & 测试
    accs = []
    f1s = []
    for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        best_val_acc, best_val_f1, final_test_acc, final_test_f1 = train_and_eval(config, model, data_all, run)
        accs.append(final_test_acc)
        f1s.append(final_test_f1)
    print(f"Test accs: {[round(a * 100, 2) for a in accs]}")
    print(f"Average Test Accuracy over {config.task.n_runs} runs: {np.mean(accs) * 100:.2f}%")
    print(f"Test f1s: {[round(a, 2) for a in f1s]}")
    print(f"Average Test F1 over {config.task.n_runs} runs: {np.mean(f1s):.2f}")