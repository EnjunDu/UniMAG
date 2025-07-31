import dgl
import torch
from torch_geometric.data import Data
import numpy as np
from model.models import GCN, GraphSAGE, GAT, MLP
from model.MMGCN import Net
from model.MGAT import MGAT
from model.REVGAT import RevGAT
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
def split_graph(nodes_num, train_ratio=0.6, val_ratio=0.2, fewshots=False, label=None):
    # 划分数据集
    indices = np.random.permutation(nodes_num)
    if not fewshots:
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        train_mask = torch.zeros(nodes_num, dtype=torch.bool)
        val_mask = torch.zeros(nodes_num, dtype=torch.bool)
        test_mask = torch.zeros(nodes_num, dtype=torch.bool)
        # indices = torch.Tensor(indices).to(torch.long)
        indices = torch.from_numpy(indices).to(torch.long)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
    return train_mask, val_mask, test_mask
    
def load_data(graph_path, v_emb_path, t_emb_path, train_ratio, val_ratio, fewshots=False, self_loop=True, undirected=True):
    graph = dgl.load_graphs(graph_path)[0][0]
    # 自环、无向图
    if undirected:
        graph = dgl.add_reverse_edges(graph)
    if self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    # 边
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
        # 嵌入、标签
    # v_x = torch.load(v_emb_path)
    # t_x = torch.load(t_emb_path)
    v_x = torch.from_numpy(np.load(v_emb_path)).to(torch.float32)
    t_x = torch.from_numpy(np.load(t_emb_path)).to(torch.float32)
    max_val_v = torch.finfo(v_x.dtype).max  # 获取该数据类型最大有限值
    min_val_v = torch.finfo(v_x.dtype).min  # 获取最小有限值
    max_val_t = torch.finfo(t_x.dtype).max  # 获取该数据类型最大有限值
    min_val_t = torch.finfo(t_x.dtype).min  # 获取最小有限值
    v_x = torch.nan_to_num(v_x, nan=0.0, posinf=max_val_v, neginf=min_val_v)
    t_x = torch.nan_to_num(t_x, nan=0.0, posinf=max_val_t, neginf=min_val_t)
    x = torch.cat([v_x, t_x], dim=1)
    print("输入数据统计:")
    print(f"Min: {x.min()}, Max: {x.max()}, Mean: {x.mean()}, Std: {x.std()}")
    print(f"NaN in x: {torch.isnan(x).any()}, Inf in x: {torch.isinf(x).any()}")
    print(x.shape)
    y = graph.ndata["label"]
    # 划分数据集
    if "train_mask" in graph.ndata:
        train_mask = graph.ndata["train_mask"].to(torch.bool)
        val_mask = graph.ndata["val_mask"].to(torch.bool)
        test_mask = graph.ndata["test_mask"].to(torch.bool)
    else:
        train_mask, val_mask, test_mask = split_graph(graph.num_nodes(), train_ratio, val_ratio, fewshots, y)
    return Data(x=x, v_dim=v_x.size(1), t_dim=t_x.size(1), edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

class NodeClassifier(nn.Module):
    # 分类头
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def reset_parameters(self):
        self.linear.reset_parameters()

    def forward(self, x):
        return self.linear(x)

class GNNModel(nn.Module):
    # 嵌入+分类头
    def __init__(self, encoder, classifier, dim_hidden, dim_v, dim_t):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.decoder_v = nn.Linear(dim_hidden, dim_v)
        self.decoder_t = nn.Linear(dim_hidden, dim_t)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        x, x_v, x_t = self.encoder(x, edge_index)
        out = self.classifier(x)
        out_v = self.decoder_v(x_v)
        out_t = self.decoder_t(x_t)
        return out, out_v, out_t

def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
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

def infoNCE_loss(out, orig_features, tau=0.07):
    """
    InfoNCE损失实现
    - out: 模型输出的特征嵌入 (batch_size, emb_dim)
    - orig_features: 原始特征 (batch_size, feat_dim)
    - tau: 温度系数
    """
    # 1. 特征归一化
    out_norm = F.normalize(out, p=2, dim=1)
    orig_norm = F.normalize(orig_features, p=2, dim=1)
    
    # 2. 计算相似度矩阵
    sim_matrix = torch.mm(out_norm, orig_norm.t()) / tau  # [batch_size, batch_size]
    
    # 3. 创建标签（对角线为正样本）
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    
    # 4. 使用交叉熵损失
    loss = F.cross_entropy(sim_matrix, labels)
    
    return loss

@torch.no_grad()
def evaluate(model, data, mask, config, num_classes):
    model.eval()
    
    # 创建NeighborLoader进行子图采样
    loader = NeighborLoader(
        data,
        num_neighbors=[config.task.num_neighbors] * config.model.num_layers,  # 每层采样邻居数
        input_nodes=mask,  # 只对评估节点采样
        batch_size=config.task.batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    for batch in loader:  # 迭代子图
        batch = batch.to(config.device)
        out, out_v, out_t = model(batch.x, batch.edge_index)
        pred = out.argmax(dim=1)
        # correct += (pred[batch.train_mask] == batch.y[batch.train_mask]).sum().item()
        # total += batch.train_mask.sum().item()
        pred = pred[:batch.batch_size]
        y_true = batch.y[:batch.batch_size]

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

def train_and_eval(config, model, data, run_id=0):
    model.reset_parameters() if hasattr(model, 'reset_parameters') else None
    model = model.to(config.device)
    data = data.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.task.lr, weight_decay=config.task.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # 创建NeighborLoader进行子图采样
    train_loader = NeighborLoader(
        data,
        num_neighbors=[config.task.num_neighbors] * config.model.num_layers,  # 每层采样邻居数
        input_nodes=data.train_mask,  # 只对训练节点采样
        batch_size=config.task.batch_size,
        shuffle=True,
        num_workers=4,
    )
    best_val_acc = 0
    best_val_f1 = 0
    final_test_acc = 0
    final_test_f1 = 0
    for epoch in range(config.task.n_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:  # 迭代子图
            batch = batch.to(config.device)
            out, out_v, out_t = model(batch.x, batch.edge_index)
            # 只计算子图中的训练节点损失
            bsz = batch.batch_size
            loss_task = criterion(out[:bsz], batch.y[:bsz])
            loss_v = infoNCE_loss(out_v[:bsz], batch.x[:bsz, :data.v_dim])
            loss_t = infoNCE_loss(out_t[:bsz], batch.x[:bsz, data.v_dim:])
            # print(loss_v)
            # 总损失：任务损失 + 加权的对比损失
            loss = loss_task + config.task.lambda_v * loss_v + config.task.lambda_t * loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.train_mask.sum().item()

        loss = total_loss / len(train_loader)
        print(loss)
        val_f1, val_acc = evaluate(model, data, data.val_mask, config, data.y.max().item() + 1)
        test_f1, test_acc = evaluate(model, data, data.test_mask, config, data.y.max().item() + 1)

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

def run_nc(config):
    print(config)
    np.random.seed(config.seed)
    # 1.加载数据
    data = load_data(config.dataset.graph_path, config.dataset.v_emb_path, config.dataset.t_emb_path, config.dataset.train_ratio, config.dataset.val_ratio, config.task.fewshots, config.task.self_loop, config.task.undirected).to(config.device)
    # 2.构建模型
    num_classes = data.y.max().item() + 1
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
    classifier = NodeClassifier(in_dim=config.model.hidden_dim, num_classes=num_classes)
    model = GNNModel(encoder, classifier, config.model.hidden_dim, data.v_dim, data.t_dim)
    # 3.训练 & 测试
    accs = []
    f1s = []
    for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        best_val_acc, best_val_f1, final_test_acc, final_test_f1 = train_and_eval(config, model, data, run)
        accs.append(final_test_acc)
        f1s.append(final_test_f1)
    print(f"Test accs: {[round(a * 100, 2) for a in accs]}")
    print(f"Average Test Accuracy over {config.task.n_runs} runs: {np.mean(accs) * 100:.2f}%")
    print(f"Test f1s: {[round(a, 2) for a in f1s]}")
    print(f"Average Test F1 over {config.task.n_runs} runs: {np.mean(f1s):.2f}")
