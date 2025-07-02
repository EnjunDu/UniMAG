import dgl
import torch
from torch_geometric.data import Data
import numpy as np
from models import GCN, GraphSAGE, GAT
import torch.nn as nn
import torch.optim as optim
def split_graph(nodes_num, train_ratio=0.6, val_ratio=0.2, fewshots=False, label=None):
    # 划分数据集
    indices = np.random.permutation(nodes_num)
    if not fewshots:
        train_size = int(nodes_num * train_ratio)
        val_size = int(nodes_num * val_ratio)

        train_mask = torch.zeros(nodes_num, dtype=torch.bool)
        val_mask = torch.zeros(nodes_num, dtype=torch.bool)
        test_mask = torch.zeros(nodes_num, dtype=torch.bool)

        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask
    
def load_data(graph_path, emb_path, train_ratio, val_ratio, fewshots=False, self_loop=True, undirected=True):
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
    x = torch.load(emb_path)
    y = graph.ndata["label"]
    # 划分数据集
    if "train_mask" in graph.ndata:
        train_mask = graph.ndata["train_mask"]
        val_mask = graph.ndata["val_mask"]
        test_mask = graph.ndata["test_mask"]
    else:
        train_mask, val_mask, test_mask = split_graph(graph.num_nodes(), train_ratio, val_ratio, fewshots, y)
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

class NodeClassifier(nn.Module):
    # 分类头
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

class GNNModel(nn.Module):
    # 嵌入+分类头
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x, edge_index):
        x = self.encoder(x, edge_index)
        out = self.classifier(x)
        return out

def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    return correct.item() / mask.sum().item()

def train_and_eval(config, model, data, run_id=0):
    model.reset_parameters() if hasattr(model, 'reset_parameters') else None
    model = model.to(config.device)
    data = data.to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.task.lr, weight_decay=config.task.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    final_test_acc = 0

    for epoch in range(config.task.n_epochs):
        model.train()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, data, data.val_mask)
        test_acc = evaluate(model, data, data.test_mask)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc

        if epoch % 10 == 0 or epoch == config.task.n_epochs - 1:
            print(f"[Run {run_id+1}] Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}")

    return best_val_acc, final_test_acc

def run_nc(config):
    print(config)
    np.random.seed(config.seed)
    # 1.加载数据
    data = load_data(config.dataset.graph_path, config.dataset.emb_path, config.dataset.train_ratio, config.dataset.val_ratio, config.task.fewshots, config.task.self_loop, config.task.undirected).to(config.device)
    # 2.构建模型
    num_classes = data.y.max().item() + 1
    if config.model.name =="GAT":
        encoder = GAT(in_dim=data.x.size(1), hidden_dim=config.model.hidden_dim, num_layers=config.model.num_layers, heads=config.model.heads, dropout=config.model.dropout)

    classifier = NodeClassifier(in_dim=config.model.hidden_dim, num_classes=num_classes)
    model = GNNModel(encoder, classifier)
    # 3.训练 & 测试
    accs = []
    for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        best_val, final_test = train_and_eval(config, model, data, run)
        accs.append(final_test)
    print(f"Test accs: {[round(a * 100, 2) for a in accs]}")
    print(f"Average Test Accuracy over {config.task.n_runs} runs: {np.mean(accs) * 100:.2f}%")