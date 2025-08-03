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
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
def split_edge(graph, val_ratio=0.2, test_ratio=0.2, num_neg=150, path=None):
    if os.path.exists(os.path.join(path, f'edge_split_{num_neg}.pt')):
        edge_split = torch.load(os.path.join(path, f'edge_split_{num_neg}.pt'))
    else:
        # edges = np.arange(graph.num_edges())
        # edges = np.random.permutation(edges)
        # edges = torch.arange(graph.num_edges()) 
        edges = torch.randperm(graph.num_edges()) 

        source, target = graph.edges()

        val_size = int(len(edges) * val_ratio)
        test_size = int(len(edges) * test_ratio)
        test_source, test_target = source[edges[:test_size]], target[edges[:test_size]]
        val_source, val_target = source[edges[test_size:test_size + val_size]], target[edges[test_size:test_size + val_size]]
        train_source, train_target = source[edges[test_size + val_size:]], target[edges[test_size + val_size:]]

        val_target_neg = torch.randint(low=0, high=graph.num_nodes(), size=(len(val_source), int(num_neg)))
        test_target_neg = torch.randint(low=0, high=graph.num_nodes(), size=(len(test_source), int(num_neg)))

        edge_split = {'train': {'source_node': train_source, 'target_node': train_target},
            'valid': {'source_node': val_source, 'target_node': val_target,
                    'target_node_neg': val_target_neg},
            'test': {'source_node': test_source, 'target_node': test_target,
                    'target_node_neg': test_target_neg}}
        if os.path.exists(path):  
            torch.save(edge_split, os.path.join(path, f'edge_split_{num_neg}.pt'))

    return edge_split

        
def load_data(graph_path, v_emb_path, t_emb_path, val_ratio=0.1, test_ratio=0.2, num_neg=150, path=None, fewshots=False, self_loop=False, undirected=False):
    graph = dgl.load_graphs(graph_path)[0][0]
    # 自环、无向图
    if undirected:
        graph = dgl.add_reverse_edges(graph)
    if self_loop:
        graph = graph.remove_self_loop().add_self_loop()
    # 嵌入、标签
    # v_x = torch.load(v_emb_path)
    # t_x = torch.load(t_emb_path)
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
    # 划分数据集
    edge_split = split_edge(graph, val_ratio=val_ratio, test_ratio=test_ratio, num_neg=num_neg, path=path)

    train_edges = torch.stack(
        (edge_split['train']['source_node'], edge_split['train']['target_node']), 
        dim=1
    ).t()
    adj_t = SparseTensor.from_edge_index(train_edges).t()
    adj_t = adj_t.to_symmetric()
    src, dst = graph.edges()
    edge_index = torch.stack([src, dst], dim=0)
    return Data(x=x, v_dim=v_x.size(1), t_dim=t_x.size(1), edge_split=edge_split, adj_t=edge_index, edge_index=edge_index)

class Linear_v_t(nn.Module):
    def __init__(self, in_channels, out_channels_v, out_channels_t):
        super(Linear_v_t, self).__init__()

        self.v = nn.Linear(in_channels, out_channels_v)
        self.t = nn.Linear(in_channels, out_channels_t)

    def reset_parameters(self):
        self.v.reset_parameters()
        self.t.reset_parameters()

    def forward(self, x_v, x_t):
        out_v = self.v(x_v)
        out_t = self.t(x_t)
        return out_v, out_t

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        # 计算两个节点的嵌入相似度
        x = x_i * x_j  # 点乘计算相似度

        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return torch.sigmoid(x)


def set_seed(seed: int):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, predictor, x, adj_t, edge_split, num_neg=1000, k_list=[1,3,10], batch_size=2048):
    model.eval()
    predictor.eval()

    # 获取验证集的边
    source_edge = edge_split['valid']['source_node'].to(x.device)  # [num_pos]
    target_edge = edge_split['valid']['target_node'].to(x.device)  # [num_pos]
    neg_target_edge = edge_split['valid']['target_node_neg'].to(x.device)  # [num_pos, num_neg]

    # 获取节点的嵌入
    emb, out_v, out_t = model(x, adj_t)

    # 分批计算正样本得分
    pos_preds = []
    for batch in DataLoader(range(source_edge.size(0)), batch_size):

        src = source_edge[batch]
        dst = target_edge[batch]

        pos_preds.append(predictor(emb[src], emb[dst]).squeeze().cpu()) 
    pos_out = torch.cat(pos_preds, dim=0)  
    # 分批计算负样本得分
    neg_preds = []
    num_pos = source_edge.size(0)

    flat_source = source_edge.view(-1, 1).repeat(1, num_neg).view(-1) 
    flat_neg_target = neg_target_edge.view(-1) 
    for batch in DataLoader(range(flat_source.size(0)), batch_size):
        src = flat_source[batch]
        neg = flat_neg_target[batch]
        neg_preds.append(predictor(emb[src], emb[neg]).squeeze().cpu())  
    neg_out = torch.cat(neg_preds, dim=0).view(-1, num_neg)
    # 计算评估指标
    pos_out = pos_out.unsqueeze(1) 
    all_scores = torch.cat([pos_out, neg_out], dim=1)  
    
    ranks = (all_scores >= pos_out).sum(dim=1) 
    # 计算各K值的Hits@K
    hits_results = {}
    for k in sorted(k_list):
        hits_at_k = (ranks <= k).float().mean().item()
        hits_results[f'Hits@{k}'] = hits_at_k
    
    # 计算MRR
    mrr = (1.0 / ranks.float()).mean().item()
    
    return {**hits_results, 'MRR': mrr}
@torch.no_grad()
def test(model, predictor, x, adj_t, edge_split, num_neg=1000, k_list=[1,3,10], batch_size=2048):
    model.eval()
    predictor.eval()

    # 获取验证集的边
    source_edge = edge_split['test']['source_node'].to(x.device)  # [num_pos]
    target_edge = edge_split['test']['target_node'].to(x.device)  # [num_pos]
    neg_target_edge = edge_split['test']['target_node_neg'].to(x.device)  # [num_pos, num_neg]

    # 获取节点的嵌入
    emb, out_v, out_t = model(x, adj_t)

    # 分批计算正样本得分
    pos_preds = []
    for batch in DataLoader(range(source_edge.size(0)), batch_size):

        src = source_edge[batch]
        dst = target_edge[batch]

        pos_preds.append(predictor(emb[src], emb[dst]).squeeze().cpu())  
    pos_out = torch.cat(pos_preds, dim=0)  # [num_pos]
    # 分批计算负样本得分
    neg_preds = []
    num_pos = source_edge.size(0)

    flat_source = source_edge.view(-1, 1).repeat(1, num_neg).view(-1) 
    flat_neg_target = neg_target_edge.view(-1)  
    for batch in DataLoader(range(flat_source.size(0)), batch_size):
        src = flat_source[batch]
        neg = flat_neg_target[batch]
        neg_preds.append(predictor(emb[src], emb[neg]).squeeze().cpu()) 
    neg_out = torch.cat(neg_preds, dim=0).view(-1, num_neg)
    # 计算评估指标
    pos_out = pos_out.unsqueeze(1)  
    all_scores = torch.cat([pos_out, neg_out], dim=1) 

    ranks = (all_scores >= pos_out).sum(dim=1)  
    # 计算各K值的Hits@K
    hits_results = {}
    for k in sorted(k_list):
        hits_at_k = (ranks <= k).float().mean().item()
        hits_results[f'Hits@{k}'] = hits_at_k
    
    # 计算MRR
    mrr = (1.0 / ranks.float()).mean().item()
    
    return {**hits_results, 'MRR': mrr}

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

def train(model, predictor, data, config, x, adj_t, edge_split, optimizer, batch_size):

    model.train()
    predictor.train()
    linear_v_t = Linear_v_t(config.model.hidden_dim, data.v_dim, data.t_dim).to(config.device)
    linear_v_t.train()
    source_edge = edge_split['train']['source_node']
    target_edge = edge_split['train']['target_node']

    total_loss = total_examples = 0
    for batch in DataLoader(range(source_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        emb, out_v, out_t = model(x, adj_t)
        out_v, out_t = linear_v_t(out_v, out_t)
        loss_v = infoNCE_loss(out_v, data.x[:, :data.v_dim])
        loss_t = infoNCE_loss(out_t, data.x[:, data.v_dim:])

        pos_out = predictor(emb[source_edge[batch]], emb[target_edge[batch]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        dst_neg = torch.randint(0, x.size(0), source_edge[batch].size(), dtype=torch.long,
                                device=x.device)
        neg_out = predictor(emb[source_edge[batch]], emb[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss + config.task.lambda_v * loss_v + config.task.lambda_t * loss_t
        loss.backward()
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    return total_loss / total_examples


def run_lp(config):
    print(config)
    np.random.seed(config.seed)
    # 1.加载数据
    data = load_data(config.dataset.graph_path, config.dataset.v_emb_path, config.dataset.t_emb_path, config.dataset.lp_val_ratio, config.dataset.lp_test_ratio, config.dataset.num_neg, config.dataset.edge_split_path).to(config.device)

    # 2.构建模型
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
    encoder.to(config.device)
    predictor = LinkPredictor(in_channels=config.model.hidden_dim, hidden_channels=config.task.predictor_hidden, out_channels=1, num_layers=config.task.predictor_layers, dropout=config.task.predictor_dropout).to(config.device)
    # 3.训练 & 测试
    accs = []
    best_mrr = 0
    final_mrr_test = 0
    for run in range(config.task.n_runs):
        set_seed(config.seed + run)
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=config.task.lr)
        encoder.reset_parameters() if hasattr(encoder, 'reset_parameters') else None
        predictor.reset_parameters() if hasattr(predictor, 'reset_parameters') else None
        best_mrr = 0
        final_mrr_test = 0
        # 训练
        for epoch in range(config.task.n_epochs):
            train_loss = train(encoder, predictor, data, config, data.x, data.adj_t, data.edge_split, optimizer, batch_size=config.task.batch_size)
            print(f"[Run {run+1}] Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
            if epoch % 1 == 0:
                results = evaluate(encoder, predictor, data.x, data.adj_t, data.edge_split, config.dataset.num_neg, k_list=config.task.k_list)
                print("[VAL]")
                print(results)
                if results["MRR"] > best_mrr:
                    best_mrr = results["MRR"]
                    results = test(encoder, predictor, data.x, data.adj_t, data.edge_split, config.dataset.num_neg, k_list=config.task.k_list)
                    final_mrr_test = results["MRR"]
                    print(f"[TEST]")
                    print(results)
        print(f"run {run} test mrr: {final_mrr_test}")
        accs.append(final_mrr_test)
    print(f"Average MRR over {config.task.n_runs} runs: {np.mean(accs) * 100:.2f}")
