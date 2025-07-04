# models/encoder.py
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        # 输入层
        self.convs.append(GCNConv(in_dim, hidden_dim))
        # 隐藏层
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # 输出嵌入，分类器在外部接

class GraphSAGE(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout

        # 输入层
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        # 隐藏层
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # 输出嵌入

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads, dropout):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout))
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x