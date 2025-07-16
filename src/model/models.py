# models/encoder.py
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.num_layers = num_layers

        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            input_dim = hidden_dim if i > 0 else in_dim
            self.linears.append(nn.Linear(input_dim, hidden_dim))

            if i < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)

        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for linear in self.linears:
            linear.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_index):
        h = x

        for i in range(self.num_layers - 1):
            h = F.relu(self.norms[i](self.linears[i](h)))
            h = self.dropout(h)
        x = self.linears[-1](h)
        # 模态级+节点级输出
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text


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
        
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 模态级+节点级输出
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text  # 输出嵌入，分类器在外部接

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
        
        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # 模态级+节点级输出
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, heads, dropout, att_dropout=0):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.dropout = dropout
        
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=att_dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=att_dropout))
        self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=att_dropout))

        self.vision_head = nn.Linear(hidden_dim, hidden_dim)
        self.text_head = nn.Linear(hidden_dim, hidden_dim)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # 模态级+节点级输出
        x_vision = F.dropout(F.relu(self.vision_head(x)), p=self.dropout, training=self.training)
        x_text = F.dropout(F.relu(self.text_head(x)), p=self.dropout, training=self.training)

        return x, x_vision, x_text
