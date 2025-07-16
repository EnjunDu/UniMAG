import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_geometric.utils import scatter_
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.inits import uniform

class BaseModel(MessagePassing):
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		super(BaseModel, self).__init__(aggr=aggr, **kwargs)
		self.aggr = aggr
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

		self.reset_parameters()

	def reset_parameters(self):
		uniform(self.in_channels, self.weight)

	def forward(self, x, edge_index, size=None):
		x = torch.matmul(x, self.weight)
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j, edge_index, size):
		return x_j

	def update(self, aggr_out):
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
     
class GCN(torch.nn.Module):
    def __init__(self, dim_feat, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None):
        super(GCN, self).__init__()
        self.dim_id = dim_id# dim_x
        self.dim_feat = dim_feat# 初始节点特征维度
        self.dim_latent = dim_latent# ？
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id

        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)     
            nn.init.xavier_normal_(self.g_layer1.weight)              
        if num_layer >= 2:
            self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)
        if num_layer >= 3:
            self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)  


    def reset_parameters(self):
        if self.dim_latent:
            nn.init.xavier_normal_(self.MLP.weight)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            nn.init.xavier_normal_(self.g_layer1.weight)
        else:
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            nn.init.xavier_normal_(self.g_layer1.weight)
        if self.num_layer >= 2:
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            nn.init.xavier_normal_(self.g_layer2.weight)
        if self.num_layer >= 3:
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            nn.init.xavier_normal_(self.g_layer3.weight)
    def forward(self, features, id_embedding, edge_index):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = temp_features
        x = F.normalize(x).cuda()

        h = F.leaky_relu(self.conv_embed_1(x, edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5 
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)
        if self.num_layer >= 2:
            h = F.leaky_relu(self.conv_embed_2(x, edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)
        if self.num_layer >= 3:
            h = F.leaky_relu(self.conv_embed_3(x, edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
            x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)

        return x


class Net(torch.nn.Module):
    def __init__(self, v_feat_dim, t_feat_dim, num_nodes, aggr_mode, concate, num_layer, has_id, dim_x, v_dim):
        super(Net, self).__init__()
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.weight = torch.tensor([[1.0],[-1.0]]).cuda()
        

        self.num_modal = 0
        self.v_dim = v_dim
        # self.v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        self.v_gcn = GCN(v_feat_dim, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256)

        # self.a_feat = torch.tensor(a_feat,dtype=torch.float).cuda()
        # self.a_gcn = GCN(a_feat_dim, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        # self.t_feat = torch.tensor(t_feat,dtype=torch.float).cuda()
        self.t_gcn = GCN(t_feat_dim, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        # self.words_tensor = torch.tensor(words_tensor, dtype=torch.long).cuda()
        # self.word_embedding = nn.Embedding(torch.max(self.words_tensor[1])+1, 128)
        # nn.init.xavier_normal_(self.word_embedding.weight) 
        # self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, 128, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_nodes, dim_x), requires_grad=True)).cuda()
        self.result = nn.init.xavier_normal_(torch.rand((num_nodes, dim_x))).cuda()

    def reset_parameters(self):
        self.v_gcn.reset_parameters()
        # self.a_gcn.reset_parameters()
        self.t_gcn.reset_parameters()
        nn.init.xavier_normal_(self.id_embedding)
        nn.init.xavier_normal_(self.result)
    def forward(self, feat, edge_index):
        batch_nodes = torch.unique(edge_index)  # 获取所有参与计算的节点
        sub_id_embedding = self.id_embedding[batch_nodes]  # 选择当前子图的id_embedding
        v_rep = self.v_gcn(feat[:, :self.v_dim], sub_id_embedding, edge_index)
        # a_rep = self.a_gcn(feat, sub_id_embedding, edge_index)

        # # self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        t_rep = self.t_gcn(feat[:, self.v_dim:], sub_id_embedding, edge_index)
        
        # representation = (v_rep+a_rep+t_rep)/3
        representation = (v_rep+t_rep)/2
        self.result = representation
        return representation, v_rep, t_rep

