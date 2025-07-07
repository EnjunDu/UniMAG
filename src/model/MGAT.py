import math
# from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.nn.inits import uniform, glorot, zeros

class GraphGAT(MessagePassing):
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		super(GraphGAT, self).__init__(aggr=aggr, **kwargs)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.dropout = 0.2

		self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()
		self.is_get_attention = False

	def reset_parameters(self):
		uniform(self.in_channels, self.weight)
		uniform(self.in_channels, self.bias)


	def forward(self, x, edge_index, size=None):
		if size is None:
			edge_index, _ = remove_self_loops(edge_index)
		x = x.unsqueeze(-1) if x.dim() == 1 else x
		x = torch.matmul(x, self.weight)

		return self.propagate(edge_index, size=size, x=x)

	def message(self, edge_index_i, x_i, x_j, size_i, edge_index, size):
		#print(edge_index_i, x_i, x_j, size_i, edge_index, size)
      
		# Compute attention coefficients.
		x_i = x_i.view(-1, self.out_channels)
		x_j = x_j.view(-1, self.out_channels)
		inner_product = torch.mul(x_i, F.leaky_relu(x_j)).sum(dim=-1)

		# gate
		row, col = edge_index
		deg = degree(row, size[0], dtype=x_i.dtype)
		deg_inv_sqrt = deg[row].pow(-0.5)
		tmp = torch.mul(deg_inv_sqrt, inner_product)
		gate_w = torch.sigmoid(tmp)
		# gate_w = F.dropout(gate_w, p=self.dropout, training=self.training)

		# attention
		tmp = torch.mul(inner_product, gate_w)
		attention_w = softmax(tmp, edge_index_i, num_nodes=size_i)
		#attention_w = F.dropout(attention_w, p=self.dropout, training=self.training)
		return torch.mul(x_j, attention_w.view(-1, 1))

	def update(self, aggr_out):
		if self.bias is not None:
			aggr_out = aggr_out + self.bias
		if self.normalize:
			aggr_out = F.normalize(aggr_out, p=2, dim=-1)
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)

class MGAT(torch.nn.Module):
    def __init__(self, v_feat_dim, t_feat_dim, num_nodes, num_layers, dim_x, v_dim):
        super(MGAT, self).__init__()
        
        # self.batch_size = batch_size
        # self.num_user = num_user
        # self.num_item = num_item

        # self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        # self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        
        # v_feat, a_feat, t_feat = features
        # self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        # self.a_feat = torch.tensor(a_feat, dtype=torch.float).cuda()
        # self.t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()
        self.v_dim = v_dim
        self.v_gnn = GNN(v_feat_dim, dim_x, num_layers, dim_latent=256)
        # self.a_gnn = GNN(self.a_feat, self.edge_index, batch_size, num_user, num_item, dim_x, dim_latent=128)
        self.t_gnn = GNN(t_feat_dim, dim_x, num_layers, dim_latent=100)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_nodes, dim_x), requires_grad=True)).cuda()

        #self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_nodes, dim_x))).cuda()

    # def forward(self, user_nodes, pos_items, neg_items):
    #     v_rep = self.v_gnn(self.id_embedding)
    #     a_rep = self.a_gnn(self.id_embedding)
    #     t_rep = self.t_gnn(self.id_embedding)
    #     representation = (v_rep + a_rep + t_rep) / 3 #torch.max_pool2d((v_rep, a_rep, t_rep))#max()#torch.cat((v_rep, a_rep, t_rep), dim=1)
    #     self.result_embed = representation
    #     user_tensor = representation[user_nodes]
    #     pos_tensor = representation[pos_items]
    #     neg_tensor = representation[neg_items]
    #     pos_scores = torch.sum(user_tensor * pos_tensor, dim=1)
    #     neg_tensor = torch.sum(user_tensor * neg_tensor, dim=1)
    #     return pos_scores, neg_tensor
    def forward(self, feat, edge_index):
        batch_nodes = torch.unique(edge_index)  # 获取所有参与计算的节点
        sub_id_embedding = self.id_embedding[batch_nodes]  # 选择当前子图的id_embedding

        v_rep = self.v_gnn(feat[:, :self.v_dim], sub_id_embedding, edge_index)
        # a_rep = self.a_gcn(feat, sub_id_embedding, edge_index)

        # # self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        t_rep = self.t_gnn(feat[:, self.v_dim:], sub_id_embedding, edge_index)
        
        # representation = (v_rep+a_rep+t_rep)/3
        representation = (v_rep+t_rep)/2
        self.result = representation
        return representation

    def reset_parameters(self):
        """重置所有可学习参数"""
        # 重置视觉GNN分支
        self.v_gnn.reset_parameters()
        
        # 重置文本GNN分支
        self.t_gnn.reset_parameters()
        
        # 重置ID嵌入层
        nn.init.xavier_normal_(self.id_embedding)
        
        # 重置结果嵌入层（如果已注册为Parameter）
        if isinstance(self.result_embed, nn.Parameter):
            nn.init.xavier_normal_(self.result_embed)
        # 如果result_embed是普通Tensor但需要重置
        elif hasattr(self, 'result_embed') and self.result_embed.requires_grad:
            nn.init.xavier_normal_(self.result_embed)

class GNN(torch.nn.Module):
    def __init__(self, dim_feat, dim_id, num_layers, dim_latent=None):
        super(GNN, self).__init__()
        # self.batch_size = batch_size
        # self.num_user = num_user
        # self.num_item = num_item
        self.num_layers = num_layers
        self.dim_id = dim_id
        self.dim_feat = dim_feat#features.size(1)
        self.dim_latent = dim_latent
        # self.edge_index = edge_index
        # self.features = features

        # self.preference = nn.Embedding(num_user, self.dim_latent)
        # nn.init.xavier_normal_(self.preference.weight).cuda()
        if self.dim_latent:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)

            self.conv_embed_1 = GraphGAT(self.dim_latent, self.dim_latent, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 
        else:
            #self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = GraphGAT(self.dim_feat, self.dim_feat, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight)


        if num_layers >= 2:
            self.conv_embed_2 = GraphGAT(self.dim_id, self.dim_id, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            self.g_layer2 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer2.weight)
        if num_layers >= 3:
            self.conv_embed_3 = GraphGAT(self.dim_id, self.dim_id, aggr='add')
            nn.init.xavier_normal_(self.conv_embed_3.weight)
            self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer3.weight)
            self.g_layer3 = nn.Linear(self.dim_id, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer3.weight)

    def forward(self, features, id_embedding, edge_index):

        x = torch.tanh(self.MLP(features)) if self.dim_latent else features
        # x = torch.cat((self.preference.weight, temp_features), dim=0)
        x = F.normalize(x).cuda()

        #layer-1
        h = F.leaky_relu(self.conv_embed_1(x, edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding
        x_1 = F.leaky_relu(self.g_layer1(h)+x_hat)
        # return x_1
        # # layer-2
        # h = F.leaky_relu(self.conv_embed_2(x_1, edge_index, None))
        # x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding
        # x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        # x = torch.cat((x_1, x_2), dim=1)

        # return x
        if self.num_layers == 1:
            x = x_1
            return x
        
        # layer-2
        h = F.leaky_relu(self.conv_embed_2(x_1, edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_1)) + id_embedding
        x_2 = F.leaky_relu(self.g_layer2(h)+x_hat)

        if self.num_layers == 2:
            x = torch.cat((x_1, x_2), dim=1)
            return x
            
        h = F.leaky_relu(self.conv_embed_2(x_2, edge_index, None))
        x_hat = F.leaky_relu(self.linear_layer2(x_2)) + id_embedding
        x_3 = F.leaky_relu(self.g_layer3(h)+x_hat)
        
        if self.num_layers == 3:
            x = torch.cat((x_1, x_2, x_3), dim=1)
            return x
    def reset_parameters(self):
        """重置所有可学习参数"""
        # 重置MLP层参数（如果存在）
        if hasattr(self, 'MLP'):
            nn.init.xavier_normal_(self.MLP.weight)
            if self.MLP.bias is not None:
                nn.init.zeros_(self.MLP.bias)
        
        # 重置第一层GAT相关参数
        nn.init.xavier_normal_(self.conv_embed_1.weight)
        nn.init.xavier_normal_(self.linear_layer1.weight)
        nn.init.zeros_(self.linear_layer1.bias) if self.linear_layer1.bias is not None else None
        nn.init.xavier_normal_(self.g_layer1.weight)
        nn.init.zeros_(self.g_layer1.bias) if self.g_layer1.bias is not None else None
        
        # 重置第二层GAT相关参数
        if self.num_layers >= 2:
            nn.init.xavier_normal_(self.conv_embed_2.weight)
            nn.init.xavier_normal_(self.linear_layer2.weight)
            nn.init.zeros_(self.linear_layer2.bias) if self.linear_layer2.bias is not None else None
            nn.init.xavier_normal_(self.g_layer2.weight)
            nn.init.zeros_(self.g_layer2.bias) if self.g_layer2.bias is not None else None
            if hasattr(self.conv_embed_2, 'reset_parameters'):
                self.conv_embed_2.reset_parameters()
        if self.num_layers >= 3:
             nn.init.xavier_normal_(self.conv_embed_3.weight)
             nn.init.xavier_normal_(self.linear_layer3.weight)
             nn.init.zeros_(self.linear_layer3.bias) if self.linear_layer3.bias is not None else None
             nn.init.xavier_normal_(self.g_layer3.weight)
             nn.init.zeros_(self.g_layer3.bias) if self.g_layer3.bias is not None else None
             if hasattr(self.conv_embed_3, 'reset_parameters'):
                self.conv_embed_3.reset_parameters()

        # 重置图注意力层的其他参数（如果存在）
        if hasattr(self.conv_embed_1, 'reset_parameters'):
            self.conv_embed_1.reset_parameters()
        
