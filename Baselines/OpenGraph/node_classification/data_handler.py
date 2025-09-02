# -*- coding: utf-8 -*-
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch as torch  # use torch.load for .pt files
import torch.utils.data as data
# import torch_geometric.transforms as T  # 未使用，已注释
from model import InitialProjector
import os

class MultiDataHandler:
    def __init__(self, trn_datasets, tst_datasets):
        all_datasets = list(set(trn_datasets + tst_datasets))
        self.trn_handlers = []
        self.tst_handlers = []
        for data_name in all_datasets:
            trn_flag = data_name in trn_datasets
            tst_flag = data_name in tst_datasets
            handler = DataHandler(data_name, trn_flag, tst_flag)
            if trn_flag:
                self.trn_handlers.append(handler)
            if tst_flag:
                self.tst_handlers.append(handler)

    def make_joint_trn_loader(self):
        trn_data = TrnData(self.trn_handlers)
        self.trn_loader = data.DataLoader(trn_data, batch_size=1, shuffle=True, num_workers=0)
    
    def remake_initial_projections(self):
        for i in range(len(self.trn_handlers)):
            self.remake_one_initial_projection(i)
    
    def remake_one_initial_projection(self, idx):
        trn_handler = self.trn_handlers[idx]
        trn_handler.initial_projector = InitialProjector(trn_handler.asym_adj)


class DataHandler:
    def __init__(self, data_name, trn_flag, tst_flag):
        self.data_name = data_name
        self.trn_flag = trn_flag
        self.tst_flag = tst_flag
        self.use_pt_graph = False  # will be set in get_data_files()
        self.get_data_files()
        self.load_data()
    
    def get_data_files(self):
        """
        支持两种文件布局：
        1) 原布局（PKL）：adj_-1.pkl / label.pkl / mask_-1.pkl
        2) books-nc（PT）：nc_edges-nodeid.pt / labels-w-missing.pt / split.pt
        """
        predir = os.path.join(args.data_dir, self.data_name)

        # 先尝试 PT 风格（books-nc）
        edge_pt = os.path.join(predir, 'nc_edges-nodeid.pt')
        label_pt = os.path.join(predir, 'labels-w-missing.pt')
        split_pt = os.path.join(predir, 'split.pt')
        if os.path.exists(edge_pt) and os.path.exists(label_pt) and os.path.exists(split_pt):
            self.edge_file = edge_pt
            self.label_file = label_pt
            self.split_file = split_pt
            self.use_pt_graph = True
            return
        
        # 否则回退到 PKL 风格
        self.adj_file = os.path.join(predir, 'adj_-1.pkl')
        self.label_file = os.path.join(predir, 'label.pkl')
        self.mask_file = os.path.join(predir, 'mask_-1.pkl')
        self.use_pt_graph = False

    def load_one_file(self, filename):
        """
        仅用于 PKL 分支：读取 pickle 的 matrix / ndarray。
        """
        with open(filename, 'rb') as fs:
            ret = pickle.load(fs)
        # adjacency pkl 可能是密集/CSR/CSC 等，这里统一转 COO + float32
        if isinstance(ret, (np.ndarray, list)):
            ret = np.asarray(ret)
        if not isinstance(ret, coo_matrix):
            ret = sp.coo_matrix(ret)
        # 确保 float32
        if ret.dtype != np.float32:
            ret = ret.astype(np.float32)
        return ret

    def load_pt_edge_to_coo(self, edge_filename, symmetrize=True):
        """
        读取 .pt 的边列表(list of [u, v])并转成 COO 稀疏邻接矩阵（float32）。
        """
        edge_list = torch.load(edge_filename, weights_only=True)
        edges = np.asarray(edge_list, dtype=np.int64)  # [E, 2]
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError('edges must be shape [E, 2]')
        num_nodes = int(edges.max()) + 1

        if symmetrize:
            rev = edges[:, [1, 0]]
            edges = np.vstack([edges, rev])
            # 去重
            edges = np.unique(edges, axis=0)

        row, col = edges[:, 0], edges[:, 1]
        data = np.ones(row.shape[0], dtype=np.float32)
        adj = coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
        adj.sum_duplicates()
        return adj

    def normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        if mat.shape[0] == mat.shape[1]:
            return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
        else:
            tem = d_inv_sqrt_mat.dot(mat)
            col_degree = np.array(mat.sum(axis=0))
            d_inv_sqrt = np.reshape(np.power(col_degree, -0.5), [-1])
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            return tem.dot(d_inv_sqrt_mat).tocoo()

    def load_feats(self, filename):
        """
        仅用于 PKL 分支读取 label 和 mask。
        """
        try:
            with open(filename, 'rb') as fs:
                feats = pickle.load(fs)
        except Exception as e:
            print(filename + str(e))
            exit()
        return feats

    def unique_numpy(self, row, col):
        hash_vals = row * self.node_num + col
        hash_vals = np.unique(hash_vals).astype(np.int64)
        col = hash_vals % self.node_num
        row = (hash_vals - col).astype(np.int64) // self.node_num
        return row, col
    
    def make_torch_adj(self, mat):
        if mat.shape[0] == mat.shape[1]:
            # to symmetric (保留原逻辑：某些数据集如 ddi 特殊处理)
            if self.data_name in ['ddi']:
                _row = mat.row
                _col = mat.col
                row = np.concatenate([_row, _col]).astype(np.int64)
                col = np.concatenate([_col, _row]).astype(np.int64)
                data = mat.data
                data = np.concatenate([data, data]).astype(np.float32)
            else:
                row, col = mat.row, mat.col
                data = mat.data
            mat = coo_matrix((data, (row, col)), mat.shape)
            if getattr(args, 'selfloop', 0) == 1:
                mat = (mat + sp.eye(mat.shape[0])) * 1.0

        normed_asym_mat = self.normalize_adj(mat)
        row = t.from_numpy(normed_asym_mat.row).long()
        col = t.from_numpy(normed_asym_mat.col).long()
        idxs = t.stack([row, col], dim=0)
        vals = t.from_numpy(normed_asym_mat.data).float()
        shape = t.Size(normed_asym_mat.shape)
        asym_adj = t.sparse.FloatTensor(idxs, vals, shape)

        if mat.shape[0] == mat.shape[1]:
            return asym_adj, asym_adj
        else:
            # bipartite case (not used for books-nc, but keep original behavior)
            a = sp.csr_matrix((self.user_num, self.user_num))
            b = sp.csr_matrix((self.item_num, self.item_num))
            mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
            mat = (mat != 0) * 1.0
            if getattr(args, 'selfloop', 0) == 1:
                mat = (mat + sp.eye(mat.shape[0])) * 1.0
            mat = self.normalize_adj(mat)

            idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
            vals = t.from_numpy(mat.data.astype(np.float32))
            shape = t.Size(mat.shape)
            return t.sparse.FloatTensor(idxs, vals, shape), asym_adj

    def _load_pt_labels_and_masks(self):
        """
        从 PT 风格的 label/split 构造 labels 与 boolean masks。
        """
        raw_labels = torch.load(self.label_file, weights_only=True)  # list[int], len = N
        labels = np.asarray(raw_labels, dtype=np.int64)
        if labels.min() != 0:
            log(f'Class label starts from {labels.min()}')
            labels = labels - labels.min()

        split = torch.load(self.split_file, weights_only=True)  # dict: train_idx/val_idx/test_idx
        train_idx = np.asarray(split['train_idx'], dtype=np.int64)
        val_idx   = np.asarray(split['val_idx'],   dtype=np.int64)
        test_idx  = np.asarray(split['test_idx'],  dtype=np.int64)

        num_nodes = self.adj.shape[0]
        train_mask = np.zeros(num_nodes, dtype=bool); train_mask[train_idx] = True
        val_mask   = np.zeros(num_nodes, dtype=bool); val_mask[val_idx]     = True
        test_mask  = np.zeros(num_nodes, dtype=bool); test_mask[test_idx]   = True

        return labels, train_mask, val_mask, test_mask

    def load_data(self):
        if self.use_pt_graph:
            # --- PT 分支：books-nc 风格 ---
            # 是否对称化：若 args 没有 symmetrize，则默认对称化
            symmetrize = bool(getattr(args, 'symmetrize', 1))
            self.adj = self.load_pt_edge_to_coo(self.edge_file, symmetrize=symmetrize)

            self.labels, self.trn_mask, self.val_mask, self.tst_mask = self._load_pt_labels_and_masks()
            args.class_num = int(self.labels.max()) + 1

        else:
            # --- PKL 分支：原始风格 ---
            self.adj = self.load_one_file(self.adj_file)
            self.labels = self.load_feats(self.label_file)
            if np.min(self.labels) != 0:
                log(f'Class label starts from {np.min(self.labels)}')
                self.labels -= np.min(self.labels)
            args.class_num = np.max(self.labels) + 1
            masks = self.load_feats(self.mask_file)
            self.trn_mask, self.val_mask, self.tst_mask = masks['train'], masks['valid'], masks['test']

        self.node_num = self.adj.shape[0]
        print('Dataset: {data_name}, Node num: {node_num}, Edge num: {edge_num}'.format(
            data_name=self.data_name, node_num=self.node_num, edge_num=self.adj.nnz))

        # 数据采样
        if args.sample_ratio < 1.0 or self.node_num > args.max_nodes:
            self.adj, self.labels, self.trn_mask, self.val_mask, self.tst_mask = self.sample_dataset()
            self.node_num = self.adj.shape[0]
            print('After sampling: Node num: {node_num}, Edge num: {edge_num}'.format(
                node_num=self.node_num, edge_num=self.adj.nnz))

        # 供 TrnData 使用的训练边集合；若有更细粒度的训练图，你可以在这里替换为 self.trn_adj
        self.trn_mat = self.adj

        self.torch_adj, self.asym_adj = self.make_torch_adj(self.adj)
        if getattr(args, 'cache_proj', 0):
            self.asym_adj = self.asym_adj.to(args.devices[0])
        if getattr(args, 'cache_adj', 0):
            self.torch_adj = self.torch_adj.to(args.devices[0])

        self.initial_projector = InitialProjector(self.asym_adj)

        if self.tst_flag:
            tst_data = NodeData(self.labels, self.tst_mask)
            self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)

            val_data = NodeData(self.labels, self.val_mask)
            self.val_loader = data.DataLoader(val_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)

        trn_data = NodeData(self.labels, self.trn_mask)
        self.trn_loader = data.DataLoader(trn_data, batch_size=args.batch, shuffle=True, num_workers=0)
    
    def sample_dataset(self):
        """对数据集进行采样"""
        print(f"开始数据采样: 方法={args.sample_method}, 比例={args.sample_ratio}")
        
        # 确定采样节点数量
        if self.node_num > args.max_nodes:
            target_nodes = args.max_nodes
        else:
            target_nodes = int(self.node_num * args.sample_ratio)
        
        print(f"目标节点数: {target_nodes} (原始: {self.node_num})")
        
        # 根据采样方法选择节点
        if args.sample_method == 'random':
            selected_nodes = self._random_sampling(target_nodes)
        elif args.sample_method == 'degree':
            selected_nodes = self._degree_based_sampling(target_nodes)
        elif args.sample_method == 'pagerank':
            selected_nodes = self._pagerank_sampling(target_nodes)
        elif args.sample_method == 'k_hop':
            selected_nodes = self._k_hop_sampling(target_nodes)
        else:
            print(f"未知的采样方法: {args.sample_method}, 使用随机采样")
            selected_nodes = self._random_sampling(target_nodes)
        
        # 创建节点映射
        node_mapping = {old_id: new_id for new_id, old_id in enumerate(selected_nodes)}
        
        # 采样邻接矩阵
        sampled_adj = self._sample_adjacency_matrix(selected_nodes, node_mapping)
        
        # 采样标签和掩码
        sampled_labels = self.labels[selected_nodes]
        sampled_trn_mask = self.trn_mask[selected_nodes]
        sampled_val_mask = self.val_mask[selected_nodes]
        sampled_tst_mask = self.tst_mask[selected_nodes]
        
        print(f"采样完成: 节点数 {self.node_num} -> {len(selected_nodes)}")
        return sampled_adj, sampled_labels, sampled_trn_mask, sampled_val_mask, sampled_tst_mask
    
    def _random_sampling(self, target_nodes):
        """随机采样"""
        np.random.seed(args.sample_seed)
        return np.random.choice(self.node_num, target_nodes, replace=False)
    
    def _degree_based_sampling(self, target_nodes):
        """基于度的采样"""
        # 计算节点度
        degrees = np.array(self.adj.sum(axis=1)).flatten()
        
        # 按度排序，选择度最高的节点
        sorted_nodes = np.argsort(degrees)[::-1]
        return sorted_nodes[:target_nodes]
    
    def _pagerank_sampling(self, target_nodes):
        """基于PageRank的采样"""
        try:
            from scipy.sparse.linalg import eigsh
            
            # 计算转移矩阵
            adj_normalized = self.adj.copy()
            row_sums = np.array(adj_normalized.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # 避免除零
            adj_normalized = adj_normalized.multiply(1.0 / row_sums[:, np.newaxis])
            
            # 计算PageRank (最大特征值对应的特征向量)
            eigenvals, eigenvecs = eigsh(adj_normalized, k=1, which='LM')
            pagerank_scores = np.abs(eigenvecs[:, 0])
            
            # 按PageRank分数排序
            sorted_nodes = np.argsort(pagerank_scores)[::-1]
            return sorted_nodes[:target_nodes]
            
        except Exception as e:
            print(f"PageRank采样失败: {e}, 回退到随机采样")
            return self._random_sampling(target_nodes)
    
    def _k_hop_sampling(self, target_nodes):
        """K跳邻居采样"""
        # 从训练节点开始，进行BFS采样
        trn_nodes = np.where(self.trn_mask)[0]
        if len(trn_nodes) == 0:
            return self._random_sampling(target_nodes)
        
        # 从训练节点开始BFS
        visited = set()
        queue = list(trn_nodes[:min(100, len(trn_nodes))])  # 限制起始节点数量
        
        while queue and len(visited) < target_nodes:
            node = queue.pop(0)
            if node in visited:
                continue
            
            visited.add(node)
            
            # 添加邻居节点
            neighbors = self.adj[node].nonzero()[1]
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < target_nodes:
                    queue.append(neighbor)
        
        # 如果采样节点不足，补充随机节点
        if len(visited) < target_nodes:
            remaining = target_nodes - len(visited)
            remaining_nodes = list(set(range(self.node_num)) - visited)
            if remaining_nodes:
                additional = np.random.choice(remaining_nodes, min(remaining, len(remaining_nodes)), replace=False)
                visited.update(additional)
        
        return np.array(list(visited))
    
    def _sample_adjacency_matrix(self, selected_nodes, node_mapping):
        """采样邻接矩阵"""
        # 创建新的邻接矩阵
        n_selected = len(selected_nodes)
        sampled_adj = sp.lil_matrix((n_selected, n_selected), dtype=np.float32)
        
        # 将COO矩阵转换为CSR格式以便索引
        adj_csr = self.adj.tocsr()
        
        # 只保留选中节点之间的边
        for i, node_i in enumerate(selected_nodes):
            for j, node_j in enumerate(selected_nodes):
                if adj_csr[node_i, node_j] != 0:
                    sampled_adj[i, j] = adj_csr[node_i, node_j]
        
        return sampled_adj.tocoo()


class NodeData(data.Dataset):
    def __init__(self, labels, mask):
        self.iter_nodes = np.reshape(np.argwhere(np.array(mask) == True), -1)
        self.labels = labels[self.iter_nodes]
    
    def __len__(self):
        return len(self.iter_nodes)
    
    def __getitem__(self, idx):
        return self.iter_nodes[idx], self.labels[idx]  # + args.node_num - args.class_num


class TrnData(data.Dataset):
    def __init__(self, trn_handlers):
        self.dataset_num = len(trn_handlers)
        self.trn_handlers = trn_handlers
        self.ancs_list = [None] * self.dataset_num
        self.poss_list = [None] * self.dataset_num
        self.negs_list = [None] * self.dataset_num
        self.edge_nums = [None] * self.dataset_num
        self.sample_nums = [None] * self.dataset_num
        for i, handler in enumerate(self.trn_handlers):
            trn_mat = handler.trn_mat
            ancs = np.array(trn_mat.row)
            poss = np.array(trn_mat.col)
            self.ancs_list[i] = ancs
            self.poss_list[i] = poss
            self.edge_nums[i] = len(ancs)
            self.sample_nums[i] = self.edge_nums[i] // args.batch + (1 if self.edge_nums[i] % args.batch != 0 else 0)
        self.total_sample_num = sum(self.sample_nums)
        self.samples = [None] * self.total_sample_num
    
    def data_shuffling(self):
        sample_idx = 0
        for i in range(self.dataset_num):
            edge_num = self.edge_nums[i]
            perms = np.random.permutation(edge_num)
            handler = self.trn_handlers[i]
            asym_flag = handler.trn_mat.shape[0] != handler.trn_mat.shape[1]
            cand_num = handler.item_num if asym_flag else handler.node_num
            self.negs_list[i] = self.neg_sampling(self.ancs_list[i], handler.trn_mat.todok(), cand_num)
            # self.negs_list[i] = np.random.randint(cand_num, size=edge_num)
            for j in range(self.sample_nums[i]):
                st_idx = j * args.batch
                ed_idx = min((j + 1) * args.batch, edge_num)
                pick_idxs = perms[st_idx: ed_idx]
                ancs = self.ancs_list[i][pick_idxs]
                poss = self.poss_list[i][pick_idxs]
                negs = self.negs_list[i][pick_idxs]
                if asym_flag:
                    poss += handler.user_num
                    negs += handler.user_num
                self.samples[sample_idx] = (ancs, poss, negs, i)
                sample_idx += 1
        assert sample_idx == self.total_sample_num
    
    def neg_sampling(self, ancs, dokmat, cand_num):
        negs = np.zeros_like(ancs)
        for i in range(len(ancs)):
            u = ancs[i]
            while True:
                i_neg = np.random.randint(cand_num)
                if (u, i_neg) not in dokmat:
                    break
            negs[i] = i_neg
        return negs
    
    def __len__(self):
        return self.total_sample_num
    
    def __getitem__(self, idx):
        ancs, poss, negs, adj_id = self.samples[idx]
        return ancs, poss, negs, adj_id