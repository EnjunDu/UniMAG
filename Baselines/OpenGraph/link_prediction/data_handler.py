import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch_geometric.transforms as T
from model import InitialProjector
import os

class MultiDataHandler:
    def __init__(self, trn_datasets, tst_datasets):
        print(f"开始创建 MultiDataHandler，训练数据集: {trn_datasets}, 测试数据集: {tst_datasets}")
        all_datasets = list(set(trn_datasets + tst_datasets))
        self.trn_handlers = []
        self.tst_handlers = []
        
        for data_name in all_datasets:
            print(f"正在处理数据集: {data_name}")
            trn_flag = data_name in trn_datasets
            tst_flag = data_name in tst_datasets
            print(f"数据集 {data_name}: 训练={trn_flag}, 测试={tst_flag}")
            
            handler = DataHandler(data_name, trn_flag, tst_flag)
            if trn_flag:
                self.trn_handlers.append(handler)
            if tst_flag:
                self.tst_handlers.append(handler)
        
        print("开始创建联合训练数据加载器...")
        self.make_joint_trn_loader()
        print("MultiDataHandler 创建完成！")

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
        print(f"开始初始化 DataHandler: {data_name}")
        self.data_name = data_name
        self.trn_flag = trn_flag
        self.tst_flag = tst_flag
        print(f"获取数据文件路径...")
        self.get_data_files()
        # 若未设置负样本缓存目录，默认到数据集目录
        if not hasattr(self, 'neg_cache_dir'):
            self.neg_cache_dir = self.predir
        print(f"开始加载数据...")
        self.load_data()
        print(f"DataHandler {data_name} 初始化完成！")
    
    def get_data_files(self):
        predir = os.path.join(args.data_dir, self.data_name)
        self.predir = predir
        
        # 检查是否为新格式的数据集（包含lp-edge-split.pt）
        new_format_file = os.path.join(predir, 'lp-edge-split.pt')
        if os.path.exists(new_format_file):
            self.data_format = 'new'
            self.edge_split_file = new_format_file
            self.node_mapping_file = os.path.join(predir, 'node_mapping.pt')
            self.embeddings_file = os.path.join(predir, f'{self.data_name}-images_clip_embeddings.pt')
            # 也尝试通用的嵌入文件名
            if not os.path.exists(self.embeddings_file):
                self.embeddings_file = os.path.join(predir, 'clip_embeddings.pt')
            # 兜底：自动扫描任意以 clip_embeddings.pt 结尾的文件
            if not os.path.exists(self.embeddings_file):
                try:
                    for fname in os.listdir(predir):
                        if fname.endswith('clip_embeddings.pt'):
                            self.embeddings_file = os.path.join(predir, fname)
                            break
                except Exception:
                    pass
        else:
            # 使用原来的pkl格式
            self.data_format = 'old'
            self.trnfile = os.path.join(predir, 'trn_mat.pkl')
            self.tstfile = os.path.join(predir, 'tst_mat.pkl')
            self.valfile = os.path.join(predir, 'val_mat.pkl')
            if not os.path.exists(self.valfile):
                self.valfile = self.tstfile

    def load_one_file(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs)).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def load_node_mapping_safe(self):
        """安全加载node_mapping文件"""
        try:
            # 首先尝试weights_only=True
            return t.load(self.node_mapping_file, map_location='cpu', weights_only=True)
        except Exception as e1:
            try:
                # 如果失败，尝试weights_only=False（不推荐，但可能必要）
                log(f'Warning: Loading node_mapping with weights_only=False due to: {e1}')
                return t.load(self.node_mapping_file, map_location='cpu', weights_only=False)
            except Exception as e2:
                log(f'Warning: Could not load node_mapping.pt: {e2}')
                log(f'Continuing without node mapping...')
                return None

    def load_new_format_data(self):
        """加载新格式的数据文件（lp-edge-split.pt等）"""
        # 加载边分割数据
        edge_split = t.load(self.edge_split_file, map_location='cpu', weights_only=True)
        
        # 从训练边构建稀疏矩阵
        train_edges = edge_split['train']
        source_nodes = train_edges['source_node'].numpy()
        target_nodes = train_edges['target_node'].numpy()
        
        # 获取节点数量
        max_node = max(np.max(source_nodes), np.max(target_nodes)) + 1
        self.node_num = max_node
        
        # 创建训练邻接矩阵
        edge_values = np.ones(len(source_nodes), dtype=np.float32)
        trn_mat = coo_matrix((edge_values, (source_nodes, target_nodes)), shape=(max_node, max_node))
        
        # 存储验证和测试数据
        self.val_edges = edge_split['valid']
        self.tst_edges = edge_split['test']
        
        # 加载节点映射（如果存在）
        if os.path.exists(self.node_mapping_file):
            self.node_mapping = self.load_node_mapping_safe()
        else:
            self.node_mapping = None
        
        # 加载CLIP嵌入（如果存在）
        if os.path.exists(self.embeddings_file):
            try:
                self.clip_embeddings = t.load(self.embeddings_file, map_location='cpu', weights_only=True)
                log(f'Loaded CLIP embeddings with shape: {self.clip_embeddings.shape}')
            except Exception as e:
                log(f'Warning: Could not load CLIP embeddings: {e}')
                self.clip_embeddings = None
        else:
            self.clip_embeddings = None
            
        return trn_mat

    def convert_edges_to_coo_matrix(self, edges, trn_mat_shape):
        """将边数据转换为COO矩阵格式"""
        source_nodes = edges['source_node'].numpy()
        target_nodes = edges['target_node'].numpy()
        edge_values = np.ones(len(source_nodes), dtype=np.float32)
        return coo_matrix((edge_values, (source_nodes, target_nodes)), shape=trn_mat_shape)

    def normalize_adj(self, mat):
        degree = np.array(mat.sum(axis=-1))
        # 添加安全检查，避免除零错误
        degree = np.maximum(degree, 1.0)  # 确保度数至少为1
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        if mat.shape[0] == mat.shape[1]:
            return mat.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()
        else:
            tem = d_inv_sqrt_mat.dot(mat)
            col_degree = np.array(mat.sum(axis=0))
            # 添加安全检查，避免除零错误
            col_degree = np.maximum(col_degree, 1.0)  # 确保度数至少为1
            d_inv_sqrt = np.reshape(np.power(col_degree, -0.5), [-1])
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
            d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
            return tem.dot(d_inv_sqrt_mat).tocoo()

    def unique_numpy(self, row, col):
        hash_vals = row * self.node_num + col
        hash_vals = np.unique(hash_vals).astype(np.int64)
        col = hash_vals % self.node_num
        row = (hash_vals - col).astype(np.int64) // self.node_num
        return row, col
    
    def make_torch_adj(self, mat):
        if mat.shape[0] == mat.shape[1]:
            # to symmetric
            if self.data_name in ['ddi']:
                _row = mat.row
                _col = mat.col
                row = np.concatenate([_row, _col]).astype(np.int64)
                col = np.concatenate([_col, _row]).astype(np.int64)
                # row, col = self.unique_numpy(row, col)
                data = mat.data
                data = np.concatenate([data, data]).astype(np.float32)
            else:
                row, col = mat.row, mat.col
                data = mat.data
            # data = np.ones_like(row)
            mat = coo_matrix((data, (row, col)), mat.shape)
            if args.selfloop == 1:
                mat = (mat + sp.eye(mat.shape[0])) * 1.0
        normed_asym_mat = self.normalize_adj(mat)
        row = t.from_numpy(normed_asym_mat.row).long()
        col = t.from_numpy(normed_asym_mat.col).long()
        idxs = t.stack([row, col], dim=0)
        vals = t.from_numpy(normed_asym_mat.data).float()
        shape = t.Size(normed_asym_mat.shape)
        asym_adj = t.sparse_coo_tensor(idxs, vals, shape)
        if mat.shape[0] == mat.shape[1]:
            return asym_adj, asym_adj
        else:
            # make ui adj
            a = sp.csr_matrix((self.user_num, self.user_num))
            b = sp.csr_matrix((self.item_num, self.item_num))
            mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
            mat = (mat != 0) * 1.0
            if args.selfloop == 1:
                mat = (mat + sp.eye(mat.shape[0])) * 1.0
            mat = self.normalize_adj(mat)

            # make cuda tensor
            idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
            vals = t.from_numpy(mat.data.astype(np.float32))
            shape = t.Size(mat.shape)
            return t.sparse_coo_tensor(idxs, vals, shape), asym_adj

    def load_data(self):
        if self.data_format == 'new':
            # 使用新格式加载数据
            trn_mat = self.load_new_format_data()
        else:
            # 使用原格式加载数据
            trn_mat = self.load_one_file(self.trnfile)
            
        self.trn_mat = trn_mat
        if trn_mat.shape[0] != trn_mat.shape[1]:
            self.user_num, self.item_num = trn_mat.shape
            self.node_num = self.user_num + self.item_num
            print('Dataset: {data_name}, User num: {user_num}, Item num: {item_num}, Node num: {node_num}, Edge num: {edge_num}'.format(data_name=self.data_name, user_num=self.user_num, item_num=self.item_num, node_num=self.node_num, edge_num=trn_mat.nnz))
        else:
            self.node_num = trn_mat.shape[0]
            print('Dataset: {data_name}, Node num: {node_num}, Edge num: {edge_num}'.format(data_name=self.data_name, node_num=self.node_num, edge_num=trn_mat.nnz))
        
        self.torch_adj, self.asym_adj = self.make_torch_adj(trn_mat)
        if args.cache_proj:
            self.asym_adj = self.asym_adj.to(args.devices[0])
        if args.cache_adj:
            self.torch_adj = self.torch_adj.to(args.devices[0])

        # 处理初始投影 - 如果有CLIP嵌入，使用它们
        if hasattr(self, 'clip_embeddings') and self.clip_embeddings is not None:
            # 使用CLIP嵌入作为初始投影
            self.initial_projector = InitialProjector(self.clip_embeddings, input_is_embeds=True)
        else:
            self.initial_projector = InitialProjector(self.asym_adj)
        
        if self.tst_flag:
            if self.data_format == 'new':
                # 为新格式创建验证和测试数据加载器
                print(f"开始创建新格式的测试数据加载器...")
                val_data = NewFormatTstData(self.val_edges, trn_mat, num_neg=args.num_neg_eval, dataset_dir=self.neg_cache_dir)
                print(f"验证集创建完成，用户数: {len(val_data)}")
                self.val_loader = data.DataLoader(val_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)
                print(f"验证集 DataLoader 创建完成")
                
                tst_data = NewFormatTstData(self.tst_edges, trn_mat, num_neg=args.num_neg_eval, dataset_dir=self.neg_cache_dir)
                print(f"测试集创建完成，用户数: {len(tst_data)}")
                self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)
                print(f"测试集 DataLoader 创建完成")
                
                log(f'数据加载完成: 验证集 {len(val_data)} 个用户, 测试集 {len(tst_data)} 个用户')
            else:
                # 原格式的测试数据加载
                print(f"开始创建原格式的测试数据加载器...")
                val_mat = self.load_one_file(self.valfile)
                val_data = TstData(val_mat, trn_mat)
                self.val_loader = data.DataLoader(val_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)
                tst_mat = self.load_one_file(self.tstfile)
                tst_data = TstData(tst_mat, trn_mat)
                self.tst_loader = data.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)
                
                log(f'数据加载完成: 验证集 {len(val_data)} 个用户, 测试集 {len(t_data)} 个用户')
        
        print(f"DataHandler {self.data_name} 的 load_data 方法完成！")

class NewFormatTstData(data.Dataset):
    """用于新格式数据的测试数据类"""
    def __init__(self, edge_data, trn_mat, num_neg=1000, dataset_dir=None):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0
        self.num_neg = num_neg
        self.dataset_dir = dataset_dir
        
        # 处理正样本和负样本（按边）
        source_nodes = edge_data['source_node'].numpy()
        target_nodes = edge_data['target_node'].numpy()
        
        # 构建 user -> 第一条边 的索引（用于为每个用户选择一个正样本）
        first_edge_idx = dict()
        for i in range(len(source_nodes)):
            u = int(source_nodes[i])
            if u not in first_edge_idx:
                first_edge_idx[u] = i
        
        # 所有出现过的测试用户（稳定排序）
        user_list = sorted(list(first_edge_idx.keys()))
        self.user_list = np.array(user_list, dtype=np.int64)
        self.user_to_idx = {u: idx for idx, u in enumerate(self.user_list)}
        
        # 为每个用户挑选一个正样本（使用第一条边）
        self.user_pos = np.zeros(len(self.user_list), dtype=np.int64)
        for u in self.user_list:
            eidx = first_edge_idx[u]
            self.user_pos[self.user_to_idx[u]] = int(target_nodes[eidx])
        
        # 加载或生成按边的负样本，然后为每个用户挑选与其选中正样本对应的负样本
        neg_samples_edges = self.load_or_generate_negative_samples(source_nodes, target_nodes, trn_mat)
        self.user_neg = np.zeros((len(self.user_list), self.num_neg), dtype=np.int64)
        for u in self.user_list:
            eidx = first_edge_idx[u]
            self.user_neg[self.user_to_idx[u]] = neg_samples_edges[eidx]
        
        # tstLocs 仍保留（可能用于其他指标）
        num_nodes = trn_mat.shape[0]
        tstLocs = [None] * num_nodes
        for i in range(len(source_nodes)):
            row = int(source_nodes[i])
            col = int(target_nodes[i])
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
        self.tstLocs = tstLocs
        
    def load_or_generate_negative_samples(self, source_nodes, target_nodes, trn_mat):
        """加载预生成的负样本，如果不存在则生成新的（按边顺序）。"""
        neg_samples_path = None
        if self.dataset_dir is not None:
            neg_samples_path = os.path.join(self.dataset_dir, f'neg_samples_{self.num_neg}.pkl')
        
        if neg_samples_path is not None and os.path.exists(neg_samples_path):
            print(f"找到预生成的负样本文件：{neg_samples_path}")
            try:
                import pickle
                with open(neg_samples_path, 'rb') as f:
                    neg_samples_data = pickle.load(f)
                # 直接返回按边顺序的负样本数组
                # 优先选择与当前边数量匹配的集合（valid 或 test 二者之一应匹配）
                if len(source_nodes) == len(neg_samples_data.get('val_neg_samples', [])):
                    print("使用验证集预生成负样本")
                    return neg_samples_data['val_neg_samples']
                if len(source_nodes) == len(neg_samples_data.get('tst_neg_samples', [])):
                    print("使用测试集预生成负样本")
                    return neg_samples_data['tst_neg_samples']
                print("预生成负样本数量不匹配，重新生成")
                return self.generate_negative_samples(source_nodes, target_nodes, trn_mat)
            except Exception as e:
                print(f"加载预生成负样本失败：{e}，重新生成")
                return self.generate_negative_samples(source_nodes, target_nodes, trn_mat)
        else:
            print(f"未找到预生成的负样本文件，开始生成...")
            return self.generate_negative_samples(source_nodes, target_nodes, trn_mat)
    
    def generate_negative_samples(self, source_nodes, target_nodes, trn_mat):
        """为每条边生成负样本 - 优化版本（按边）。"""
        print(f"开始生成负样本，总共 {len(source_nodes)} 个正样本...")
        neg_samples = []
        num_nodes = trn_mat.shape[0]
        print("预计算可用负样本...")
        available_neg_samples = set(range(num_nodes))
        for i in range(len(source_nodes)):
            if i % 1000 == 0:
                print(f"处理进度: {i}/{len(source_nodes)}")
            pos_target = int(target_nodes[i])
            candidates = list(available_neg_samples - {pos_target})
            if len(candidates) >= self.num_neg:
                neg_targets = np.random.choice(candidates, self.num_neg, replace=False)
            else:
                neg_targets = np.random.choice(candidates, self.num_neg, replace=True)
            neg_samples.append(neg_targets)
        print("负样本生成完成！")
        return np.array(neg_samples)
    
    def __len__(self):
        return len(self.user_list)
    
    def __getitem__(self, idx):
        return self.user_list[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trn_mat):
        self.csrmat = (trn_mat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tst_nodes = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tst_nodes.add(row)
        tst_nodes = np.array(list(tst_nodes))
        self.tst_nodes = tst_nodes
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tst_nodes)

    def __getitem__(self, idx):
        return self.tst_nodes[idx]

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
