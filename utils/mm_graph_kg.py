import os
import pandas as pd
import torch
from torch_geometric.data.data import Data
from torch_geometric.data import Dataset
import torch_geometric as pyg
from dgl.data.utils import load_graphs
from PIL import Image
from tqdm import tqdm
import json
import pickle
import numpy as np
from collections import Counter
import tarfile
import torchvision.transforms as transforms
import zipfile
from utils.UniMAG import UniMultimodalGraph

Dataname2nodenum = {'mm-codex-s': 1383, 'mm-codex-m': 7697}
Dataname2relnum = {'mm-codex-s': 40, 'mm-codex-m': 49}

class mm_graph_kg(UniMultimodalGraph):
    def __init__(self, root, dataset_name, device):
        # super().__init__()
        self.datadir = os.path.join(root, dataset_name)
        self.codex_dir = os.path.join(self.datadir, 'codex-s') if dataset_name == 'mm-codex-s' else os.path.join(self.datadir, 'codex-m')
        self.root = root
        self.dataset_name = dataset_name
        self.device = device
        self.data = self.get_raw_graph()
        # 加载实体和关系的ID映射
        self.load_mappings()
    
    def load_pickle_data(self, file_path):
        """从pickle文件加载数据"""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def load_mappings(self):
        """加载实体和关系的ID映射"""
        # 加载实体ID映射
        ent_id_path = os.path.join(self.codex_dir, 'ent_id')
        self.ent_to_id = {}
        self.id_to_ent = {}
        with open(ent_id_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        ent, eid = parts
                        eid = int(eid)
                        self.ent_to_id[ent] = eid
                        self.id_to_ent[eid] = ent
        
        # 加载关系ID映射
        rel_id_path = os.path.join(self.codex_dir, 'rel_id')
        self.rel_to_id = {}
        self.id_to_rel = {}
        with open(rel_id_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        rel, rid = parts
                        rid = int(rid)
                        self.rel_to_id[rel] = rid
                        self.id_to_rel[rid] = rel

    def get_raw_graph(self):
        self.num_nodes = Dataname2nodenum[self.dataset_name]
        self.num_relations = Dataname2relnum[self.dataset_name]
        
        x = torch.zeros(self.num_nodes, 768, dtype=torch.float).to(self.device)
        
        train_file_path = os.path.join(self.codex_dir, 'train.pickle')
        train_data = self.load_pickle_data(train_file_path)
        
        head_entities = []
        relations = []
        tail_entities = []
        
        for triple in train_data:
            h, r, t = triple
            head_entities.append(h)
            relations.append(r)
            tail_entities.append(t)
        
        head_entities = torch.tensor(head_entities, dtype=torch.long).to(self.device)
        relations = torch.tensor(relations, dtype=torch.long).to(self.device)
        tail_entities = torch.tensor(tail_entities, dtype=torch.long).to(self.device)
        
        edge_index = torch.stack([head_entities, tail_entities])
        
        data = Data(
            x=x,
            edge_index=edge_index,
            # edge_type=relations,
            edge_attr=relations,
        ).to(self.device)
        
        # 验证集、测试集
        valid_file_path = os.path.join(self.codex_dir, 'valid.pickle')
        test_file_path = os.path.join(self.codex_dir, 'test.pickle')
        valid_neg_file_path = os.path.join(self.codex_dir, 'valid_negatives.pickle')
        test_neg_file_path = os.path.join(self.codex_dir, 'test_negatives.pickle')
        
        valid_data = self.load_pickle_data(valid_file_path)
        test_data = self.load_pickle_data(test_file_path)
        valid_neg_data = self.load_pickle_data(valid_neg_file_path)
        test_neg_data = self.load_pickle_data(test_neg_file_path)
        
        # 处理验证集
        valid_heads = []
        valid_relations = []
        valid_tails = []
        
        for triple in valid_data:
            h, r, t = triple
            valid_heads.append(h)
            valid_relations.append(r)
            valid_tails.append(t)
        
        # 处理测试集
        test_heads = []
        test_relations = []
        test_tails = []
        
        for triple in test_data:
            h, r, t = triple
            test_heads.append(h)
            test_relations.append(r)
            test_tails.append(t)
        
        self.valid_triples = {
            'head': torch.tensor(valid_heads, dtype=torch.long).to(self.device),
            'relation': torch.tensor(valid_relations, dtype=torch.long).to(self.device),
            'tail': torch.tensor(valid_tails, dtype=torch.long).to(self.device),
        }
        
        self.test_triples = {
            'head': torch.tensor(test_heads, dtype=torch.long).to(self.device),
            'relation': torch.tensor(test_relations, dtype=torch.long).to(self.device),
            'tail': torch.tensor(test_tails, dtype=torch.long).to(self.device),
        }
        
        self.valid_negatives = valid_neg_data
        self.test_negatives = test_neg_data
        # neg valid
        valid_neg_heads = []
        valid_neg_relations = []
        valid_neg_tails = []
        
        for triple in valid_neg_data:
            h, r, t = triple
            valid_neg_heads.append(h)
            valid_neg_relations.append(r)
            valid_neg_tails.append(t)
        
        self.valid_neg_triples = {
            'head': torch.tensor(valid_neg_heads, dtype=torch.long).to(self.device),
            'relation': torch.tensor(valid_neg_relations, dtype=torch.long).to(self.device),
            'tail': torch.tensor(valid_neg_tails, dtype=torch.long).to(self.device),
        }

        # neg test
        test_neg_heads = []
        test_neg_relations = []
        test_neg_tails = []
        
        for triple in test_neg_data:
            h, r, t = triple
            test_neg_heads.append(h)
            test_neg_relations.append(r)
            test_neg_tails.append(t)

        self.test_neg_triples = {
            'head': torch.tensor(test_neg_heads, dtype=torch.long).to(self.device),
            'relation': torch.tensor(test_neg_relations, dtype=torch.long).to(self.device),
            'tail': torch.tensor(test_neg_tails, dtype=torch.long).to(self.device),
        }

        return data
    
    def get_split(self):
        return self.valid_triples ,self.test_triples, self.valid_neg_triples, self.test_neg_triples

    def get_triple_str(self, h, r, t):
        """将三元组ID转换为字符串形式"""
        if hasattr(self, 'id_to_ent') and hasattr(self, 'id_to_rel'):
            h_str = self.id_to_ent.get(h.item() if torch.is_tensor(h) else h, f"实体_{h}")
            r_str = self.id_to_rel.get(r.item() if torch.is_tensor(r) else r, f"关系_{r}")
            t_str = self.id_to_ent.get(t.item() if torch.is_tensor(t) else t, f"实体_{t}")
            return f"{h_str} --[{r_str}]--> {t_str}"
        else:
            return f"({h}, {r}, {t})"
            
    def get_raw_image(self):
        image_tar_path = os.path.join(self.root, 'codex-images.tar')
        image_dir = os.path.join(self.root, 'codex-images')
        
        if not os.path.exists(image_dir) and os.path.exists(image_tar_path):
            print(f"unzip images to {image_dir}...")
            with tarfile.open(image_tar_path, 'r') as tar:
                tar.extractall(path=self.root)
            print("finish")
        
        entity_images = {}
        if os.path.exists(image_dir):
            for eid, entity in tqdm(self.id_to_ent.items(), desc="加载实体图像"):
                image_path = os.path.join(image_dir, f"{entity}.jpg")
                img = Image.open(image_path).convert('RGB')
                entity_images[eid] = img
              
        return entity_images
    
    def get_raw_text(self):
        entity_zip_path = os.path.join(self.root, 'entity_test.zip')
        entity_dir = os.path.join(self.root, 'entity_test')
        extracts_dir = os.path.join(entity_dir, 'extracts')
        
        if not os.path.exists(extracts_dir) and os.path.exists(entity_zip_path):
            print(f"unzip file to {entity_dir}...")
            with zipfile.ZipFile(entity_zip_path, 'r') as zip_ref:
                zip_ref.extractall(entity_dir)
            print("finish")
        
        relations_path = os.path.join(self.root, 'relations.json')
        
        entity_texts = {}
        relation_texts = {}
        
        # 加载实体文本
        for eid, entity in tqdm(self.id_to_ent.items(), desc="加载实体文本"):
            text_path = os.path.join(extracts_dir, f"{entity}.txt")
            with open(text_path, 'r', encoding='utf-8') as f:
                entity_texts[eid] = f.read().strip()
        
        # 加载关系文本
        with open(relations_path, 'r', encoding='utf-8') as f:
            relations_data = json.load(f)
            for rid, relation in self.id_to_rel.items():
                if relation in relations_data:
                    rel_info = relations_data[relation]
                    relation_texts[rid] = {
                        'label': rel_info.get('label', ''),
                        'description': rel_info.get('description', '')
                    }
        
        return {'entity_texts': entity_texts, 'relation_texts': relation_texts}

if __name__ == "__main__":
    dataset = mm_graph_kg('/home/ai/MMAG/mm-code', 'mm-codex-m', 'cuda:0')
    # images = dataset.get_raw_image()
    # print(f"共加载了 {len(images)} 个实体图像")
    
    # texts = dataset.get_raw_text()
    image = dataset.get_raw_image()
    