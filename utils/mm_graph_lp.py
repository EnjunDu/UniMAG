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
from utils.UniMAG import UniMultimodalGraph

Dataname2nodenum = {'books-lp':636502,'cloth':125839,'sports':50250}

class mm_graph_lp(UniMultimodalGraph):
    def __init__(self,root,dataset_name,device):
        # super().__init__()
        self.datadir=os.path.join(root,dataset_name)
        self.root = root
        self.dataset_name = dataset_name
        self.device = device
        self.data=self.get_raw_graph()

    def get_raw_graph(self):
        edge_split_path = os.path.join(self.datadir, 'lp-edge-split.pt')
        edge_split = torch.load(edge_split_path, map_location=self.device)

        self.num_nodes = Dataname2nodenum[self.dataset_name]
        x = torch.zeros(self.num_nodes,768,dtype=torch.float).to(self.device)

        edges = torch.stack([edge_split['train']['source_node'],edge_split['train']['target_node']]).to(self.device)
        data = Data(x=x,
                    edge_index=edges,
        ).to(self.device)
        return data
    
    def get_raw_text(self):
        if self.dataset_name == 'books-lp':
            raise Exception('books-lp 的node_map文件有问题,无法用get_raw_text函数')
        raw_text_path = os.path.join(self.datadir,f'{self.dataset_name}-raw-text.jsonl')
        text = {}
        # with open(raw_text_path,'r',encoding='utf-8') as f :
        #     for line in f:
        #         text[line['asin']] = line['raw_text']
        df = pd.read_json(raw_text_path,lines=True)
        for _ , row in tqdm(df.iterrows(),total=len(df),desc=f'processing raw text data from {self.dataset_name}') :
            text[row['asin']] = row['raw_text'][0] if self.dataset_name == 'books-lp' else row['raw_text']
        node_map = torch.load(os.path.join(self.datadir,'node_mapping.pt'))
        # node_2_asin = {v:k for k,v in node_map.items()}
        node_2_asin = {v:k for k,v in sorted(node_map.items(),key=lambda item:item[1])}
        res = [text[v] for k,v in node_2_asin.items()]
        return res

    def get_raw_image(self):
        if self.dataset_name == 'books-lp':
            raise Exception('books-lp 的node_map文件有问题,无法用get_raw_image函数')
        image={}
        asins = []
        raw_image_path = os.path.join(self.datadir,f'{self.dataset_name}-images_extracted',f'{self.dataset_name}-images')
        raw_text_path = os.path.join(self.datadir,f'{self.dataset_name}-raw-text.jsonl')
        df = pd.read_json(raw_text_path,lines=True)
        asins = [df.iloc[i]['asin'] for i in range(self.num_nodes)]
        for asin in tqdm(asins,total=len(asins),desc=f'processing raw image data form {self.dataset_name}'):
            img_path=os.path.join(raw_image_path,str(asin)+'.jpg')
            image[str(asin)] = Image.open(img_path).convert('RGB')
        node_map = torch.load(os.path.join(self.datadir,'node_mapping.pt'))
        node_2_asin = {v:k for k,v in sorted(node_map.items(),key=lambda item:item[1])}

        res = [image[v] for k,v in node_2_asin.items()]
        return res
    
    def get_split(self):
        edge_split_path = os.path.join(self.datadir, 'lp-edge-split.pt')
        edge_split = torch.load(edge_split_path, map_location=self.device)
        return edge_split


if __name__ == "__main__":
    dataset = mm_graph_lp('/home/ai/MMAG','sports','cuda:2')
    raw_text = dataset.get_raw_text()
    # raw_graph = dataset.get_raw_image()
    edge_split = dataset.get_split()
    print(dataset)