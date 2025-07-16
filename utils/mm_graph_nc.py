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

Dataname2nodenum = {'books-nc':685294,'ele-fashion':97766}

class mm_graph_nc(UniMultimodalGraph):
    def get_raw_graph(self):
        edge_path = os.path.join(self.datadir, 'nc_edges-nodeid.pt')
        edges = torch.tensor(torch.load(edge_path), dtype=torch.int64).T.to(self.device)

        self.num_nodes = Dataname2nodenum[self.dataset_name]
        x = torch.zeros(self.num_nodes,768,dtype=torch.float).to(self.device)

        labels_path = os.path.join(self.datadir, 'labels-w-missing.pt')
        labels = torch.tensor(torch.load(labels_path), dtype=torch.int64).to(self.device)
        
        data = Data(x=x,
                    edge_index=edges,
                    y=labels
        ).to(self.device)
        return data
    
    def get_raw_text(self):
        raw_text_path = os.path.join(self.datadir,f'{self.dataset_name}-raw-text.jsonl')
        text = {}
        # with open(raw_text_path,'r',encoding='utf-8') as f :
        #     for line in f:
        #         text[line['asin']] = line['raw_text']
        df = pd.read_json(raw_text_path,lines=True)
        node_map = torch.load(os.path.join(self.datadir,'node_mapping.pt'))
        for _ , row in tqdm(df.iterrows(),total=len(df),desc=f'processing raw text data from {self.dataset_name}') :
            text[str(row['asin'])] = row['raw_text'][0] if self.dataset_name == 'books-nc' else row['raw_text']
        # node_2_asin = {v:k for k,v in node_map.items()}
        node_2_asin = {v:k for k,v in sorted(node_map.items(),key=lambda item:item[1])}
        res = [text[v] for k,v in node_2_asin.items()]
        return res

    def get_raw_image(self):
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
        node_split_path = os.path.join(self.datadir, 'split.pt')
        node_split = torch.load(node_split_path)
        train_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(self.num_nodes, dtype=torch.bool).to(self.device)
        train_mask[node_split['train_idx']] = True
        val_mask[node_split['val_idx']] = True
        test_mask[node_split['test_idx']] = True
        return train_mask,val_mask,test_mask


if __name__ == "__main__":
    dataset = mm_graph_nc('/home/ai/MMAG','books-nc','cuda:1')
    raw_text = dataset.get_raw_text()
    raw_graph = dataset.get_raw_image()
    train_mask,val_mask,test_mask = dataset.get_split()
    print(dataset)