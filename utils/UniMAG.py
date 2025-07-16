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
from utils.embedding_manager import EmbeddingManager

class UniMultimodalGraph(Dataset):
    def __init__(self,rootdir,dataset_name,device,encoder_name):
        self.rootdir = rootdir
        self.datadir = os.path.join(rootdir,dataset_name)
        self.dataset_name = dataset_name
        self.device = device
        self.encoder_name = encoder_name  # "Qwen/Qwen2.5-VL-3B-Instruct"
        super().__init__(self.datadir)
        self.data = torch.load(self.processed_paths[0]).to(self.device)
    
    def get_multimodal_embs(self,data):
        manager = EmbeddingManager(self.rootdir)
        if self.dataset_name in ['Grocery','Toys','Movies','RedditS','RedditM']:
            node_2_asin = {i:f"{i}" for i in range(self.num_nodes)}
        else:
            node_map = torch.load(os.path.join(self.datadir,'node_mapping.pt'))
            node_2_asin = {v:k for k,v in sorted(node_map.items(),key=lambda item:item[1])}

        node_id_path = os.path.join(self.datadir,'node_ids.json')
        with open(node_id_path,'r',encoding='utf-8') as f:
            json_data = json.load(f)
            # for line in f.readlines():
            #     json_data = json.load(line)
                
            asin_2_loc = {k:step for step,k in enumerate(json_data)}
        
        text_embeddings = manager.get_embedding(
            dataset_name=f"{self.dataset_name}",
            modality="text",
            encoder_name=f"{self.encoder_name}",
            dimension=768
        )
        text_embeddings = torch.from_numpy(text_embeddings)
        node_text_embeddings = torch.stack([text_embeddings[asin_2_loc[node_2_asin[i]]] for i in range(self.num_nodes)]).to(self.device)

        image_embeddings = manager.get_embedding(
            dataset_name=f"{self.dataset_name}",
            modality="image",
            encoder_name=f"{self.encoder_name}",
            dimension=768
        )
        image_embeddings = torch.from_numpy(image_embeddings)
        node_image_embeddings = torch.stack([image_embeddings[asin_2_loc[node_2_asin[i]]] for i in range(self.num_nodes)]).to(self.device)
        new_x = torch.concat([node_text_embeddings,node_image_embeddings],dim=1)
        data.x = new_x
        return data

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ["geometric_data_processed.pt"]
    
    def download(self):
        '''
            在检测到原始文件不存在时，download方法用于下载原始文件，并对原始文件进行相应的处理
            主要包括以下几点：
            1.从huggingface上下载原始数据集（包括数据集、数据集原始文本、数据集原始图像）
            2.对图像压缩包解压缩
            3.……
            由于不同数据集的下载后处理方式，以及文件形式都不同，所以download方法暂时省略
        '''
        os.makedirs(self.raw_dir, exist_ok=True)
        print('download方法代码有待补充')

    def process(self):
        data = self.get_raw_graph()
        data_with_embs = self.get_multimodal_embs(data)

        print('Saving data……')
        torch.save(data_with_embs,self.processed_paths[0])
        