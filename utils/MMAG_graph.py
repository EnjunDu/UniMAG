import os
import pandas as pd
import torch
from torch_geometric.data import Data,Dataset
from dgl.data.utils import load_graphs
from PIL import Image
from utils.UniMAG import UniMultimodalGraph

class MMAG_graph(UniMultimodalGraph):
    def get_raw_graph(self):
        self.pt_file = f"{self.dataset_name}Graph.pt"
        self.csv_file = f"{self.dataset_name}.csv"
        graph=load_graphs(os.path.join(self.datadir,self.pt_file))[0][0]
        x=torch.zeros(graph.nodes().shape[0],768,dtype=torch.int32)
        edge_index=torch.stack([graph.edges()[0],graph.edges()[1]])
        y=graph.ndata['label']
        self.num_nodes = y.shape[0]
        data=Data(x=x,edge_index=edge_index,y=y)    
        return data
    
    def get_raw_text(self):
        df=pd.read_csv(os.path.join(self.datadir,self.csv_file))
        raw_text=df['text'].to_list()
        return raw_text
    
    def get_raw_image(self):
        raw_image=[]
        raw_image_path = os.path.join(self.datadir,f'{self.dataset_name}-images_extracted',f'{self.dataset_name}-images')
        graph=load_graphs(os.path.join(self.datadir,self.pt_file))[0][0]
        for i in range(graph.nodes().shape[0]):
            img_path=os.path.join(self.datadir,raw_image_path,str(i)+'.jpg')
            raw_image.append(Image.open(img_path).convert('RGB'))
        return raw_image

if __name__ == "__main__":
    grocery=MMAG_graph('/home/ai/MMAG','Grocery','cuda:1')
    print(grocery)