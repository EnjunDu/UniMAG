import os
import pandas as pd
import torch
from torch_geometric.data import Data,Dataset
from dgl.data.utils import load_graphs
from PIL import Image
from data_utils.UniMAG import UniMAG


class Grocery(UniMAG):
    def __init__(self,datadir,pt_file,csv_file,img_dir):
        self.datadir=datadir
        self.data=self.get_raw_graph(pt_file,csv_file,img_dir)

    def get_raw_graph(self,pt_file,csv_file,img_dir):
        graph=load_graphs(os.path.join(self.datadir,pt_file))[0][0]
        x=torch.zeros(graph.nodes().shape[0],768,dtype=torch.int32)
        edge_index=torch.stack([graph.edges()[0],graph.edges()[1]])
        y=graph.ndata['label']
        data=Data(x=x,edge_index=edge_index,y=y)    
        return data
    
    def get_raw_text(self,pt_file,csv_file,img_dir):
        df=pd.read_csv(os.path.join(self.datadir,csv_file))
        raw_text=df['text'].to_list()
        return raw_text
    
    def get_raw_image(self,pt_file,csv_file,img_dir):
        raw_image=[]
        graph=load_graphs(os.path.join(self.datadir,pt_file))[0][0]
        for i in range(graph.nodes().shape[0]):
            img_path=os.path.join(self.datadir,img_dir,str(i)+'.jpg')
            raw_image.append(Image.open(img_path))
        return raw_image


if __name__ == "__main__":
    grocery=Grocery('/home/ai/MMAG/Grocery','GroceryGraph.pt','Grocery.csv','GroceryImages_extracted/GrocerySImages')
    print(grocery)