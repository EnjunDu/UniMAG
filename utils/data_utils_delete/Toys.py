import os
import pandas as pd
import torch
from torch_geometric.data import Data,Dataset
from dgl.data.utils import load_graphs
from PIL import Image
from data_utils.UniMAG import UniMAG


class Toys(UniMAG):
    def __init__(self,datadir,pt_file,csv_file,img_dir):
        self.datadir=datadir
        self.data=self.get_raw_graph(pt_file,csv_file,img_dir)
        
    def get_raw_graph(self,pt_file,csv_file,img_dir):
        graph=load_graphs(os.path.join(self.datadir,pt_file))[0][0]
        x=torch.zeros(graph.nodes().shape[0],768,dtype=torch.float)
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
    toy=Toys('/home/ai/MMAG/Toys','ToysGraph.pt','Toys.csv','ToysImages_extracted/ToysImages')
    print(toy)

# class Toy(Dataset):
#     def __init__(self,datadir,pt_file,csv_file,img_dir):
#         self.datadir=datadir
#         self.data,self.raw_text,self.raw_image=self.get_raw_data(pt_file,csv_file,img_dir)
#     def get_raw_data(self,pt_file,csv_file,img_dir):
#         graph=load_graphs(os.path.join(self.datadir,pt_file))[0][0]
#         x=torch.zeros(graph.nodes().shape[0],768,dtype=torch.int32)
#         edge_index=torch.stack([graph.edges()[0],graph.edges()[1]])
#         y=graph.ndata['label']
#         data=Data(x=x,edge_index=edge_index,y=y)
#         df=pd.read_csv(os.path.join(self.datadir,csv_file))
#         raw_text=df['text'].to_list()
#         raw_image=[]
#         for i in range(graph.nodes().shape[0]):
#             img_path=os.path.join(self.datadir,img_dir,str(i)+'.jpg')
#             raw_image.append(Image.open(img_path))
#         return data,raw_text,raw_image

# print(len(load_graphs(os.path.join(self.datadir,pt_file))[0]))

# movie=Movies('/home/ai/MMAG/Movies','MoviesGraph.pt','Movies.csv','MoviesImages_extracted/MoviesImages')
# grocery=Grocery('/home/ai/MMAG/Grocery','GroceryGraph.pt','Grocery.csv','GroceryImages_extracted/GrocerySImages')
# toy=Toy('/home/ai/MMAG/Toys','ToysGraph.pt','Toys.csv','ToysImages_extracted/ToysImages')
# cloth=Cloth('cloth','node_mapping.pt')

# graph=load_graphs(os.path.join(self.datadir,pt_file))[0][0]
# x=torch.zeros(graph.nodes().shape[0],768,dtype=torch.int32)
# edge_index=torch.stack([graph.edges()[0],graph.edges()[1]])
# y=graph.ndata['label']
model=torch.load('/home/ai/MMAG/books-lp/lp-edge-split.pt')
#model=torch.load('/home/ai/MMAG/books-lp/node_mapping.pt')
print(model)