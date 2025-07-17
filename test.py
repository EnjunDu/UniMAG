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
from utils.data_utils import Datasetname2class
from utils.MMAG_graph import MMAG_graph

# rootdir,dataset_name,device,encoder_name
dataset = Datasetname2class['Movies']('/home/ai/MMAG',
                                      'Movies',
                                      'cuda:2',
                                      "Qwen/Qwen2.5-VL-3B-Instruct",
)