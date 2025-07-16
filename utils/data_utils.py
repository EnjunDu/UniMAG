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
from utils.MMAG_graph import MMAG_graph
from utils.mm_graph_kg import mm_graph_kg
from utils.mm_graph_lp import mm_graph_lp
from utils.mm_graph_nc import mm_graph_nc
from utils.embedding_manager import EmbeddingManager

Datasetname2class = {'Movies':MMAG_graph,'Toys':MMAG_graph,'Grocery':MMAG_graph,'RedditS':MMAG_graph,'RedditM':MMAG_graph,
                     'books-nc':mm_graph_nc,'ele-fashion':mm_graph_nc,
                     'books-lp':mm_graph_lp,'cloth':mm_graph_lp,'sports':mm_graph_lp,
                     'mm-codex-m':mm_graph_kg,'mm-codex-s':mm_graph_kg}