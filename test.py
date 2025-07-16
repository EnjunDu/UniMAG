import os
from utils.data_utils import Datasetname2class

dataset = Datasetname2class['books-nc']('/home/ai/MMAG',
                                       'books-nc',
                                       'cuda:1',
                                       "Qwen/Qwen2.5-VL-3B-Instruct")
print(dataset)