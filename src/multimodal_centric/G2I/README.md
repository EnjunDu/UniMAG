# G2Image

## environment
ues the environment same as InstructG2I:https://github.com/PeterGriffinJin/InstructG2I/tree/main

## process dataset
```
python process.py --train_ratio=(你希望划分的训练集占总体的比例) \
				--read_data_path=(数据集.pt文件路径) \
				--csv_path=(数据集.csv文件路径) \
				--save_data_path=(处理后的数据集保存位置) \
```


## train/test

1.请在G2I文件夹下进行操作

2.在config文件夹下，建立对应预训练数据集的train.json 

3.训练：
```bash
python decoder/train.py
```

测试：
```
python decoder/test.py
```

