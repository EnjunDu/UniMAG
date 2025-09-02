# OpenGraph 新数据格式支持

本文档说明如何使用修改后的 OpenGraph 代码来支持新的数据格式。

## 支持的数据格式

### 新格式（自动检测）
代码会自动检测数据目录中是否存在以下文件来判断使用新格式：
- `lp-edge-split.pt` - 边分割数据（训练/验证/测试）
- `node_mapping.pt` - 节点映射文件（可选）
- `{dataset_name}-images_clip_embeddings.pt` - CLIP嵌入特征（可选）

### 旧格式（向后兼容）
如果没有找到新格式文件，会自动使用原有格式：
- `trn_mat.pkl` - 训练邻接矩阵
- `tst_mat.pkl` - 测试邻接矩阵  
- `val_mat.pkl` - 验证邻接矩阵

## 新格式数据结构

### lp-edge-split.pt
```python
{
    'train': {
        'source_node': torch.Tensor,  # 训练边的源节点
        'target_node': torch.Tensor   # 训练边的目标节点
    },
    'valid': {
        'source_node': torch.Tensor,      # 验证边的源节点
        'target_node': torch.Tensor,      # 验证边的目标节点
        'target_node_neg': torch.Tensor   # 负样本节点 (可选)
    },
    'test': {
        'source_node': torch.Tensor,      # 测试边的源节点
        'target_node': torch.Tensor,      # 测试边的目标节点
        'target_node_neg': torch.Tensor   # 负样本节点 (可选)
    }
}
```

### CLIP嵌入文件
- 形状: `[num_nodes, embedding_dim]`
- 如果存在，会自动用作初始节点特征

## 使用方法

### 方法1：直接使用main.py
```bash
python main.py --trndata cloth-copurchase --tstdata cloth-copurchase --data_dir /path/to/data
```

### 方法2：使用新的运行脚本
```bash
python run_new_dataset.py --trndata cloth-copurchase --tstdata cloth-copurchase --data_dir /path/to/data
```

### 方法3：修改main.py中的数据集设置
在main.py的最后部分修改：
```python
trn_datasets = ['cloth-copurchase']  # 或其他新格式数据集
tst_datasets = ['cloth-copurchase']
```

## 数据目录结构示例

```
/path/to/data/
├── cloth-copurchase/
│   ├── lp-edge-split.pt                      # 必需：边分割数据
│   ├── node_mapping.pt                       # 可选：节点映射
│   └── cloth-copurchase-images_clip_embeddings.pt  # 可选：CLIP嵌入
└── old-dataset/
    ├── trn_mat.pkl                           # 旧格式训练数据
    ├── tst_mat.pkl                           # 旧格式测试数据
    └── val_mat.pkl                           # 旧格式验证数据
```

## 主要特性

1. **自动格式检测** - 无需手动指定数据格式
2. **向后兼容** - 完全支持原有的pkl格式
3. **CLIP嵌入集成** - 自动使用预训练的CLIP特征
4. **错误处理** - 对加载失败的文件进行优雅处理
5. **灵活文件命名** - 支持多种嵌入文件命名约定

## 参数说明

- `--data_dir`: 数据根目录路径
- `--trndata`: 训练数据集名称
- `--tstdata`: 测试数据集名称
- 其他参数与原版OpenGraph保持一致

## 注意事项

1. 确保数据文件路径正确
2. 新格式的数据集会自动构建对称的邻接矩阵
3. 如果CLIP嵌入加载失败，会自动回退到传统的SVD投影
4. node_mapping.pt文件如果加载失败不会影响主要功能


 python main.py --trndata cloth-copurchase --tstdata cloth-copurchase --data_dir /data/EnjunDu/MMAG --gpu 4

 cd /home/daihengwei/EnjunDu/M2AGraph/Baselines/OpenGraph/OpenGraph/link_prediction && CUDA_VISIBLE_DEVICES=4 python main.py --trndata cloth-copurchase --tstdata cloth-copurchase --data_dir /data/EnjunDu/MMAG --gpu 4