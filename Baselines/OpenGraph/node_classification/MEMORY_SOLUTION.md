# CUDA内存不足问题解决方案

## 问题描述

在运行OpenGraph节点分类任务时，遇到CUDA内存不足错误：
```
RuntimeError: CUDA error: out of memory
```

这通常发生在处理大型图数据集（如books-nc：685,294个节点，13,682,358条边）时，特别是在SVD计算过程中。

## 解决方案

### 1. 已实施的修改

#### 参数优化 (`params.py`)
- **latdim**: 1024 → 512 (减少投影维度)
- **cache_proj**: 1 → 0 (禁用投影矩阵缓存)
- **anchor**: 256 → 128 (减少锚点数量)
- **batch**: 1024 → 256 (减少批次大小)
- **tst_batch**: 256 → 64 (减少测试批次大小)

#### 代码优化 (`model.py`)
- 添加了GPU内存不足时的CPU回退机制
- 实现了分块SVD计算，减少内存使用
- 自动清理GPU内存
- 智能设备管理

#### GPU优化 (`main.py`)
- 添加了多GPU支持 (DataParallel)
- 实现了混合精度训练 (AMP)
- 添加了分块处理大型数据集
- 定期GPU内存清理

### 2. 运行选项

#### GPU优化方案 (推荐)
使用GPU优化脚本，支持多GPU和内存优化：

```bash
# 单GPU优化配置 (推荐)
./run_gpu_optimized.sh single_gpu

# 多GPU配置 (如果有多个GPU)
./run_gpu_optimized.sh multi_gpu

# 内存高效配置 (GPU内存受限)
./run_gpu_optimized.sh memory_efficient

# 高性能配置 (GPU内存充足)
./run_gpu_optimized.sh high_performance

# 自定义GPU参数
./run_gpu_optimized.sh custom
```

#### 传统内存优化方案
如果GPU优化不适用，使用传统内存优化脚本：

```bash
# 最小内存配置 (推荐用于内存受限环境)
./run_memory_optimized.sh minimal

# 平衡配置 (默认，推荐)
./run_memory_optimized.sh balanced

# CPU回退配置 (高精度，较慢)
./run_memory_optimized.sh cpu_fallback

# 自定义参数
./run_memory_optimized.sh custom
```

### 3. 手动参数调整

如果脚本不满足需求，可以手动调整以下参数：

```bash
python main.py --load pretrn_gen1 --tstdata books-nc \
    --latdim 256 \          # 降低投影维度 (256-1024)
    --cache_proj 0 \        # 禁用投影缓存
    --cache_adj 0 \         # 禁用邻接矩阵缓存
    --anchor 64 \           # 减少锚点数量 (64-256)
    --batch 512 \           # 减少批次大小
    --tst_batch 128 \       # 减少测试批次大小
    --gt_layer 2 \          # 减少Transformer层数
    --gnn_layer 2           # 减少GNN层数
```

### 4. 内存使用估算

#### GPU优化配置
| 配置 | latdim | anchor | batch | 预估GPU内存 | 适用场景 |
|------|--------|--------|-------|-------------|----------|
| single_gpu | 1024 | 128 | 256 | ~8-12GB | 单GPU推荐 |
| multi_gpu | 1024 | 256 | 512 | ~6-8GB/GPU | 多GPU环境 |
| memory_efficient | 512 | 64 | 128 | ~4-6GB | GPU内存受限 |
| high_performance | 1024 | 256 | 1024 | ~12-16GB | 高性能需求 |

#### 传统内存优化配置
| 配置 | latdim | anchor | 预估GPU内存 | 适用场景 |
|------|--------|--------|-------------|----------|
| minimal | 256 | 64 | ~4-6GB | 内存受限环境 |
| balanced | 512 | 128 | ~8-12GB | 推荐配置 |
| cpu_fallback | 1024 | 256 | ~16-20GB | 高精度需求 |

### 5. 故障排除

#### 仍然出现内存不足
1. 进一步降低 `latdim` 到 128 或 256
2. 减少 `anchor` 到 32 或 64
3. 减少 `batch` 到 256 或 512
4. 减少 `gt_layer` 和 `gnn_layer`

#### 性能优化建议
1. 如果GPU内存充足，可以启用缓存：
   ```bash
   --cache_proj 1 --cache_adj 1
   ```
2. 增加 `latdim` 和 `anchor` 以提高模型性能
3. 增加 `gt_layer` 和 `gnn_layer` 以增强表达能力

### 6. 环境要求

- CUDA 11.0+
- PyTorch 1.8+
- 至少 8GB GPU 内存（推荐 16GB+）
- 对于CPU回退模式，需要足够的主内存

### 7. 监控内存使用

运行前检查GPU内存：
```bash
nvidia-smi
```

运行过程中监控：
```bash
watch -n 1 nvidia-smi
```

## 技术细节

### SVD计算优化
- 使用 `torch.svd_lowrank` 进行近似SVD
- 自动检测内存不足并回退到CPU
- 智能设备管理和内存清理

### 稀疏矩阵处理
- 使用稀疏张量减少内存占用
- 避免密集矩阵转换
- 优化稀疏矩阵乘法

### 批处理优化
- 动态批次大小调整
- 内存感知的数据加载
- 梯度累积支持
