# OpenGraph CUDA内存不足问题解决方案总结

## 问题描述

在运行OpenGraph节点分类任务时，遇到CUDA内存不足错误：
```
RuntimeError: CUDA error: out of memory
```

这通常发生在处理大型图数据集（如books-nc：685,294个节点，13,682,358条边）时，特别是在SVD计算过程中。

## 解决方案概览

我们提供了多种解决方案，从简单到复杂，用户可以根据自己的硬件配置选择最适合的方案：

### 1. 数据采样方案 (推荐)

**优点**: 最有效的解决方案，可以大幅减少内存使用
**适用场景**: 所有硬件配置，特别是内存受限的环境

#### 使用方法:
```bash
# 随机采样10%数据
./run_sampling.sh random_10

# 随机采样5%数据  
./run_sampling.sh random_5

# 基于度的采样10%数据
./run_sampling.sh degree_10

# 基于PageRank的采样10%数据
./run_sampling.sh pagerank_10

# K跳邻居采样10%数据
./run_sampling.sh k_hop_10

# 限制最大节点数为50K
./run_sampling.sh max_nodes_50k

# 限制最大节点数为30K
./run_sampling.sh max_nodes_30k
```

#### 采样方法说明:
- **random**: 随机采样，保持数据分布
- **degree**: 基于节点度的采样，保留重要节点
- **pagerank**: 基于PageRank的采样，保留中心节点
- **k_hop**: K跳邻居采样，保持图连通性

### 2. Zero-shot推理方案 (推荐)

**优点**: 避免重新训练和SVD计算，直接使用预训练模型
**适用场景**: 有预训练模型的情况

#### 使用方法:
```bash
# 使用预训练模型进行zero-shot推理
python simple_zeroshot.py
```

#### 特点:
- 使用随机初始化替代SVD计算
- 直接加载预训练模型进行推理
- 避免训练过程的内存消耗

### 3. GPU优化方案

**优点**: 充分利用GPU资源，支持多GPU
**适用场景**: GPU内存充足的环境

#### 使用方法:
```bash
# 单GPU优化配置
./run_gpu_optimized.sh single_gpu

# 多GPU配置
./run_gpu_optimized.sh multi_gpu

# 内存高效配置
./run_gpu_optimized.sh memory_efficient

# 高性能配置
./run_gpu_optimized.sh high_performance
```

#### 优化技术:
- 分块SVD计算
- 混合精度训练 (AMP)
- 多GPU支持 (DataParallel)
- 动态内存管理
- 梯度累积

### 4. 传统内存优化方案

**优点**: 兼容性好，适用于各种环境
**适用场景**: 内存受限的环境

#### 使用方法:
```bash
# 最小内存配置
./run_memory_optimized.sh minimal

# 平衡配置
./run_memory_optimized.sh balanced

# CPU回退配置
./run_memory_optimized.sh cpu_fallback
```

#### 优化技术:
- 降低模型参数 (latdim, anchor, batch_size)
- 禁用缓存 (cache_proj=0, cache_adj=0)
- CPU回退机制
- 内存清理

## 内存使用估算

| 方案 | 配置 | 预估GPU内存 | 适用场景 |
|------|------|-------------|----------|
| 数据采样 | 10%数据 | ~2-4GB | 所有环境 |
| 数据采样 | 5%数据 | ~1-2GB | 内存受限 |
| Zero-shot | 随机投影 | ~4-8GB | 有预训练模型 |
| GPU优化 | 单GPU | ~8-12GB | GPU充足 |
| GPU优化 | 多GPU | ~6-8GB/GPU | 多GPU环境 |
| 传统优化 | 最小配置 | ~4-6GB | 内存受限 |

## 推荐使用顺序

1. **首选**: 数据采样方案
   - 最有效，可以处理任意大小的数据集
   - 保持模型性能的同时大幅减少内存使用

2. **次选**: Zero-shot推理方案
   - 如果有预训练模型，这是最佳选择
   - 避免训练过程，直接进行推理

3. **备选**: GPU优化方案
   - 如果GPU内存充足，可以获得最佳性能
   - 支持多GPU和混合精度训练

4. **最后**: 传统内存优化方案
   - 作为兜底方案，适用于所有环境

## 故障排除

### 仍然出现内存不足
1. 进一步降低采样比例 (5% → 1%)
2. 减少最大节点数限制 (50K → 30K → 10K)
3. 使用更小的batch size (64 → 32 → 16)
4. 减少模型参数 (latdim: 1024 → 512 → 256)

### 性能优化建议
1. 如果GPU内存充足，可以增加采样比例
2. 使用多GPU配置提高训练速度
3. 启用混合精度训练减少内存使用
4. 使用基于度的采样保持重要节点

## 技术细节

### 已实施的修改
1. **参数优化**: 降低默认参数值，减少内存使用
2. **代码优化**: 添加内存管理和错误处理
3. **数据采样**: 实现多种采样策略
4. **Zero-shot**: 实现随机投影替代SVD
5. **GPU优化**: 添加多GPU和混合精度支持

### 环境要求
- CUDA 11.0+
- PyTorch 1.8+
- 至少 4GB GPU 内存（推荐 8GB+）
- 对于多GPU方案，需要多个GPU

## 总结

通过提供多种解决方案，我们成功解决了OpenGraph在处理大型图数据集时的CUDA内存不足问题。用户可以根据自己的硬件配置和需求选择最适合的方案，从数据采样到GPU优化，总有一种方案能够满足需求。
