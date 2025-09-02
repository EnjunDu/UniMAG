# OpenGraph 性能优化说明

## 评估指标变更

### 原来：Recall 和 NDCG
- **优点**：计算快速，适合推荐系统
- **缺点**：不是标准的链接预测评估指标

### 现在：MRR、Hits@1、Hits@10
- **优点**：标准的链接预测评估指标，与学术论文一致
- **缺点**：需要生成负样本，计算复杂度增加

## 性能优化策略

### 1. 负样本数量控制
```bash
# 默认使用100个负样本（平衡性能和准确性）
python main.py --trndata cloth-copurchase --tstdata cloth-copurchase --data_dir /data/EnjunDu/MMAG --gpu 4

# 使用更少负样本（更快但可能不够准确）
python main.py --trndata cloth-copurchase --tstdata cloth-copurchase --data_dir /data/EnjunDu/MMAG --gpu 4 --num_neg_eval 50

# 使用更多负样本（更准确但更慢）
python main.py --trndata cloth-copurchase --tstdata cloth-copurchase --data_dir /data/EnjunDu/MMAG --gpu 4 --num_neg_eval 200
```

### 2. 性能对比

| 负样本数量 | 评估时间 | 准确性 | 推荐场景 |
|------------|----------|--------|----------|
| 50         | 最快     | 一般   | 快速验证 |
| 100        | 中等     | 良好   | 默认设置 |
| 200        | 较慢     | 更好   | 最终评估 |
| 1000       | 最慢     | 最好   | 论文发表 |

### 3. 优化建议

#### 训练阶段
- 使用 `--num_neg_eval 50` 进行快速验证
- 关注训练损失变化，而不是评估指标

#### 最终评估
- 使用 `--num_neg_eval 100` 或 `200` 进行准确评估
- 可以运行多次取平均值

#### 如果仍然太慢
- 减少 `--tst_batch` 大小
- 使用更少的验证/测试数据
- 考虑使用原来的 Recall/NDCG 指标

## 技术细节

### 负样本生成优化
- 使用高效的稀疏矩阵操作
- 避免重复的边检查
- 批量处理用户预测

### 内存优化
- 及时释放不需要的张量
- 使用 `t.cuda.empty_cache()` 清理GPU内存
- 避免存储过多的中间结果

## 恢复原指标（如果需要）

如果性能问题无法解决，可以临时恢复原来的评估指标：

```python
# 在 main.py 中注释掉新的评估逻辑
# 使用原来的 calc_recall_ndcg 方法
```

## 总结

新的评估指标提供了更标准的链接预测评估，但确实会增加计算开销。通过调整负样本数量和优化算法，可以在准确性和性能之间找到平衡点。
