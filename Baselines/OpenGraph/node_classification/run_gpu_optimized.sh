#!/bin/bash

# GPU优化运行脚本 - 支持多GPU和内存优化
# 使用方法: ./run_gpu_optimized.sh [option]

echo "OpenGraph GPU优化运行脚本"
echo "========================="

# 检查GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $GPU_COUNT 个GPU"

# 默认选项
OPTION=${1:-"single_gpu"}

case $OPTION in
    "single_gpu")
        echo "使用单GPU优化配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 256 \
            --tst_batch 64 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --multi_gpu 0 \
            --gpu_memory_fraction 0.8 \
            --chunk_size 5000 \
            --mixed_precision 1
        ;;
    "mini_batch")
        echo "使用超小batch配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 64 \
            --batch 32 \
            --tst_batch 16 \
            --gt_layer 2 \
            --gnn_layer 2 \
            --multi_gpu 0 \
            --gpu_memory_fraction 0.6 \
            --chunk_size 2000 \
            --mixed_precision 1
        ;;
    "micro_batch")
        echo "使用微batch配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 512 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 32 \
            --batch 16 \
            --tst_batch 8 \
            --gt_layer 1 \
            --gnn_layer 1 \
            --multi_gpu 0 \
            --gpu_memory_fraction 0.5 \
            --chunk_size 1000 \
            --mixed_precision 1
        ;;
    "multi_gpu")
        if [ $GPU_COUNT -gt 1 ]; then
            echo "使用多GPU配置..."
            python main.py --load pretrn_gen1 --tstdata books-nc \
                --latdim 1024 \
                --cache_proj 0 \
                --cache_adj 0 \
                --anchor 256 \
                --batch 512 \
                --tst_batch 128 \
                --gt_layer 4 \
                --gnn_layer 3 \
                --multi_gpu 1 \
                --gpu_memory_fraction 0.7 \
                --chunk_size 10000 \
                --mixed_precision 1
        else
            echo "警告: 只有一个GPU，使用单GPU配置"
            ./run_gpu_optimized.sh single_gpu
        fi
        ;;
    "memory_efficient")
        echo "使用内存高效配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 512 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 64 \
            --batch 128 \
            --tst_batch 32 \
            --gt_layer 2 \
            --gnn_layer 2 \
            --multi_gpu 0 \
            --gpu_memory_fraction 0.6 \
            --chunk_size 2000 \
            --mixed_precision 1
        ;;
    "high_performance")
        echo "使用高性能配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 1 \
            --cache_adj 1 \
            --anchor 256 \
            --batch 1024 \
            --tst_batch 256 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --multi_gpu 1 \
            --gpu_memory_fraction 0.9 \
            --chunk_size 15000 \
            --mixed_precision 1
        ;;
    "custom")
        echo "请输入自定义参数:"
        echo "latdim (推荐: 256-1024):"
        read latdim
        echo "batch (推荐: 8-512):"
        read batch
        echo "anchor (推荐: 16-256):"
        read anchor
        echo "是否使用多GPU? (y/n):"
        read use_multi_gpu
        echo "GPU内存使用比例 (0.1-1.0):"
        read gpu_memory_fraction
        
        if [ "$use_multi_gpu" = "y" ] && [ $GPU_COUNT -gt 1 ]; then
            multi_gpu=1
        else
            multi_gpu=0
        fi
        
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim $latdim \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor $anchor \
            --batch $batch \
            --tst_batch $((batch/4)) \
            --gt_layer 2 \
            --gnn_layer 2 \
            --multi_gpu $multi_gpu \
            --gpu_memory_fraction $gpu_memory_fraction \
            --chunk_size 2000 \
            --mixed_precision 1
        ;;
    *)
        echo "可用选项:"
        echo "  single_gpu      - 单GPU优化 (latdim=1024, batch=256)"
        echo "  mini_batch      - 超小batch (latdim=1024, batch=32)"
        echo "  micro_batch     - 微batch (latdim=512, batch=16)"
        echo "  multi_gpu       - 多GPU配置 (latdim=1024, batch=512)"
        echo "  memory_efficient - 内存高效 (latdim=512, batch=128)"
        echo "  high_performance - 高性能 (latdim=1024, batch=1024)"
        echo "  custom          - 自定义参数"
        echo ""
        echo "使用方法: ./run_gpu_optimized.sh [option]"
        exit 1
        ;;
esac
