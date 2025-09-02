#!/bin/bash

# 内存优化运行脚本 - 解决CUDA内存不足问题
# 使用方法: ./run_memory_optimized.sh [option]

echo "OpenGraph 内存优化运行脚本"
echo "=========================="

# 默认选项
OPTION=${1:-"balanced"}

case $OPTION in
    "minimal")
        echo "使用最小内存配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 256 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 64 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 2 \
            --gnn_layer 2
        ;;
    "balanced")
        echo "使用平衡配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 512 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 1024 \
            --tst_batch 256 \
            --gt_layer 4 \
            --gnn_layer 3
        ;;
    "cpu_fallback")
        echo "使用CPU回退配置..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 256 \
            --batch 1024 \
            --tst_batch 256 \
            --gt_layer 4 \
            --gnn_layer 3
        ;;
    "custom")
        echo "请输入自定义参数:"
        echo "latdim (推荐: 256-1024):"
        read latdim
        echo "anchor (推荐: 64-256):"
        read anchor
        echo "batch (推荐: 512-1024):"
        read batch
        
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim $latdim \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor $anchor \
            --batch $batch \
            --tst_batch $((batch/4))
        ;;
    *)
        echo "可用选项:"
        echo "  minimal      - 最小内存使用 (latdim=256, anchor=64)"
        echo "  balanced     - 平衡配置 (latdim=512, anchor=128) [默认]"
        echo "  cpu_fallback - 使用CPU回退 (latdim=1024, anchor=256)"
        echo "  custom       - 自定义参数"
        echo ""
        echo "使用方法: ./run_memory_optimized.sh [option]"
        exit 1
        ;;
esac
