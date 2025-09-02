#!/bin/bash

# 数据采样运行脚本 - 通过采样减少数据集大小
# 使用方法: ./run_sampling.sh [option]

echo "OpenGraph 数据采样运行脚本"
echo "=========================="

# 默认选项
OPTION=${1:-"random_10"}

case $OPTION in
    "random_10")
        echo "使用随机采样10%数据..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio 0.1 \
            --sample_method random \
            --sample_seed 42 \
            --max_nodes 50000
        ;;
    "random_5")
        echo "使用随机采样5%数据..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio 0.05 \
            --sample_method random \
            --sample_seed 42 \
            --max_nodes 30000
        ;;
    "degree_10")
        echo "使用基于度的采样10%数据..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio 0.1 \
            --sample_method degree \
            --sample_seed 42 \
            --max_nodes 50000
        ;;
    "pagerank_10")
        echo "使用基于PageRank的采样10%数据..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio 0.1 \
            --sample_method pagerank \
            --sample_seed 42 \
            --max_nodes 50000
        ;;
    "k_hop_10")
        echo "使用K跳邻居采样10%数据..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio 0.1 \
            --sample_method k_hop \
            --sample_seed 42 \
            --max_nodes 50000
        ;;
    "max_nodes_50k")
        echo "限制最大节点数为50K..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio 1.0 \
            --sample_method random \
            --sample_seed 42 \
            --max_nodes 50000
        ;;
    "max_nodes_30k")
        echo "限制最大节点数为30K..."
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch 512 \
            --tst_batch 128 \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio 1.0 \
            --sample_method random \
            --sample_seed 42 \
            --max_nodes 30000
        ;;
    "custom")
        echo "请输入自定义采样参数:"
        echo "采样比例 (0.01-1.0, 例如0.1表示10%):"
        read sample_ratio
        echo "采样方法 (random/degree/pagerank/k_hop):"
        read sample_method
        echo "最大节点数 (推荐: 10000-100000):"
        read max_nodes
        echo "batch size (推荐: 256-1024):"
        read batch_size
        
        python main.py --load pretrn_gen1 --tstdata books-nc \
            --latdim 1024 \
            --cache_proj 0 \
            --cache_adj 0 \
            --anchor 128 \
            --batch $batch_size \
            --tst_batch $((batch_size/4)) \
            --gt_layer 4 \
            --gnn_layer 3 \
            --sample_ratio $sample_ratio \
            --sample_method $sample_method \
            --sample_seed 42 \
            --max_nodes $max_nodes
        ;;
    *)
        echo "可用选项:"
        echo "  random_10      - 随机采样10%数据"
        echo "  random_5       - 随机采样5%数据"
        echo "  degree_10      - 基于度的采样10%数据"
        echo "  pagerank_10    - 基于PageRank的采样10%数据"
        echo "  k_hop_10       - K跳邻居采样10%数据"
        echo "  max_nodes_50k  - 限制最大节点数为50K"
        echo "  max_nodes_30k  - 限制最大节点数为30K"
        echo "  custom         - 自定义采样参数"
        echo ""
        echo "使用方法: ./run_sampling.sh [option]"
        exit 1
        ;;
esac
