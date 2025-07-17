# -*- coding: utf-8 -*-
"""
QE (Quality Evaluation) 任务运行器

该脚本是所有下游多模态质量评估任务的统一入口。
它负责：
1.  解析配置文件。
2.  调用训练器 (trainer.py) 来获取一个为特定数据集和GNN模型训练好的模型。
3.  根据配置选择、实例化并运行相应的评估器 (例如 Modality Matching)。
"""

import argparse
import yaml
from pathlib import Path
import sys
import os
from collections import OrderedDict
import json

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.trainer import GNNTrainer
from src.multimodal_centric.qe.modality_matching import MatchingEvaluator
from src.multimodal_centric.qe.modality_retrieval import RetrievalEvaluator
from src.multimodal_centric.qe.modality_alignment import AlignmentEvaluator

def convert_ordereddict_to_dict(d):
    """递归地将OrderedDict转换为dict，以便美观地打印。"""
    if isinstance(d, OrderedDict):
        d = dict(d)
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = convert_ordereddict_to_dict(value)
    return d

def main():
    """主函数，作为整个QE任务的入口。"""
    parser = argparse.ArgumentParser(description="运行多模态质量评估（QE）任务")
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, 
        help='指向任务配置文件的路径 (例如: configs/task/qe_matching_gcn_grocery.yaml)'
    )
    args = parser.parse_args()

    # 1. 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"解析配置文件时出错: {e}")
        sys.exit(1)

    print("--- 配置加载成功 ---")
    print(yaml.dump(config, indent=2))

    # 2. 训练或加载GNN模型
    print("\n--- 步骤 2: 准备GNN模型 ---")
    trainer = GNNTrainer(config)
    trained_gnn_model = trainer.train_or_load_model()
    print("GNN模型已准备就绪。")

    # 3. 根据任务选择并实例化评估器
    print("\n--- 步骤 3: 准备评估器 ---")
    task_name = config.get('task', {}).get('name')
    evaluator = None

    print(f"根据任务 '{task_name}' 实例化评估器。")
    if task_name == 'modality_matching':
        evaluator = MatchingEvaluator(config, trained_gnn_model)
    elif task_name == 'modality_retrieval':
        evaluator = RetrievalEvaluator(config, trained_gnn_model)
    elif task_name == 'modality_alignment':
        evaluator = AlignmentEvaluator(config, trained_gnn_model)
    else:
        print(f"错误: 未知的任务名称 '{task_name}'。请检查配置文件。")
        sys.exit(1)
    
    print(f"{task_name.capitalize()}评估器已成功实例化。")

    # 4. 运行评估
    print("\n--- 步骤 4: 开始评估 ---")
    results = evaluator.evaluate()
    
    # 在打印前将所有OrderedDict转换为dict
    results_dict = convert_ordereddict_to_dict(results)

    print("\n--- 评估完成 ---")
    print("最终结果:")
    print(yaml.dump(results_dict, indent=2, default_flow_style=False))

if __name__ == '__main__':
    main()