# -*- coding: utf-8 -*-
"""
QE (Quality Evaluation) 任务运行器

该脚本是所有下游多模态质量评估任务的统一入口。
它负责：
1.  解析配置文件。
2.  根据任务类型，调用相应的训练器获取一个或多个训练好的模型。
3.  根据配置选择、实例化并运行相应的评估器。
"""

import argparse
import yaml
from pathlib import Path
import sys
import os
from collections import OrderedDict

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.trainers.gnn_trainer import GNNTrainer
from src.multimodal_centric.qe.trainers.retrieval_trainer import RetrievalTrainer
from src.multimodal_centric.qe.evaluators.modality_matching import MatchingEvaluator
from src.multimodal_centric.qe.evaluators.modality_retrieval import RetrievalEvaluator
from src.multimodal_centric.qe.evaluators.modality_alignment import AlignmentEvaluator

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
        help='指向任务配置文件的路径'
    )
    args = parser.parse_args()

    # 1. 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 at {args.config}")
        sys.exit(1)

    print("--- 配置加载成功 ---")
    print(yaml.dump(config, indent=2))

    task_name = config.get('task', {}).get('name')
    evaluator = None

    # 根据任务类型，执行不同的训练和评估流程
    if task_name == 'modality_matching':
        print("\n--- 任务: 模态匹配 ---")
        # 阶段一: 训练或加载GNN模型
        gnn_trainer = GNNTrainer(config)
        gnn_model = gnn_trainer.train_or_load_model()
        # 实例化评估器
        evaluator = MatchingEvaluator(config, gnn_model)

    elif task_name == 'modality_retrieval':
        print("\n--- 任务: 模态检索 (两阶段) ---")
        # RetrievalTrainer 内部会处理第一阶段的GNN模型加载
        retrieval_trainer = RetrievalTrainer(config)
        # 第二阶段: 训练或加载检索专用的双塔模型
        retrieval_model = retrieval_trainer.train_or_load_model()
        # 从检索训练器中获取第一阶段的GNN模型以供评估器使用
        gnn_model = retrieval_trainer.gnn_trainer.model
        evaluator = RetrievalEvaluator(config, gnn_model, retrieval_model)

    elif task_name == 'modality_alignment':
        print("\n--- 任务: 模态对齐 (暂未完全实现) ---")
        # 未来可能也会有两阶段训练
        gnn_trainer = GNNTrainer(config)
        gnn_model = gnn_trainer.train_or_load_model()
        evaluator = AlignmentEvaluator(config, gnn_model)
        
    else:
        print(f"错误: 未知的任务名称 '{task_name}'。请检查配置文件。")
        sys.exit(1)
    
    print(f"{task_name.capitalize()}评估器已成功实例化。")

    # 运行评估
    print("\n--- 开始评估 ---")
    results = evaluator.evaluate()
    
    # 在打印前将所有OrderedDict转换为dict
    results_dict = convert_ordereddict_to_dict(results)

    print("\n--- 评估完成 ---")
    print("最终结果:")
    print(yaml.dump(results_dict, indent=2, default_flow_style=False))

if __name__ == '__main__':
    main()