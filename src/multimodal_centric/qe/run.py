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
from omegaconf import DictConfig, OmegaConf

# 将项目根目录添加到Python路径中
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.multimodal_centric.qe.trainers.gnn_trainer import GNNTrainer
from src.multimodal_centric.qe.trainers.retrieval_trainer import RetrievalTrainer
from src.multimodal_centric.qe.evaluators.modality_matching import MatchingEvaluator
from src.multimodal_centric.qe.evaluators.modality_retrieval import RetrievalEvaluator
from src.multimodal_centric.qe.evaluators.modality_alignment import AlignmentEvaluator

def represent_ordereddict(dumper, data):
    """自定义YAML Dumper以正确处理OrderedDict。"""
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

yaml.add_representer(OrderedDict, represent_ordereddict)

def run_qe(cfg: DictConfig):
    """
    QE任务的统一执行函数，由 src/main.py 调用。
    """
    print("--- QE 任务模块已启动 ---")
    print(OmegaConf.to_yaml(cfg))

    task_name = cfg.task.name
    evaluator = None

    # 根据任务类型，执行不同的训练和评估流程
    if task_name == 'modality_matching':
        print("\n--- 任务: 模态匹配 ---")
        gnn_trainer = GNNTrainer(cfg)
        gnn_model = gnn_trainer.train_or_load_model()
        evaluator = MatchingEvaluator(cfg, gnn_model)

    elif task_name == 'modality_retrieval':
        print("\n--- 任务: 模态检索 (两阶段) ---")
        retrieval_trainer = RetrievalTrainer(cfg)
        retrieval_model = retrieval_trainer.train_or_load_model()
        gnn_model = retrieval_trainer.gnn_trainer.model
        evaluator = RetrievalEvaluator(cfg, gnn_model, retrieval_model)

    elif task_name == 'modality_alignment':
        print("\n--- 任务: 模态对齐 ---")
        gnn_trainer = GNNTrainer(cfg)
        gnn_model = gnn_trainer.train_or_load_model()
        evaluator = AlignmentEvaluator(cfg, gnn_model)
        
    else:
        print(f"错误: 未知的任务名称 '{task_name}'。请检查配置文件。")
        sys.exit(1)
    
    print(f"{task_name.capitalize()}评估器已成功实例化。")

    # 运行评估
    print("\n--- 开始评估 ---")
    results = evaluator.evaluate()

    print("\n--- 评估完成 ---")
    print("最终结果:")
    # 使用 OmegaConf.to_yaml 以更好地处理 Hydra 配置对象
    print(OmegaConf.to_yaml(results))
    return results

def main_standalone():
    """主函数，用于独立、快速地调试QE任务，不通过 src/main.py。"""
    parser = argparse.ArgumentParser(description="独立运行多模态质量评估（QE）任务")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='指向任务配置文件的路径 (例如: configs/task/qe_matching_gcn_grocery.yaml)'
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 at {args.config}")
        sys.exit(1)
    
    # 将字典转换为 OmegaConf 对象以调用新函数
    cfg = OmegaConf.create(config_dict)
    run_qe(cfg)

if __name__ == '__main__':
    main_standalone()