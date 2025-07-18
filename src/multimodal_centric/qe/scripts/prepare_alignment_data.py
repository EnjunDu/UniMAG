# -*- coding: utf-8 -*-
"""
为模态对齐任务准备预处理数据（并行版）。

该脚本整合了两个核心阶段，并使用多进程并行处理来加速：
1.  (Stage 1) 生成基准真值: 
    -   从原始数据中提取名词短语。
    -   使用GroundingDINO找到短语对应的边界框。
    -   将结果保存为 .jsonl 文件。
2.  (Stage 2) 提取并缓存特征:
    -   读取 .jsonl 真值文件。
    -   为每个 (短语, 边界框) 对提取文本嵌入和图像区域嵌入。
    -   将所有信息保存到一个统一的 .pt 文件中。
"""
import sys
import os
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import torch
import spacy
from PIL import Image
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np
import torch.multiprocessing as mp
import time
import yaml
import traceback

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.embedding_manager import EmbeddingManager

def worker_stage1(task_queue, result_queue, device_id):
    """Stage 1 的工作进程函数。"""
    # 延迟加载：在worker函数内部为每个进程创建独立的模型实例
    device = f"cuda:{device_id}"
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    nlp = spacy.load("en_core_web_sm")
    manager = EmbeddingManager()
    max_text_length = model.config.text_config.max_position_embeddings // 2

    while True:
        task = task_queue.get()
        if task is None: break
        
        node_index, dataset_name = task
        raw_data = manager.get_raw_data_by_index(dataset_name, node_index)
        
        status = "unknown_error"
        grounding_results = []
        error_info = ""

        if not raw_data or not raw_data.get("image_path") or not raw_data.get("text"):
            status = "skipped_no_raw_data"
        else:
            image_path, text = raw_data["image_path"], raw_data["text"]
            if not os.path.exists(image_path): 
                status = "image_not_found"
            else:
                image = None
                for attempt in range(3):
                    try:
                        image = Image.open(image_path).convert("RGB")
                        break
                    except Exception:
                        time.sleep(0.1 * (attempt + 1))
                
                if image is None:
                    status = "image_open_failed"
                else:
                    try:
                        doc = nlp(text)
                        phrases = [chunk.text for chunk in doc.noun_chunks]
                        if not phrases: status = "no_noun_phrases"
                        else:
                            text_for_grounding = ". ".join(phrases)
                            inputs = processor(
                                images=image, 
                                text=text_for_grounding, 
                                return_tensors="pt", 
                                padding="max_length", # 填充到max_length
                                truncation=True,      # 截断超过max_length的部分
                                max_length=max_text_length
                            ).to(device)
                            with torch.no_grad():
                                outputs = model(**inputs)
                            results = processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.25, target_sizes=[image.size[::-1]])
                            grounding_results = [{"phrase": label, "box": box.cpu().numpy().tolist()} for label, box in zip(results[0]["labels"], results[0]["boxes"])]
                            if not grounding_results: status = "grounding_failed"
                            else: status = "success"
                    except Exception:
                        status = "processing_error_after_open"
                        error_info = traceback.format_exc()
        
        result_queue.put((node_index, status, raw_data, grounding_results, error_info))

def run_stage1_parallel(dataset_name: str, output_file: Path, workers_per_gpu: List[int]):
    """执行 Stage 1 的并行版本。"""
    print("\n--- [Stage 1] 开始并行生成基准真值文件 ---")
    manager = EmbeddingManager()
    try:
        node_ids = manager._storage.load_node_ids(dataset_name)
        num_nodes = len(node_ids)
    except FileNotFoundError as e:
        print(f"错误: 无法加载节点ID文件: {e}")
        sys.exit(1)

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for i in range(num_nodes):
        task_queue.put((i, dataset_name))

    processes = []
    device_map = []
    for gpu_id, num_workers in enumerate(workers_per_gpu):
        for _ in range(num_workers):
            device_map.append(gpu_id)
    
    for _ in range(len(device_map)):
        task_queue.put(None)

    for rank, device_id in enumerate(device_map):
        p = mp.Process(target=worker_stage1, args=(task_queue, result_queue, device_id))
        p.start()
        processes.append(p)

    stats = defaultdict(int)
    results_buffer = {}
    failed_nodes_info = []
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(num_nodes), desc="Stage 1: 生成真值"):
            node_index, status, raw_data, grounding_results, error_info = result_queue.get()
            stats["total_nodes"] += 1
            stats[status] += 1
            if status == "success":
                record = {"node_index": node_index, "node_id": node_ids[node_index], "image_path": raw_data["image_path"], "text": raw_data["text"], "grounding": grounding_results}
                results_buffer[node_index] = record
            elif error_info:
                failed_nodes_info.append(f"  - 节点 {node_index}: 状态={status}\n    错误详情:\n{error_info}\n")

        for i in sorted(results_buffer.keys()):
            f.write(json.dumps(results_buffer[i], ensure_ascii=False) + "\n")

    for p in processes: p.join()

    print("\n--- [Stage 1] 诊断统计 ---")
    print(f"总共处理节点: {stats['total_nodes']}, 成功生成真值: {stats['success']}")
    print("失败/跳过原因:")
    for reason, count in stats.items():
        if reason not in ["total_nodes", "success"]: 
            print(f"  - {reason}: {count}")
    
    if failed_nodes_info:
        print("\n--- 详细错误报告 ---")
        for info in failed_nodes_info:
            print(info)

    print(f"--- [Stage 1] 完成，真值文件已保存到: {output_file} ---")
    return output_file

# ... Stage 2 的并行实现将遵循类似模式 ...
# 为了保持代码清晰，我们可以在确认Stage 1工作正常后再实现Stage 2的并行化。
# 这里暂时保留Stage 2的串行实现。

def run_stage2_extract_features(ground_truth_file: Path, output_file: Path, config: Dict[str, Any]):
    """执行 Stage 2，提取并缓存特征。"""
    print("\n--- [Stage 2] 开始提取并缓存特征 ---")
    if not ground_truth_file.exists():
        print(f"错误: 找不到真值文件 '{ground_truth_file}'。请先运行 Stage 1 或提供正确的文件路径。")
        sys.exit(1)

    manager = EmbeddingManager()
    preprocessed_data = []
    
    encoder_type = config['embedding']['encoder_type']
    encoder_name = config['embedding']['encoder_name']
    dimension = config['embedding']['dimension']

    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Stage 2: 提取特征"):
            record = json.loads(line)
            node_index = record["node_index"]
            image_path = record["image_path"]
            
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"警告: 无法打开图像 {image_path}，跳过节点 {node_index}。错误: {e}")
                continue

            for grounding_pair in record["grounding"]:
                phrase = grounding_pair["phrase"]
                box = grounding_pair["box"]

                phrase_embed = manager.generate_embedding(data=[phrase], modality="text", encoder_type=encoder_type, encoder_name=encoder_name, dimension=dimension)
                if phrase_embed is None: continue

                region_image = image.crop((box[0], box[1], box[2], box[3]))
                region_embed = manager.generate_embedding(data=[region_image], modality="image", encoder_type=encoder_type, encoder_name=encoder_name, dimension=dimension)
                if region_embed is None: continue
                
                preprocessed_data.append({
                    "node_index": node_index,
                    "phrase": phrase,
                    "box": box,
                    "phrase_embedding": torch.from_numpy(phrase_embed).float().squeeze(),
                    "region_embedding": torch.from_numpy(region_embed).float().squeeze()
                })

    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(preprocessed_data, output_file)
    print(f"--- [Stage 2] 完成，预处理特征已保存到: {output_file} ---")


def main():
    parser = argparse.ArgumentParser(description="为模态对齐任务准备数据（并行版）。")
    parser.add_argument("--dataset", type=str, required=True, help="要处理的数据集名称。")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "1", "generate", "2", "extract"], help="要执行的阶段。")
    parser.add_argument("--config", type=str, help="指向任务配置文件的路径 (用于Stage 2获取嵌入参数)。")
    parser.add_argument("--ground_truth_file", type=Path, default=None, help="[Stage 2] 手动指定输入的真值文件路径。")
    parser.add_argument("--output_file", type=Path, default=None, help="[Stage 2] 手动指定最终预处理数据的输出文件路径。")
    parser.add_argument("--workers-per-gpu", type=int, nargs='+', help="[Stage 1] 指定每张GPU上启动的worker数量，例如 '2 4 1 1'。")
    args = parser.parse_args()

    base_output_dir = Path(__file__).resolve().parent / "ground_truth"
    default_gt_file = base_output_dir / f"{args.dataset}_ground_truth.jsonl"
    default_output_file = base_output_dir / f"{args.dataset}_alignment_preprocessed.pt"

    ground_truth_file_to_use = args.ground_truth_file or default_gt_file
    final_output_file = args.output_file or default_output_file

    if args.stage in ["all", "1", "generate"]:
        if not args.workers_per_gpu:
            print("错误: 在执行Stage 1时，必须通过 '--workers-per-gpu' 参数指定worker配置。")
            sys.exit(1)
        generated_gt_path = run_stage1_parallel(args.dataset, default_gt_file, args.workers_per_gpu)
        if args.stage in ["all"]:
            if not args.config:
                print("错误: 在 'all' 模式下，必须提供 '--config' 文件以进行Stage 2。")
                sys.exit(1)
            with open(args.config, 'r') as f: config = yaml.safe_load(f)
            run_stage2_extract_features(generated_gt_path, final_output_file, config)

    elif args.stage in ["2", "extract"]:
        if not args.config:
            print("错误: 在执行Stage 2时，必须提供 '--config' 文件。")
            sys.exit(1)
        with open(args.config, 'r') as f: config = yaml.safe_load(f)
        run_stage2_extract_features(ground_truth_file_to_use, final_output_file, config)

if __name__ == "__main__":
    # 设置多进程启动方法为 'spawn' 以确保CUDA在子进程中正确初始化
    mp.set_start_method("spawn", force=True)
    main()
