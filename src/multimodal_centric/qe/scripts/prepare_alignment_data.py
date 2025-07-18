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
from hydra import initialize, compose
from omegaconf import OmegaConf

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
    print("\n--- [Stage 1] Start parallel generation of ground truth file ---")
    manager = EmbeddingManager()
    try:
        node_ids = manager._storage.load_node_ids(dataset_name)
        num_nodes = len(node_ids)
    except FileNotFoundError as e:
        print(f"Error: Failed to load node ID file: {e}")
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
        for _ in tqdm(range(num_nodes), desc="Stage 1: Generating ground truth"):
            node_index, status, raw_data, grounding_results, error_info = result_queue.get()
            stats["total_nodes"] += 1
            stats[status] += 1
            if status == "success":
                record = {"node_index": node_index, "node_id": node_ids[node_index], "image_path": raw_data["image_path"], "text": raw_data["text"], "grounding": grounding_results}
                results_buffer[node_index] = record
            elif error_info:
                failed_nodes_info.append(f"  - Node {node_index}: Status={status}\n     Error details:\n{error_info}\n")

        for i in sorted(results_buffer.keys()):
            f.write(json.dumps(results_buffer[i], ensure_ascii=False) + "\n")

    for p in processes: p.join()

    print("\n--- [Stage 1] Diagnostics ---")
    print(f"Total nodes processed: {stats['total_nodes']}, Successfully generated ground truth: {stats['success']}")
    print("Failed/Skipped reasons:")
    for reason, count in stats.items():
        if reason not in ["total_nodes", "success"]: 
            print(f"  - {reason}: {count}")
    
    if failed_nodes_info:
        print("\n--- Detailed error report ---")
        for info in failed_nodes_info:
            print(info)

    print(f"--- [Stage 1] Completed, ground truth file saved to: {output_file} ---")
    return output_file

def worker_stage2(tasks: List[Dict], device_id: int, config: Dict[str, Any], temp_file_path: Path):
    """Stage 2 的工作进程函数（静态分配版）。"""
    device = f"cuda:{device_id}"
    manager = EmbeddingManager(config=config, device=device)
    
    results = []
    # 使用 position=device_id 让每个worker的进度条显示在不同行
    for task in tqdm(tasks, desc=f"Worker (GPU:{device_id}) Stage 2", position=device_id+1):
        try:
            image = Image.open(task["image_path"]).convert("RGB")
            for grounding_pair in task["grounding"]:
                phrase = grounding_pair["phrase"]
                box = grounding_pair["box"]

                phrase_embed = manager.generate_embedding(data=[phrase], modality="text")
                if phrase_embed is None: continue

                region_image = image.crop((box[0], box[1], box[2], box[3]))
                region_embed = manager.generate_embedding(data=[region_image], modality="image")
                if region_embed is None: continue
                
                results.append({
                    "node_index": task["node_index"],
                    "phrase": phrase,
                    "box": box,
                    "phrase_embedding": torch.from_numpy(phrase_embed).float().squeeze(),
                    "region_embedding": torch.from_numpy(region_embed).float().squeeze()
                })
        except Exception:
            # 在静态模式下，简单跳过有问题的任务
            continue
            
    torch.save(results, temp_file_path)

def run_stage2_static_parallel(ground_truth_file: Path, output_file: Path, config: Dict[str, Any], workers_per_gpu: List[int]):
    """执行 Stage 2 的静态并行版本。"""
    print("\n--- [Stage 2] Start static parallel extraction and caching of features ---")
    if not ground_truth_file.exists():
        print(f"Error: Ground truth file not found, please run stage 1 first."); sys.exit(1)

    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    device_map = [gpu_id for gpu_id, num_workers in enumerate(workers_per_gpu) for _ in range(num_workers)]
    total_workers = len(device_map)
    
    # 将任务列表静态地分割成N个子列表
    chunks = [tasks[i::total_workers] for i in range(total_workers)]
    
    temp_dir = output_file.parent / f"temp_stage2_{int(time.time())}"
    temp_dir.mkdir(exist_ok=True)
    
    processes = []
    for i in range(total_workers):
        temp_file = temp_dir / f"part_{i}.pt"
        p = mp.Process(target=worker_stage2, args=(chunks[i], device_map[i], config, temp_file))
        p.start()
        processes.append(p)
        
    for p in processes: p.join()
    
    # 合并所有临时文件的结果
    final_data = []
    print("--- [Stage 2] All worker processes completed, merging results... ---")
    for temp_file_path in sorted(temp_dir.glob("part_*.pt")):
        final_data.extend(torch.load(temp_file_path))
        os.remove(temp_file_path)
    os.rmdir(temp_dir)
    
    torch.save(final_data, output_file)
    print(f"--- [Stage 2] Completed, processed {len(final_data)} feature pairs, preprocessed features saved to: {output_file} ---")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for modality alignment task (parallel version).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to process (e.g., toys). This must match a file in configs/dataset/.")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "1", "2"], help="Stage to execute. '1' for ground truth, '2' for features.")
    parser.add_argument("--ground_truth_file", type=Path, default=None, help="Manually specify the input ground truth file path for Stage 2.")
    parser.add_argument("--output_file", type=Path, default=None, help="Manually specify the output file path for the final preprocessed data for Stage 2.")
    parser.add_argument("--workers-per-gpu", type=int, nargs='+', required=True, help="Specify the number of workers to start per GPU, e.g., '1 3 3 1'.")
    args = parser.parse_args()

    relative_config_path = "../../../../configs"
    with initialize(config_path=relative_config_path, job_name="alignment_preprocess", version_base="1.2"):
        cfg = compose(config_name="config", overrides=[f"dataset={args.dataset.lower()}"])
        config = OmegaConf.to_container(cfg, resolve=True)
    
    dataset_name = cfg.dataset.name

    base_output_dir = Path(__file__).resolve().parent / "ground_truth"
    default_gt_file = base_output_dir / f"{dataset_name}_ground_truth.jsonl"
    default_output_file = base_output_dir / f"{dataset_name}_alignment_preprocessed.pt"

    ground_truth_file_to_use = args.ground_truth_file or default_gt_file
    final_output_file = args.output_file or default_output_file

    if args.stage in ["all", "1"]:
        generated_gt_path = run_stage1_parallel(dataset_name, default_gt_file, args.workers_per_gpu)
        if args.stage == "all":
            run_stage2_static_parallel(generated_gt_path, final_output_file, config, args.workers_per_gpu)
    elif args.stage == "2":
        run_stage2_static_parallel(ground_truth_file_to_use, final_output_file, config, args.workers_per_gpu)

if __name__ == "__main__":
    # 设置多进程启动方法为 'spawn' 以确保CUDA在子进程中正确初始化
    mp.set_start_method("spawn", force=True)
    main()
