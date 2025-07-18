# -*- coding: utf-8 -*-
"""
为模态对齐任务准备预处理数据。

该脚本整合了两个核心阶段：
1.  (Stage 1) 生成基准真值: 
    -   从原始数据中提取名词短语。
    -   使用GroundingDINO找到短语对应的边界框。
    -   将结果保存为 .jsonl 文件。
2.  (Stage 2) 提取并缓存特征:
    -   读取 .jsonl 真值文件。
    -   为每个 (短语, 边界框) 对提取文本嵌入和图像区域嵌入。
    -   将所有信息 (节点ID, 短语, 框, 文本嵌入, 区域嵌入) 保存到一个统一的 .pt 文件中，供评估器快速加载。

该脚本可以通过命令行参数灵活控制执行流程。
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
import yaml

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.embedding_manager import EmbeddingManager

class GroundTruthGenerator:
    """(Stage 1) 封装了生成基准真值的核心逻辑。"""
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Stage 1: 初始化 GroundTruthGenerator，使用设备: {self.device}")
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("未找到spaCy的'en_core_web_sm'模型，正在尝试下载...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        model_id = "IDEA-Research/grounding-dino-base"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.max_text_length = self.model.config.text_config.max_position_embeddings // 2

    def extract_noun_phrases(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def generate_for_single_item(self, image_path: str, text: str) -> Tuple[List[Dict[str, Any]], str]:
        if not image_path or not text: return [], "missing_path_or_text"
        if not os.path.exists(image_path): return [], "image_not_found"
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception: return [], "image_open_failed"
        phrases = self.extract_noun_phrases(text)
        if not phrases: return [], "no_noun_phrases"
        text_for_grounding = ". ".join(phrases)
        inputs = self.processor(images=image, text=text_for_grounding, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_text_length, return_attention_mask=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=0.4, text_threshold=0.25, target_sizes=[image.size[::-1]])
        grounding_results = [{"phrase": label, "box": box.cpu().numpy().tolist()} for label, box in zip(results[0]["labels"], results[0]["boxes"])]
        if not grounding_results: return [], "grounding_failed"
        return grounding_results, "success"

def run_stage1_generate_ground_truth(dataset_name: str, output_dir: Path) -> Path:
    """执行 Stage 1，生成真值文件。"""
    print("\n--- [Stage 1] 开始生成基准真值文件 ---")
    manager = EmbeddingManager()
    generator = GroundTruthGenerator()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_name}_ground_truth.jsonl"

    try:
        node_ids = manager._storage.load_node_ids(dataset_name)
        num_nodes = len(node_ids)
    except FileNotFoundError as e:
        print(f"错误: 无法加载节点ID文件: {e}")
        sys.exit(1)

    stats = defaultdict(int)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(num_nodes), desc=f"Stage 1: 处理 {dataset_name}"):
            stats["total_nodes"] += 1
            raw_data = manager.get_raw_data_by_index(dataset_name, i)
            if not raw_data or not raw_data.get("image_path") or not raw_data.get("text"):
                stats["skipped_no_raw_data"] += 1
                continue
            grounding_results, status = generator.generate_for_single_item(raw_data["image_path"], raw_data["text"])
            if status == "success":
                stats["success"] += 1
                record = {"node_index": i, "node_id": node_ids[i], "image_path": raw_data["image_path"], "text": raw_data["text"], "grounding": grounding_results}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                stats[f"failed_{status}"] += 1
    
    print("\n--- [Stage 1] 诊断统计 ---")
    print(f"总共处理节点: {stats['total_nodes']}, 成功生成真值: {stats['success']}")
    for reason, count in stats.items():
        if reason not in ["total_nodes", "success"]: print(f"  - {reason}: {count}")
    print(f"--- [Stage 1] 完成，真值文件已保存到: {output_file} ---")
    return output_file

def run_stage2_extract_features(ground_truth_file: Path, output_file: Path, config: Dict[str, Any]):
    """执行 Stage 2，提取并缓存特征。"""
    print("\n--- [Stage 2] 开始提取并缓存特征 ---")
    if not ground_truth_file.exists():
        print(f"错误: 找不到真值文件 '{ground_truth_file}'。请先运行 Stage 1 或提供正确的文件路径。")
        sys.exit(1)

    manager = EmbeddingManager()
    preprocessed_data = []
    
    # 从配置文件获取嵌入参数
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

                # 提取短语的文本嵌入
                phrase_embed = manager.generate_embedding(data=[phrase], modality="text", encoder_type=encoder_type, encoder_name=encoder_name, dimension=dimension)
                if phrase_embed is None: continue

                # 裁剪图像并提取区域嵌入
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
    parser = argparse.ArgumentParser(description="为模态对齐任务准备数据。")
    parser.add_argument("--dataset", type=str, required=True, help="要处理的数据集名称。")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "1", "generate", "2", "extract"], help="要执行的阶段。")
    parser.add_argument("--config", type=str, required=True, help="指向任务配置文件的路径 (用于Stage 2获取嵌入参数)。")
    parser.add_argument("--ground_truth_file", type=Path, default=None, help="[Stage 2] 手动指定输入的真值文件路径。")
    parser.add_argument("--output_file", type=Path, default=None, help="[Stage 2] 手动指定最终预处理数据的输出文件路径。")
    args = parser.parse_args()

    # 加载配置文件
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误: 配置文件未找到 at {args.config}")
        sys.exit(1)

    # 定义默认路径
    base_output_dir = Path(__file__).resolve().parent / "ground_truth"
    default_gt_file = base_output_dir / f"{args.dataset}_ground_truth.jsonl"
    default_output_file = base_output_dir / f"{args.dataset}_alignment_preprocessed.pt"

    ground_truth_file_to_use = args.ground_truth_file or default_gt_file
    final_output_file = args.output_file or default_output_file

    # 根据stage执行任务
    if args.stage in ["all", "1", "generate"]:
        generated_gt_path = run_stage1_generate_ground_truth(args.dataset, base_output_dir)
        if args.stage in ["all"]:
            # 在 'all' 模式下，自动将 stage 1 的输出作为 stage 2 的输入
            run_stage2_extract_features(generated_gt_path, final_output_file, config)

    elif args.stage in ["2", "extract"]:
        run_stage2_extract_features(ground_truth_file_to_use, final_output_file, config)

if __name__ == "__main__":
    main()
