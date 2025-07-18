# -*- coding: utf-8 -*-
"""
生成模态对齐任务的基准真值文件。

该脚本负责执行耗时的预处理步骤：
1.  遍历指定数据集的每一个节点。
2.  为每个节点提取名词短语 (text phrases)。
3.  使用GroundingDINO模型为每个短语在对应的图像中定位边界框 (bounding boxes)。
4.  将结果保存为一个jsonl文件，供AlignmentEvaluator后续使用。
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

# 将项目根目录添加到 sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.embedding_manager import EmbeddingManager

class GroundTruthGenerator:
    """封装了生成基准真值的核心逻辑。"""
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"初始化 GroundTruthGenerator，使用设备: {self.device}")
        
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
        """从给定文本中提取名词短语。"""
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def generate_for_single_item(self, image_path: str, text: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        为单个图文对生成定位结果。
        返回一个元组: (定位结果, 失败原因)
        """
        if not image_path or not text:
            return [], "missing_path_or_text"
        if not os.path.exists(image_path):
            return [], "image_not_found"
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            return [], "image_open_failed"

        phrases = self.extract_noun_phrases(text)
        if not phrases:
            return [], "no_noun_phrases"

        text_for_grounding = ". ".join(phrases)
        
        inputs = self.processor(
            images=image, 
            text=text_for_grounding, 
            return_tensors="pt",
            padding="max_length", 
            truncation=True, 
            max_length=self.max_text_length,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        
        grounding_results = []
        for label, box in zip(results[0]["labels"], results[0]["boxes"]):
            grounding_results.append({"phrase": label, "box": box.cpu().numpy().tolist()})
        
        if not grounding_results:
            return [], "grounding_failed"

        return grounding_results, "success"

def main():
    parser = argparse.ArgumentParser(description="为模态对齐任务生成基准真值文件。")
    parser.add_argument("--dataset", type=str, required=True, help="要处理的数据集名称 (例如 'books-nc-50')。")
    parser.add_argument("--output_dir", type=str, default="src/multimodal_centric/qe/evaluators/ground_truth", help="输出jsonl文件的目录。")
    args = parser.parse_args()

    print(f"开始为数据集 '{args.dataset}' 生成基准真值...")
    
    manager = EmbeddingManager()
    generator = GroundTruthGenerator()
    
    output_path = Path(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    output_file = output_path / f"{args.dataset}_ground_truth.jsonl"

    try:
        node_ids = manager._storage.load_node_ids(args.dataset)
        num_nodes = len(node_ids)
        print(f"共找到 {num_nodes} 个节点。")
    except FileNotFoundError as e:
        print(f"错误: 无法加载节点ID文件: {e}")
        sys.exit(1)

    stats = defaultdict(int)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in tqdm(range(num_nodes), desc=f"处理 {args.dataset}"):
            stats["total_nodes"] += 1
            raw_data = manager.get_raw_data_by_index(args.dataset, i)
            
            if not raw_data or not raw_data.get("image_path") or not raw_data.get("text"):
                stats["skipped_no_raw_data"] += 1
                continue

            grounding_results, status = generator.generate_for_single_item(raw_data["image_path"], raw_data["text"])
            
            if status == "success":
                stats["success"] += 1
                record = {
                    "node_index": i,
                    "node_id": node_ids[i],
                    "image_path": raw_data["image_path"],
                    "text": raw_data["text"],
                    "grounding": grounding_results
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                stats[f"failed_{status}"] += 1
    
    print("\n--- 诊断统计 ---")
    print(f"总共处理节点: {stats['total_nodes']}")
    print(f"成功生成真值: {stats['success']}")
    print("失败/跳过原因:")
    for reason, count in stats.items():
        if reason not in ["total_nodes", "success"]:
            print(f"  - {reason}: {count}")
    print(f"------------------")
    print(f"\n处理完成！基准真值文件已保存到: {output_file}")

if __name__ == "__main__":
    main()