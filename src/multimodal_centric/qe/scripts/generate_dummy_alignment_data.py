# -*- coding: utf-8 -*-
"""
生成用于快速测试的模拟对齐任务基准真值文件。
"""
import json
import random
import argparse
from pathlib import Path
import os
from tqdm import tqdm

def generate_dummy_data(num_samples: int, max_phrases_per_sample: int = 5):
    """生成单个模拟数据记录。"""
    dummy_data = []
    possible_phrases = ["a red apple", "a green bottle", "the blue sky", "a wooden table", "a running shoe", "a shiny car"]
    
    for i in range(num_samples):
        num_phrases = random.randint(1, max_phrases_per_sample)
        grounding = []
        for _ in range(num_phrases):
            # 生成随机边界框 [x, y, w, h]
            x = random.randint(0, 200)
            y = random.randint(0, 200)
            w = random.randint(20, 100)
            h = random.randint(20, 100)
            grounding.append({
                "phrase": random.choice(possible_phrases),
                "box": [x, y, x + w, y + h]
            })
        
        record = {
            "node_index": i,
            "node_id": f"dummy_id_{i}",
            "image_path": f"/path/to/dummy_image_{i}.jpg",
            "text": "This is a dummy text description for the node.",
            "grounding": grounding
        }
        dummy_data.append(record)
    return dummy_data

def main():
    parser = argparse.ArgumentParser(description="Generate dummy alignment ground truth file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name to generate dummy data for (e.g., 'Grocery').")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of dummy samples to generate.")
    parser.add_argument("--output_dir", type=str, default="src/multimodal_centric/qe/evaluators/ground_truth", help="Directory to save the jsonl file.")
    args = parser.parse_args()

    print(f"Generating {args.num_samples} dummy samples for dataset '{args.dataset}'...")
    
    output_path = Path(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    # 使用与真实脚本相同的文件名，以便直接替换
    output_file = output_path / f"{args.dataset}_ground_truth.jsonl"

    dummy_records = generate_dummy_data(args.num_samples)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in tqdm(dummy_records, desc="Writing dummy data"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nProcessing completed! Dummy ground truth file saved to: {output_file}")
    print(f"Now you can run the evaluation task, which will use this dummy data for quick testing.")

if __name__ == "__main__":
    main()