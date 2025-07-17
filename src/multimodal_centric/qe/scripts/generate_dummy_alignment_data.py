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
    parser = argparse.ArgumentParser(description="生成模拟的对齐基准真值文件。")
    parser.add_argument("--dataset", type=str, required=True, help="要为其生成模拟数据的数据集名称 (例如 'Grocery')。")
    parser.add_argument("--num_samples", type=int, default=100, help="要生成的模拟样本数量。")
    parser.add_argument("--output_dir", type=str, default="src/multimodal_centric/qe/evaluators/ground_truth", help="输出jsonl文件的目录。")
    args = parser.parse_args()

    print(f"开始为数据集 '{args.dataset}' 生成 {args.num_samples} 条模拟数据...")
    
    output_path = Path(args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    # 使用与真实脚本相同的文件名，以便直接替换
    output_file = output_path / f"{args.dataset}_ground_truth.jsonl"

    dummy_records = generate_dummy_data(args.num_samples)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in tqdm(dummy_records, desc="写入模拟数据"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n处理完成！模拟的基准真值文件已保存到: {output_file}")
    print(f"现在，您可以运行评估任务，它将使用这份模拟数据进行快速测试。")

if __name__ == "__main__":
    main()