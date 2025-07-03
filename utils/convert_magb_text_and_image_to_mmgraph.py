"""
MAGB 到 mm-graph 格式转换器 (增强版)

本脚本负责将 MAGB 数据集（使用.csv格式）的文本数据转换为
mm-graph 数据集所使用的 .jsonl 格式，以便统一后续的特征提取流程。

核心功能：
1. 读取 MAGB 的 .csv 文件。
2. 动态识别所有包含文本信息的列。
3. 将多个文本相关字段以 "字段名: 内容." 的格式进行结构化拼接，生成一个信息丰富的 'raw_text' 字段。
4. 将 'id' 字段映射为 'asin' 字段，以保持与 mm-graph 的一致性。
5. 将结果保存为 .jsonl 文件。
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_textual_columns(df: pd.DataFrame) -> list[str]:
    """
    动态识别DataFrame中可能包含文本信息的列。
    通过排除已知的非文本列来实现。
    """
    all_columns = df.columns.tolist()
    
    # 定义需要排除的列名或模式（不区分大小写）
    excluded_patterns = [
        'id', 'asin', 'label', 'imageURL', 'url', 'also_buy', 'also_view', 
        'text_length', 'category', 'price', 'second_category'
    ]
    
    text_cols = []
    for col in all_columns:
        is_excluded = False
        for pattern in excluded_patterns:
            if pattern.lower() in col.lower():
                is_excluded = True
                break
        if not is_excluded:
            text_cols.append(col)
            
    return text_cols

def convert_csv_to_jsonl(csv_path: Path, output_path: Path):
    """
    将MAGB的CSV文件转换为mm-graph的JSONL格式。

    Args:
        csv_path (Path): 输入的CSV文件路径。
        output_path (Path): 输出的JSONL文件路径。
    """
    if not csv_path.exists():
        logger.error(f"输入文件不存在: {csv_path}")
        return

    logger.info(f"正在读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 动态识别ID列和文本列
    id_col = 'id' if 'id' in df.columns else 'asin'
    text_cols = get_textual_columns(df)
    
    if not text_cols:
        logger.error(f"在CSV文件中找不到任何有效的文本列。")
        return

    logger.info(f"使用ID列: '{id_col}'")
    logger.info(f"将合并以下文本列: {text_cols}")

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"开始转换并写入到: {output_path}")
    
    num_records = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            # 结构化地拼接文本字段
            text_parts = []
            for col in text_cols:
                if pd.notna(row[col]) and str(row[col]).strip():
                    # 按照 "字段名: 内容." 的格式拼接
                    text_parts.append(f"{col}: {str(row[col]).strip()}.")
            
            raw_text = " ".join(text_parts)
            
            # 创建符合mm-graph格式的记录
            record = {
                "asin": str(row[id_col]),
                "raw_text": [raw_text]
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            num_records += 1
            
    logger.info(f"转换完成！共处理 {num_records} 条记录。")

def main():
    """主函数，用于命令行调用"""
    parser = argparse.ArgumentParser(description="将MAGB数据集的CSV文件转换为mm-graph的JSONL格式。")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="包含MAGB数据集的输入目录路径 (例如, ./hugging_face/Grocery)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="转换后的JSONL文件的输出目录 (默认为输入目录)"
    )
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    # 查找CSV文件 (通常与目录同名)
    csv_file = input_dir / f"{input_dir.name}.csv"
    if not csv_file.exists():
        csv_files = list(input_dir.glob("*.csv"))
        if not csv_files:
            logger.error(f"在目录 {input_dir} 中未找到任何CSV文件。")
            return
        csv_file = csv_files[0]
        logger.warning(f"未找到 {csv_file.name}, 将使用找到的第一个CSV文件: {csv_file.name}")

    # 定义输出文件路径
    output_jsonl_file = output_dir / f"{input_dir.name}-raw-text.jsonl"

    convert_csv_to_jsonl(csv_file, output_jsonl_file)

if __name__ == "__main__":
    main()