import os
import json
import random
import shutil
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_image_file(image_dir: Path, node_id: str) -> Optional[Path]:
    """根据节点ID在目录中查找对应的图像文件（忽略扩展名）"""
    # 使用glob查找所有可能匹配的文件
    matched_files = list(image_dir.glob(f"{node_id}.*"))
    for file in matched_files:
        if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            return file
    return None

def create_sample_dataset(
    source_dataset_name: str,
    output_dataset_name: str,
    sample_size: int,
    base_path: str = "hugging_face"
):
    """
    从源数据集中创建指定大小的随机样本数据集。

    Args:
        source_dataset_name (str): 源数据集的名称 (例如 "books-nc")。
        output_dataset_name (str): 输出样本数据集的名称 (例如 "books-nc-50")。
        sample_size (int): 要抽取的样本数量。
        base_path (str): 包含数据集的根目录。
    """
    logger.info(f"开始创建样本数据集 '{output_dataset_name}'，大小为 {sample_size}...")

    # 1. 定义路径
    source_dir = Path(base_path) / source_dataset_name
    output_dir = Path(base_path) / output_dataset_name
    
    source_text_file = source_dir / f"{source_dataset_name}-raw-text.jsonl"
    source_image_dir_name = f"{source_dataset_name}-images_extracted"
    source_image_dir = source_dir / source_image_dir_name

    if not source_dir.exists() or not source_text_file.exists():
        logger.error(f"源数据集路径或文件不存在: {source_dir} / {source_text_file}")
        return

    # 2. 创建输出目录
    if output_dir.exists():
        logger.warning(f"输出目录 '{output_dir}' 已存在，将覆盖内容。")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    output_image_dir = output_dir / f"{output_dataset_name}-images_extracted"
    output_image_dir.mkdir()

    # 3. 读取并随机抽样文本数据
    logger.info(f"从 '{source_text_file}' 读取所有文本行...")
    with open(source_text_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    if len(all_lines) < sample_size:
        logger.warning(f"数据总行数 ({len(all_lines)}) 小于请求的样本数 ({sample_size})。将使用所有数据。")
        sample_size = len(all_lines)
    
    sampled_lines = random.sample(all_lines, sample_size)
    logger.info(f"已随机抽取 {len(sampled_lines)} 个文本样本。")

    # 4. 处理样本，写入新文件并复制图像
    output_text_file = output_dir / f"{output_dataset_name}-raw-text.jsonl"
    copied_images = 0
    
    with open(output_text_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(sampled_lines, desc=f"处理样本并复制图像"):
            try:
                record = json.loads(line.strip())
                # 写入新的jsonl文件
                f_out.write(line)

                # 获取ID并复制图像
                node_id = str(record.get("id") or record.get("asin"))
                if node_id:
                    source_image_path = find_image_file(source_image_dir, node_id)
                    if source_image_path and source_image_path.exists():
                        target_image_path = output_image_dir / source_image_path.name
                        shutil.copy(source_image_path, target_image_path)
                        copied_images += 1
                    else:
                        logger.warning(f"未找到节点ID '{node_id}' 对应的图像文件。")

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"跳过格式错误的行: {line.strip()} - {e}")

    logger.info("样本数据集创建完成！")
    logger.info(f"总共处理了 {len(sampled_lines)} 个文本记录。")
    logger.info(f"成功复制了 {copied_images} 个图像文件。")
    logger.info(f"新的文本文件位于: {output_text_file}")
    logger.info(f"新的图像目录位于: {output_image_dir}")


def main():
    parser = argparse.ArgumentParser(description="从一个大数据集创建一个小的随机样本数据集。")
    parser.add_argument(
        "--source_dataset_name",
        type=str,
        required=True,
        help="源数据集的目录名 (例如 'books-nc')."
    )
    parser.add_argument(
        "--output_dataset_name",
        type=str,
        required=True,
        help="要创建的样本数据集的目录名 (例如 'books-nc-50')."
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="要抽取的随机样本数量。"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="hugging_face",
        help="包含所有数据集的根目录。"
    )
    args = parser.parse_args()

    create_sample_dataset(
        source_dataset_name=args.source_dataset_name,
        output_dataset_name=args.output_dataset_name,
        sample_size=args.sample_size,
        base_path=args.base_path
    )

if __name__ == "__main__":
    main()