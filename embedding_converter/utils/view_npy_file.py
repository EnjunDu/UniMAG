import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def view_npy(file_path: Path, head: int = 5, tail: int = 0, all_rows: bool = False):
    """
    加载并以表格形式显示.npy文件的内容和信息。

    Args:
        file_path (Path): .npy文件的路径。
        head (int): 显示文件头部的行数。
        tail (int): 显示文件尾部的行数。
        all_rows (bool): 如果为True，则显示所有行，忽略head和tail。
    """
    if not file_path.exists():
        logger.error(f"文件不存在: {file_path}")
        return

    try:
        logger.info(f"正在加载文件: {file_path}")
        embeddings = np.load(file_path)
        
        # 1. 打印基本信息
        print("\n" + "="*50)
        print(f"文件: {file_path.name}")
        print(f"路径: {file_path.resolve()}")
        print(f"形状 (Shape): {embeddings.shape}")
        print(f"数据类型 (Dtype): {embeddings.dtype}")
        
        # 2. 打印统计信息
        if embeddings.size > 0:
            mean_val = np.mean(embeddings, dtype=np.float64)
            std_val = np.std(embeddings, dtype=np.float64)
            min_val = np.min(embeddings)
            max_val = np.max(embeddings)
            print("\n--- 统计信息 ---")
            print(f"均值 (Mean): {mean_val:.6f}")
            print(f"标准差 (Std Dev): {std_val:.6f}")
            print(f"最小值 (Min): {min_val:.6f}")
            print(f"最大值 (Max): {max_val:.6f}")
        
        # 3. 以表格形式显示数据
        print("\n--- 向量预览 ---")
        df = pd.DataFrame(embeddings)
        
        pd.set_option('display.max_rows', 200)
        pd.set_option('display.max_columns', 50)
        pd.set_option('display.width', 200)

        if all_rows:
            print(df)
        else:
            if head > 0:
                print("--- 头部 (First 5 rows) ---")
                print(df.head(head))
            if tail > 0:
                print("\n--- 尾部 (Last 5 rows) ---")
                print(df.tail(tail))

        print("="*50 + "\n")

    except Exception as e:
        logger.error(f"处理文件时发生错误: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="一个用于查看.npy文件内容的工具。")
    parser.add_argument(
        "file_path",
        type=str,
        help="要查看的.npy文件的路径。"
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="显示文件头部的行数。设置为0则不显示头部。"
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=0,
        help="显示文件尾部的行数。设置为0则不显示尾部。"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="显示所有行，此选项会覆盖 --head 和 --tail。"
    )
    args = parser.parse_args()

    view_npy(
        file_path=Path(args.file_path),
        head=args.head,
        tail=args.tail,
        all_rows=args.all
    )

if __name__ == "__main__":
    main()