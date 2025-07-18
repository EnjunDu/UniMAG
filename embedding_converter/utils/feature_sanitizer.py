import numpy as np
import logging
from pathlib import Path
from typing import Union
import argparse
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureSanitizer:
    """
    特征清理器
    
    用于清理特征向量文件中的NaN和Inf值。
    """
    
    def sanitize_file(self, file_path: Union[str, Path], replacement_value: float = 1.0) -> bool:
        """
        清理单个特征文件中的NaN、Inf值和全零向量。
        
        Args:
            file_path: 特征文件的路径。
            replacement_value: 用于替换NaN/Inf以及作为全零向量的替代值，默认为1.0。
            
        Returns:
            如果文件被修改则返回True，否则返回False。
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在，无法清理: {file_path}")
            return False
            
        try:
            # 加载特征文件
            embeddings = np.load(file_path)
            original_embeddings = embeddings.copy()

            # 1. 替换NaN和Inf值
            sanitized_embeddings = np.nan_to_num(
                embeddings,
                nan=replacement_value,
                posinf=replacement_value,
                neginf=replacement_value
            )
            
            # 2. 替换全零向量
            # 在替换了nan/inf之后，检查是否存在全零行
            zero_rows_mask = np.all(sanitized_embeddings == 0, axis=1)
            if np.any(zero_rows_mask):
                num_zero_vectors = np.sum(zero_rows_mask)
                logger.info(f"在 {file_path} 中发现并替换 {num_zero_vectors} 个全零向量。")
                # 创建一个与行形状相同的全`replacement_value`数组
                replacement_vectors = np.full_like(sanitized_embeddings[zero_rows_mask], replacement_value)
                sanitized_embeddings[zero_rows_mask] = replacement_vectors

            # 检查是否有任何更改
            if np.array_equal(original_embeddings, sanitized_embeddings):
                logger.debug(f"文件 {file_path} 无需清理。")
                return False
            
            # 覆盖保存文件
            np.save(file_path, sanitized_embeddings)
            logger.info(f"文件已清理并保存: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"清理文件 {file_path} 时发生错误: {e}", exc_info=True)
            return False

    def sanitize_directory(self, directory_path: Union[str, Path], replacement_value: float = 1.0):
        """
        递归清理指定目录下的所有.npy文件。
        
        Args:
            directory_path: 目标目录的路径。
            replacement_value: 用于替换NaN和Inf的值。
        """
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            logger.error(f"提供的路径不是一个目录: {directory_path}")
            return
            
        logger.info(f"开始递归扫描并清理目录: {directory_path}")
        
        # 查找所有.npy文件
        npy_files = list(directory_path.rglob("*.npy"))
        
        if not npy_files:
            logger.warning(f"在目录 {directory_path} 中未找到任何.npy文件。")
            return
            
        modified_count = 0
        for file_path in tqdm(npy_files, desc=f"清理 {directory_path}"):
            if self.sanitize_file(file_path, replacement_value):
                modified_count += 1
        
        logger.info(f"目录清理完成。共检查 {len(npy_files)} 个文件，其中 {modified_count} 个文件被修改。")

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="清理.npy特征文件中的NaN、Inf值和全零向量。")
    parser.add_argument(
        "path",
        type=str,
        help="要清理的.npy文件或包含.npy文件的目录的路径。"
    )
    parser.add_argument(
        "--value",
        type=float,
        default=1.0,
        help="用于替换NaN/Inf和全零向量的数值，默认为1.0。"
    )
    args = parser.parse_args()
    
    sanitizer = FeatureSanitizer()
    target_path = Path(args.path)
    
    if target_path.is_file() and target_path.suffix == '.npy':
        sanitizer.sanitize_file(target_path, args.value)
    elif target_path.is_dir():
        sanitizer.sanitize_directory(target_path, args.value)
    else:
        logger.error(f"路径无效: {target_path}. 请提供一个.npy文件或一个目录。")

if __name__ == "__main__":
    main()