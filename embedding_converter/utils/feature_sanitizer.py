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
    
    def sanitize_file(self, file_path: Union[str, Path], replacement_value: float = 0.0) -> bool:
        """
        清理单个特征文件中的NaN和Inf值。
        
        Args:
            file_path: 特征文件的路径。
            replacement_value: 用于替换NaN和Inf的值，默认为0。
            
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
            
            # 检查是否存在NaN或Inf值
            if not np.any(np.isnan(embeddings)) and not np.any(np.isinf(embeddings)):
                logger.debug(f"文件 {file_path} 无需清理。")
                return False
            
            # 使用np.nan_to_num替换NaN, inf, -inf
            # nan -> 0.0 (或指定值)
            # inf -> 正的最大浮点数 (或指定值)
            # -inf -> 负的最小浮点数 (或指定值)
            # 为了统一替换为0，我们先用nan_to_num处理nan，再手动处理inf
            sanitized_embeddings = np.nan_to_num(
                embeddings, 
                nan=replacement_value,
                posinf=replacement_value,
                neginf=replacement_value
            )
            
            # 覆盖保存文件
            np.save(file_path, sanitized_embeddings)
            logger.info(f"文件已清理并保存: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"清理文件 {file_path} 时发生错误: {e}", exc_info=True)
            return False

    def sanitize_directory(self, directory_path: Union[str, Path], replacement_value: float = 0.0):
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
    parser = argparse.ArgumentParser(description="清理.npy特征文件中的NaN和Inf值。")
    parser.add_argument(
        "path",
        type=str,
        help="要清理的.npy文件或包含.npy文件的目录的路径。"
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.0,
        help="用于替换NaN和Inf值的数值，默认为0.0。"
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