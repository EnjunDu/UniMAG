import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings

logger = logging.getLogger(__name__)

class QualityChecker:
    """
    特征质量检查器
    
    提供多种质量验证方法，确保特征向量的完整性和数值稳定性
    """
    
    def __init__(self, 
                 nan_threshold: float = 0.01,
                 inf_threshold: float = 0.01,
                 zero_threshold: float = 0.95): # 允许多模态的零值比例更高
        """
        初始化质量检查器
        
        Args:
            nan_threshold: NaN值比例阈值
            inf_threshold: Inf值比例阈值 
            zero_threshold: 零值比例阈值
        """
        self.nan_threshold = nan_threshold
        self.inf_threshold = inf_threshold
        self.zero_threshold = zero_threshold
        
    def check_completeness(self, embeddings: np.ndarray, name: str = "embeddings") -> Dict[str, Union[bool, float, str]]:
        """
        检查嵌入向量的完整性
        
        Args:
            embeddings: 嵌入矩阵 [num_samples, embedding_dim]
            name: 检查项的名称
            
        Returns:
            检查结果字典
        """
        if embeddings.size == 0:
            logger.warning(f"{name}: 嵌入矩阵为空")
            return {
                "is_complete": False,
                "nan_ratio": 0.0,
                "inf_ratio": 0.0,
                "zero_ratio": 0.0,
                "reason": "空矩阵"
            }
        
        total_elements = embeddings.size
        
        # 检查NaN值
        nan_count = np.isnan(embeddings).sum()
        nan_ratio = nan_count / total_elements
        
        # 检查Inf值
        inf_count = np.isinf(embeddings).sum()
        inf_ratio = inf_count / total_elements
        
        # 检查零值比例
        zero_count = (embeddings == 0).sum()
        zero_ratio = zero_count / total_elements
        
        # 判断是否完整
        is_complete = (
            nan_ratio <= self.nan_threshold and
            inf_ratio <= self.inf_threshold and
            zero_ratio <= self.zero_threshold
        )
        
        result = {
            "is_complete": is_complete,
            "nan_ratio": nan_ratio,
            "inf_ratio": inf_ratio,
            "zero_ratio": zero_ratio
        }
        
        # 记录检查结果
        if is_complete:
            logger.info(f"{name}: 完整性检查通过")
        else:
            reasons = []
            if nan_ratio > self.nan_threshold:
                reasons.append(f"NaN比例过高: {nan_ratio:.4f}")
            if inf_ratio > self.inf_threshold:
                reasons.append(f"Inf比例过高: {inf_ratio:.4f}")
            if zero_ratio > self.zero_threshold:
                reasons.append(f"零值比例过高: {zero_ratio:.4f}")
            
            result["reason"] = "; ".join(reasons)
            logger.warning(f"{name}: 完整性检查失败 - {result['reason']}")
        
        return result
    
    def check_dimensions(self,
                        embeddings: np.ndarray,
                        expected_dim: int,
                        name: str = "embeddings") -> Dict[str, Union[bool, int, str, None]]:
        """
        检查嵌入维度是否一致
        
        Args:
            embeddings: 嵌入矩阵 [num_samples, embedding_dim]
            expected_dim: 期望的嵌入维度
            name: 检查项的名称
            
        Returns:
            维度检查结果
        """
        if embeddings.ndim != 2:
            logger.error(f"{name}: 嵌入矩阵应为2维，当前为{embeddings.ndim}维")
            return {
                "dimension_correct": False,
                "expected_dim": expected_dim,
                "actual_dim": None,
                "reason": f"矩阵维度错误: {embeddings.ndim}D"
            }
        
        actual_dim = embeddings.shape[1]
        is_correct = actual_dim == expected_dim
        
        result = {
            "dimension_correct": is_correct,
            "expected_dim": expected_dim,
            "actual_dim": actual_dim
        }
        
        if is_correct:
            logger.info(f"{name}: 维度检查通过 ({actual_dim})")
        else:
            result["reason"] = f"维度不匹配: 期望{expected_dim}, 实际{actual_dim}"
            logger.warning(f"{name}: {result['reason']}")
        
        return result
    
    def check_distribution(self, 
                          embeddings: np.ndarray,
                          name: str = "embeddings") -> Dict[str, float]:
        """
        检查嵌入向量的分布统计
        
        Args:
            embeddings: 嵌入矩阵 [num_samples, embedding_dim]
            name: 检查项的名称
            
        Returns:
            分布统计信息
        """
        if embeddings.size == 0:
            logger.warning(f"{name}: 无法计算空矩阵的分布统计")
            return {}
        
        # 使用 with warnings.catch_warnings() 来优雅地处理计算中可能出现的警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            # 过滤掉NaN和Inf值
            finite_mask = np.isfinite(embeddings)
            finite_embeddings = embeddings[finite_mask]
            
            if finite_embeddings.size == 0:
                logger.warning(f"{name}: 没有有效的数值进行统计")
                return {
                    "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                    "median": 0.0, "finite_ratio": 0.0
                }
            
            # 在进行统计计算时，使用更高精度的float64来防止溢出
            stats = {
                "mean": float(np.mean(finite_embeddings, dtype=np.float64)),
                "std": float(np.std(finite_embeddings, dtype=np.float64)),
                "min": float(np.min(finite_embeddings)),
                "max": float(np.max(finite_embeddings)),
                "median": float(np.median(finite_embeddings)),
                "finite_ratio": finite_embeddings.size / embeddings.size
            }
        
        logger.info(f"{name}: 分布统计 - "
                   f"均值: {stats['mean']:.4f}, "
                   f"标准差: {stats['std']:.4f}, "
                   f"范围: [{stats['min']:.4f}, {stats['max']:.4f}], "
                   f"有效比例: {stats['finite_ratio']:.4f}")
        
        return stats
    
    def check_feature_file(self,
                          file_path: Union[str, Path],
                          expected_dim: Optional[int] = None) -> Dict[str, Union[bool, Dict, str, Tuple, Path]]:
        """
        检查特征文件的质量
        
        Args:
            file_path: 特征文件路径
            expected_dim: 期望的嵌入维度
            
        Returns:
            完整的质量检查报告
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"特征文件不存在: {file_path}")
            return {
                "file_exists": False,
                "overall_quality": False,
                "reason": "文件不存在"
            }
        
        try:
            # 加载特征文件
            embeddings = np.load(file_path)
            logger.info(f"加载特征文件: {file_path}, 形状: {embeddings.shape}")
            
        except Exception as e:
            logger.error(f"无法加载特征文件 {file_path}: {e}")
            return {
                "file_exists": True,
                "overall_quality": False,
                "reason": f"文件加载失败: {e}"
            }
        
        # 执行各项检查
        report = {
            "file_exists": True,
            "file_path": str(file_path),
            "shape": embeddings.shape,
            "completeness": self.check_completeness(embeddings, file_path.name),
            "distribution": self.check_distribution(embeddings, file_path.name)
        }
        
        # 检查维度（如果提供了期望维度）
        if expected_dim is not None:
            report["dimensions"] = self.check_dimensions(embeddings, expected_dim, file_path.name)
        
        # 评估整体质量
        overall_quality = report["completeness"]["is_complete"]
        if "dimensions" in report and report["dimensions"] is not None:
            overall_quality = overall_quality and report["dimensions"]["dimension_correct"]
        
        report["overall_quality"] = overall_quality
        
        return report
    
    def check_batch_files(self, 
                         file_paths: List[Union[str, Path]],
                         expected_dims: Optional[Dict[str, int]] = None) -> Dict[str, Dict]:
        """
        批量检查多个特征文件
        
        Args:
            file_paths: 特征文件路径列表
            expected_dims: 期望维度字典，键为文件名模式，值为维度
            
        Returns:
            批量检查报告
        """
        logger.info(f"开始批量检查 {len(file_paths)} 个特征文件")
        
        reports = {}
        passed_count = 0
        
        for file_path in file_paths:
            file_path = Path(file_path)
            
            # 确定期望维度
            expected_dim = None
            if expected_dims:
                for pattern, dim in expected_dims.items():
                    if pattern in file_path.name:
                        expected_dim = dim
                        break
            
            # 执行检查
            report = self.check_feature_file(file_path, expected_dim)
            reports[str(file_path)] = report
            
            if report.get("overall_quality", False):
                passed_count += 1
        
        # 生成汇总报告
        summary = {
            "total_files": len(file_paths),
            "passed_files": passed_count,
            "failed_files": len(file_paths) - passed_count,
            "pass_rate": passed_count / len(file_paths) if file_paths else 0.0,
            "individual_reports": reports
        }
        
        logger.info(f"批量检查完成: {passed_count}/{len(file_paths)} 文件通过质量检查 "
                   f"(通过率: {summary['pass_rate']:.2%})")
        
        return summary

def create_quality_checker(config: Optional[Dict] = None) -> QualityChecker:
    """
    便捷函数：创建质量检查器实例
    
    Args:
        config: 配置字典
        
    Returns:
        质量检查器实例
    """
    if config is None:
        config = {}
    
    return QualityChecker(**config)

if __name__ == "__main__":
    # 测试质量检查器
    checker = QualityChecker()
    
    # 创建测试数据
    test_embeddings = np.random.randn(100, 768).astype(np.float32)
    
    # 添加一些质量问题
    test_embeddings[0, 0] = np.nan  # 添加NaN
    test_embeddings[1, 1] = np.inf  # 添加Inf
    test_embeddings[2:5, :] = 0     # 添加零值
    test_embeddings[6, :] = 1e30    # 添加可能导致溢出的值
    
    print("=== 质量检查器测试 ===")
    
    # 测试完整性检查
    completeness = checker.check_completeness(test_embeddings, "测试数据")
    print(f"完整性检查: {completeness}")
    
    # 测试维度检查
    dimensions = checker.check_dimensions(test_embeddings, 768, "测试数据")
    print(f"维度检查: {dimensions}")
    
    # 测试分布统计
    distribution = checker.check_distribution(test_embeddings, "测试数据")
    print(f"分布统计: {distribution}")