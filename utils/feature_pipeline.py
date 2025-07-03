"""
特征提取管道

负责协调整个特征提取流程，从原始数据到最终的向量特征
"""

import sys
from pathlib import Path

# 将项目根目录添加到Python路径中, 以解决相对导入问题
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm import tqdm
import logging
import tarfile
from PIL import Image

from utils.storage_manager import StorageManager
from embedding_converter import EncoderFactory, ModalityType, EncoderConfig, QualityChecker
from embedding_converter.config_loader import load_embedding_config
from .convert_magb_to_mmgraph import convert_csv_to_jsonl

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeaturePipeline:
    MAGB_DATASETS = {"Grocery", "Toys", "Movies", "Reddit-S", "Reddit-M"}
    """
    特征提取管道
    
    完整的数据处理流程：
    1. 加载原始数据集
    2. 提取文本和图像特征
    3. 保存向量化特征
    4. 更新元数据
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        使用配置字典初始化特征提取管道。

        Args:
            config: 从YAML文件加载的配置字典。
        """
        self.config = config
        
        # 解析配置
        pipeline_cfg = config.get("pipeline_settings", {})
        encoder_cfg = config.get("encoder_settings", {})
        dataset_cfg = config.get("dataset_settings", {})

        self.environment = pipeline_cfg.get("environment", "local")
        self.force_reprocess = pipeline_cfg.get("force_reprocess", False)
        self.batch_size = pipeline_cfg.get("batch_size", 8)

        self.encoder_model = encoder_cfg.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        # 假设 encoder_type 可以从 model_name 推断或在配置中明确
        self.encoder_type = encoder_cfg.get("encoder_type", "qwen_vl")
        self.cache_dir = encoder_cfg.get("cache_dir")
        self.device = encoder_cfg.get("device")
        self.target_dimensions = encoder_cfg.get("target_dimensions", {})
        self.encoder_config = encoder_cfg.get("encoder_config", {})

        self.datasets_to_process = dataset_cfg.get("datasets_to_process")
        self.modalities_to_process = dataset_cfg.get("modalities_to_process", ["text", "image"])
        
        # 初始化存储管理器
        self.storage_manager = StorageManager(self.environment)
        
        # 初始化质量检查器
        self.quality_checker = QualityChecker()
        
        # 编码器实例（延迟初始化）
        self._encoders = {}
    
    def _get_encoder(self, modality: str, target_dim: Optional[int] = None):
        """获取指定模态的编码器实例"""
        encoder_key = f"{modality}_{target_dim}" if target_dim else modality
        
        if encoder_key not in self._encoders:
            logger.info(f"初始化编码器: {self.encoder_type} - {encoder_key}")
            
            # 创建编码器配置
            config = EncoderConfig(
                model_name=self.encoder_model,
                cache_dir=self.cache_dir,
                device=self.device,
                **self.encoder_config
            )
            
            # 根据编码器类型创建实例
            if target_dim is not None:
                # 需要维度变换
                if self.encoder_type == "qwen_vl":
                    encoder_type = "qwen_vl_with_dim"
                else:
                    encoder_type = f"{self.encoder_type}_with_dim"
                
                self._encoders[encoder_key] = EncoderFactory.create_encoder(
                    encoder_type=encoder_type,
                    config=config,
                    target_dimension=target_dim
                )
            else:
                # 使用原生维度
                self._encoders[encoder_key] = EncoderFactory.create_encoder(
                    encoder_type=self.encoder_type,
                    config=config
                )
        
        return self._encoders[encoder_key]
    
    def _load_dataset_metadata(self, dataset_name: str) -> Dict:
        """加载数据集元数据"""
        dataset_path = self.storage_manager.get_dataset_path(dataset_name)
        
        # 尝试读取现有的元数据文件
        metadata_files = ['metadata.json', 'dataset_info.json', 'info.json']
        for metadata_file in metadata_files:
            metadata_path = dataset_path / metadata_file
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # 如果没有元数据文件，创建基础元数据
        logger.warning(f"未找到数据集 {dataset_name} 的元数据文件，创建基础元数据")
        return {
            'dataset_name': dataset_name,
            'files': [],
            'text_files': [],
            'image_files': [],
            'graph_files': []
        }
    
    def _extract_images_from_tar(self, tar_path: Path, extract_dir: Path) -> List[str]:
        """从tar文件中提取图像"""
        if extract_dir.exists():
            logger.info(f"图像已提取到: {extract_dir}")
            return [str(f) for f in extract_dir.glob("*") if f.is_file()]
        
        logger.info(f"提取图像从 {tar_path} 到 {extract_dir}")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_dir)
        
        # 返回提取的图像文件路径
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
            image_files.extend(extract_dir.glob(ext))
        
        return [str(f) for f in image_files]
    
    def _load_text_data(self, text_file_path: Path) -> List[Dict]:
        """加载文本数据"""
        text_data = []
        
        if text_file_path.suffix == '.jsonl':
            with open(text_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    text_data.append(json.loads(line.strip()))
        elif text_file_path.suffix == '.csv':
            df = pd.read_csv(text_file_path)
            text_data = df.to_dict('records')
        else:
            logger.warning(f"不支持的文本文件格式: {text_file_path}")
        
        return text_data
    
    def run(self):
        """
        根据加载的配置运行整个特征提取流程。
        """
        logger.info("特征提取管道启动...")
        
        datasets = self.datasets_to_process
        if not datasets:
            logger.info("未在配置中指定数据集，将处理所有可用数据集。")
            datasets = self.storage_manager.list_datasets()
        
        logger.info(f"目标数据集: {datasets}")
        logger.info(f"目标模态: {self.modalities_to_process}")

        all_results = {}
        for dataset_name in datasets:
            try:
                results = self._process_single_dataset(dataset_name)
                all_results[dataset_name] = results
            except Exception as e:
                logger.error(f"处理数据集 {dataset_name} 时发生严重错误: {e}", exc_info=True)
                all_results[dataset_name] = {"error": str(e)}
        
        logger.info("所有指定的数据集处理完成。")
        logger.info(f"最终结果: \n{json.dumps(all_results, indent=2)}")
        return all_results

    def _process_single_dataset(self, dataset_name: str) -> Dict[str, str]:
        """处理单个数据集，确保所有模态基于节点ID对齐。"""
        logger.info(f"======== 开始处理数据集: {dataset_name} ========")
        dataset_path = self.storage_manager.get_dataset_path(dataset_name)
        metadata = self._load_dataset_metadata(dataset_name)
        results = {}

        # 1. 加载所有模态的ID->数据映射
        text_data_map = self._load_text_data_by_id(dataset_path)
        image_data_map = self._load_image_data_by_id(dataset_path)

        # 2. 创建权威的、排序后的节点ID列表
        all_node_ids = sorted(list(set(text_data_map.keys()) | set(image_data_map.keys())))
        if not all_node_ids:
            logger.warning(f"数据集 {dataset_name} 未找到任何文本或图像数据，跳过处理。")
            return {}
        
        # 保存这个权威的ID列表
        self._save_ordered_node_ids(dataset_path, all_node_ids)
        
        logger.info(f"找到 {len(all_node_ids)} 个唯一节点ID，将按此顺序生成特征。")

        # 3. 按顺序处理每个模态
        for modality in self.modalities_to_process:
            logger.info(f"--- 处理模态: {modality} ---")
            
            target_dim = self.target_dimensions.get(modality)
            feature_path = self.storage_manager.get_feature_path(
                dataset_name=dataset_name,
                modality=modality,
                encoder_name=self.encoder_model,
                dimension=target_dim
            )
            
            if feature_path.exists() and not self.force_reprocess:
                logger.info(f"特征文件已存在，跳过: {feature_path}")
                results[modality] = str(feature_path)
                continue
            
            if modality == "text":
                results[modality] = self._process_text_modality(
                    dataset_name, all_node_ids, text_data_map, target_dim
                )
            elif modality == "image":
                results[modality] = self._process_image_modality(
                    dataset_name, all_node_ids, image_data_map, target_dim
                )
            elif modality == "multimodal":
                results[modality] = self._process_multimodal_modality(
                    dataset_name, all_node_ids, text_data_map, image_data_map, target_dim
                )
        
        self._update_dataset_metadata(dataset_name, metadata, results)
        logger.info(f"======== 数据集 {dataset_name} 处理完成 ========")
        return results
    
    def _process_text_modality(self,
                               dataset_name: str,
                               ordered_node_ids: List[str],
                               text_data_map: Dict[str, str],
                               target_dim: Optional[int]) -> str:
        """按顺序处理文本模态。"""
        if not text_data_map:
            logger.warning(f"文本数据为空，无法处理文本模态: {dataset_name}")
            return ""

        # 按照权威顺序准备待编码的文本列表
        texts_to_encode = [text_data_map.get(node_id, "") for node_id in ordered_node_ids]
        
        # 编码文本
        encoder = self._get_encoder("text", target_dim)
        logger.info(f"编码 {len(texts_to_encode)} 个文本样本 (包括缺失值的填充)")
        
        embeddings = encoder.encode(
            ModalityType.TEXT,
            texts=texts_to_encode,
            batch_size=self.batch_size
        )
        
        # 保存特征
        feature_path = self.storage_manager.get_feature_path(
            dataset_name=dataset_name,
            modality="text",
            encoder_name=self.encoder_model,
            dimension=target_dim
        )
        self.storage_manager.save_features(embeddings, feature_path)
        
        # 进行质量检查
        logger.info("进行文本特征质量检查...")
        quality_report = self.quality_checker.check_feature_file(
            file_path=feature_path,
            expected_dim=target_dim
        )
        
        if quality_report.get("overall_quality", False):
            logger.info(f"文本特征质量检查通过: {feature_path}")
        else:
            logger.warning(f"文本特征质量检查未通过: {feature_path}")
            logger.warning(f"检查结果: {quality_report.get('reason', '未知原因')}")
        
        logger.info(f"文本特征已保存: {feature_path}")
        return str(feature_path)
    
    def _process_image_modality(self,
                               dataset_name: str,
                               ordered_node_ids: List[str],
                               image_data_map: Dict[str, Path],
                               target_dim: Optional[int]) -> str:
        """按顺序处理图像模态。"""
        if not image_data_map:
            logger.warning(f"图像数据为空，无法处理图像模态: {dataset_name}")
            return ""

        # 按照权威顺序准备待编码的图像路径列表
        # 注意：对于缺失的图像，我们传递一个空字符串作为路径
        image_paths_to_encode = [str(image_data_map.get(node_id, "")) for node_id in ordered_node_ids]
        
        # 编码图像
        encoder = self._get_encoder("image", target_dim)
        logger.info(f"编码 {len(image_paths_to_encode)} 个图像样本 (包括缺失值的填充)")
        
        embeddings = encoder.encode(
            ModalityType.IMAGE,
            image_paths=image_paths_to_encode,
            batch_size=self.batch_size
        )
        
        # 保存特征
        feature_path = self.storage_manager.get_feature_path(
            dataset_name=dataset_name,
            modality="image",
            encoder_name=self.encoder_model,
            dimension=target_dim
        )
        self.storage_manager.save_features(embeddings, feature_path)
        
        # 进行质量检查
        logger.info("进行图像特征质量检查...")
        quality_report = self.quality_checker.check_feature_file(
            file_path=feature_path,
            expected_dim=target_dim
        )
        
        if quality_report.get("overall_quality", False):
            logger.info(f"图像特征质量检查通过: {feature_path}")
        else:
            logger.warning(f"图像特征质量检查未通过: {feature_path}")
            logger.warning(f"检查结果: {quality_report.get('reason', '未知原因')}")
        
        logger.info(f"图像特征已保存: {feature_path}")
        return str(feature_path)
    
    def _process_multimodal_modality(self,
                                    dataset_name: str,
                                    ordered_node_ids: List[str],
                                    text_data_map: Dict[str, str],
                                    image_data_map: Dict[str, Path],
                                    target_dim: Optional[int]) -> str:
        """按顺序处理多模态数据。"""
        if not text_data_map and not image_data_map:
            logger.warning(f"文本和图像数据均为空，无法处理多模态模态: {dataset_name}")
            return ""

        # 按照权威顺序准备待编码的文本和图像路径列表
        texts_to_encode = [text_data_map.get(node_id, "") for node_id in ordered_node_ids]
        image_paths_to_encode = [str(image_data_map.get(node_id, "")) for node_id in ordered_node_ids]
        
        # 编码多模态数据
        encoder = self._get_encoder("multimodal", target_dim)
        logger.info(f"编码 {len(texts_to_encode)} 个多模态样本 (包括缺失值的填充)")
        
        embeddings = encoder.encode(
            ModalityType.MULTIMODAL,
            texts=texts_to_encode,
            image_paths=image_paths_to_encode,
            batch_size=self.batch_size
        )
        
        # 保存特征
        feature_path = self.storage_manager.get_feature_path(
            dataset_name=dataset_name,
            modality="multimodal",
            encoder_name=self.encoder_model,
            dimension=target_dim
        )
        self.storage_manager.save_features(embeddings, feature_path)
        
        # 进行质量检查
        logger.info("进行多模态特征质量检查...")
        quality_report = self.quality_checker.check_feature_file(
            file_path=feature_path,
            expected_dim=target_dim
        )
        
        if quality_report.get("overall_quality", False):
            logger.info(f"多模态特征质量检查通过: {feature_path}")
        else:
            logger.warning(f"多模态特征质量检查未通过: {feature_path}")
            logger.warning(f"检查结果: {quality_report.get('reason', '未知原因')}")
        
        logger.info(f"多模态特征已保存: {feature_path}")
        return str(feature_path)
    
    def _load_text_data_by_id(self, dataset_path: Path) -> Dict[str, str]:
        """加载文本数据并返回ID到文本的映射。如果需要，会自动转换MAGB数据集。"""
        dataset_name = dataset_path.name
        
        # 检查是否需要自动转换
        if dataset_name in self.MAGB_DATASETS:
            target_jsonl_path = dataset_path / f"{dataset_name}-raw-text.jsonl"
            if not target_jsonl_path.exists():
                logger.info(f"检测到MAGB数据集 '{dataset_name}' 且标准化的JSONL文件不存在，开始自动转换...")
                
                # 查找源CSV文件
                csv_path = dataset_path / f"{dataset_name}.csv"
                if not csv_path.exists():
                    logger.error(f"自动转换失败：在 {dataset_path} 中未找到源文件 {csv_path.name}")
                    return {}
                
                # 调用转换函数
                try:
                    convert_csv_to_jsonl(csv_path, target_jsonl_path)
                    logger.info(f"自动转换成功，已生成: {target_jsonl_path}")
                except Exception as e:
                    logger.error(f"自动转换过程中发生错误: {e}", exc_info=True)
                    return {}

        # 加载（可能刚刚生成的）JSONL文件
        text_files = list(dataset_path.glob("*-raw-text.jsonl"))
        if not text_files:
            logger.warning(f"在 {dataset_path} 中未找到 *-raw-text.jsonl 文件。")
            return {}
        
        text_file = text_files[0]
        logger.info(f"加载文本数据从: {text_file}")
        return self._load_text_data(text_file)

    def _load_image_data_by_id(self, dataset_path: Path) -> Dict[str, Path]:
        """加载图像数据并返回ID到图像路径的映射。"""
        image_tar_files = list(dataset_path.glob("*images.tar*"))
        if not image_tar_files:
            logger.warning(f"在 {dataset_path} 中未找到 *images.tar* 文件。")
            return {}
            
        tar_file = image_tar_files[0]
        extract_dir = dataset_path / f"{tar_file.stem}_extracted"
        return self._extract_images_from_tar(tar_file, extract_dir)

    def _save_ordered_node_ids(self, dataset_path: Path, ordered_node_ids: List[str]):
        """将排序后的节点ID列表保存到文件。"""
        id_file_path = dataset_path / "node_ids.json"
        with open(id_file_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_node_ids, f)
        logger.info(f"权威节点ID列表已保存到: {id_file_path}")

    def _update_dataset_metadata(self,
                                dataset_name: str,
                                metadata: Dict,
                                processing_results: Dict[str, str]) -> None:
        """更新数据集元数据"""
        if 'processed_features' not in metadata:
            metadata['processed_features'] = {}
        
        # 使用更详细的键来避免覆盖
        encoder_key = self.encoder_model.replace('/', '_')
        dim_key = "_".join([f"{m}{d}" for m, d in self.target_dimensions.items()])
        feature_key = f"{encoder_key}_{dim_key}"
        
        metadata['processed_features'][feature_key] = processing_results
        metadata['node_id_file'] = "node_ids.json"
        
        # 保存元数据
        dataset_path = self.storage_manager.get_dataset_path(dataset_name)
        metadata_path = dataset_path / 'processing_metadata.json'
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"元数据已更新: {metadata_path}")


def main():
    """主函数，用于通过配置文件运行特征提取管道"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="通过配置文件运行特征提取管道。")
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="指向YAML配置文件的路径。",
        default="configs/embedding/qwen_vl_default.yaml"
    )
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_embedding_config(args.config)
        
        # 创建并运行管道
        pipeline = FeaturePipeline(config)
        pipeline.run()
        
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.error(f"管道运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()