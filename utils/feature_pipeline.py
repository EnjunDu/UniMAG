"""
特征提取管道

负责协调整个特征提取流程，从原始数据到最终的向量特征
"""

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

from .storage_manager import StorageManager
from embedding_converter import EncoderFactory, ModalityType, EncoderConfig, QualityChecker
from embedding_converter.config_loader import load_embedding_config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeaturePipeline:
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
        """处理单个数据集"""
        logger.info(f"======== 开始处理数据集: {dataset_name} ========")
        
        dataset_path = self.storage_manager.get_dataset_path(dataset_name)
        metadata = self._load_dataset_metadata(dataset_name)
        results = {}
        
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
                    dataset_name, dataset_path, metadata, target_dim
                )
            elif modality == "image":
                results[modality] = self._process_image_modality(
                    dataset_name, dataset_path, metadata, target_dim
                )
            elif modality == "multimodal":
                results[modality] = self._process_multimodal_modality(
                    dataset_name, dataset_path, metadata, target_dim
                )
        
        self._update_dataset_metadata(dataset_name, metadata, results)
        logger.info(f"======== 数据集 {dataset_name} 处理完成 ========")
        return results
    
    def _process_text_modality(self,
                              dataset_name: str,
                              dataset_path: Path,
                              metadata: Dict,
                              target_dim: Optional[int]) -> str:
        """处理文本模态"""
        # 查找文本数据文件
        text_files = []
        for pattern in ['*text*.jsonl', '*.csv', '*raw-text*']:
            text_files.extend(dataset_path.glob(pattern))
        
        if not text_files:
            logger.warning(f"未找到文本文件: {dataset_name}")
            return ""
        
        # 使用第一个找到的文本文件
        text_file = text_files[0]
        logger.info(f"处理文本文件: {text_file}")
        
        # 加载文本数据
        text_data = self._load_text_data(text_file)
        if not text_data:
            logger.warning(f"文本数据为空: {text_file}")
            return ""
        
        # 提取文本内容
        texts = []
        for item in text_data:
            # 尝试不同的文本字段名
            text_content = ""
            for field in ['text', 'title', 'description', 'content']:
                if field in item and item[field]:
                    text_content += str(item[field]) + " "
            texts.append(text_content.strip())
        
        # 编码文本
        encoder = self._get_encoder("text", target_dim)
        logger.info(f"编码 {len(texts)} 个文本样本")
        
        embeddings = encoder.encode(
            ModalityType.TEXT,
            texts=texts,
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
                               dataset_path: Path,
                               metadata: Dict,
                               target_dim: Optional[int]) -> str:
        """处理图像模态"""
        # 查找图像文件
        image_tar_files = list(dataset_path.glob("*images*.tar*"))
        image_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and "image" in d.name.lower()]
        
        image_paths = []
        
        # 处理tar文件
        for tar_file in image_tar_files:
            extract_dir = dataset_path / f"{tar_file.stem}_extracted"
            extracted_images = self._extract_images_from_tar(tar_file, extract_dir)
            image_paths.extend(extracted_images)
        
        # 处理图像目录
        for img_dir in image_dirs:
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
                image_paths.extend([str(f) for f in img_dir.glob(ext)])
        
        if not image_paths:
            logger.warning(f"未找到图像文件: {dataset_name}")
            return ""
        
        logger.info(f"找到 {len(image_paths)} 个图像文件")
        
        # 编码图像
        encoder = self._get_encoder("image", target_dim)
        
        embeddings = encoder.encode(
            ModalityType.IMAGE,
            image_paths=image_paths,
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
                                    dataset_path: Path,
                                    metadata: Dict,
                                    target_dim: Optional[int]) -> str:
        """处理多模态数据"""
        # 这里需要同时加载文本和图像数据，并确保它们的对应关系
        logger.info("处理多模态数据")
        
        # 加载文本数据
        text_files = []
        for pattern in ['*text*.jsonl', '*.csv']:
            text_files.extend(dataset_path.glob(pattern))
        
        if not text_files:
            logger.warning(f"多模态处理需要文本数据: {dataset_name}")
            return ""
        
        text_data = self._load_text_data(text_files[0])
        
        # 提取图像路径（这里假设有node_mapping或类似的映射关系）
        image_tar_files = list(dataset_path.glob("*images*.tar*"))
        if not image_tar_files:
            logger.warning(f"多模态处理需要图像数据: {dataset_name}")
            return ""
        
        # 提取图像
        extract_dir = dataset_path / f"{image_tar_files[0].stem}_extracted"
        extracted_images = self._extract_images_from_tar(image_tar_files[0], extract_dir)
        
        # 创建文本-图像对应关系（简化版本，实际可能需要更复杂的映射）
        texts = []
        image_paths = []
        
        min_length = min(len(text_data), len(extracted_images))
        for i in range(min_length):
            # 提取文本
            text_content = ""
            for field in ['text', 'title', 'description']:
                if field in text_data[i]:
                    text_content += str(text_data[i][field]) + " "
            texts.append(text_content.strip())
            image_paths.append(extracted_images[i])
        
        # 编码多模态数据
        encoder = self._get_encoder("multimodal", target_dim)
        
        embeddings = encoder.encode(
            ModalityType.MULTIMODAL,
            texts=texts,
            image_paths=image_paths,
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
    
    def _update_dataset_metadata(self,
                                dataset_name: str,
                                metadata: Dict,
                                processing_results: Dict[str, str]) -> None:
        """更新数据集元数据"""
        # 添加处理信息
        if 'processed_features' not in metadata:
            metadata['processed_features'] = {}
        
        # 使用更详细的键来避免覆盖
        feature_key = f"{self.encoder_model}_{self.target_dimensions.get('text', 'native')}d"
        metadata['processed_features'][feature_key] = processing_results
        
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