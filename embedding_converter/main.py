"""
特征提取管道主程序

负责协调整个特征提取流程，从原始数据到最终的向量特征。
这是嵌入转换器子模块的执行入口。
"""

import sys
from pathlib import Path
import torch
import json
import numpy as np
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import logging
import tarfile

# 将项目根目录添加到Python路径中, 以解决相对导入问题
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# 导入此子模块内的组件
from embedding_converter.encoder_factory import EncoderFactory
from embedding_converter.base_encoder import BaseEncoder, ModalityType
from embedding_converter.utils.quality_checker import QualityChecker
from embedding_converter.utils.config_loader import load_embedding_config
from embedding_converter.utils.storage_manager import StorageManager
from embedding_converter.utils.convert_magb_text_to_mmgraph import convert_csv_to_jsonl

# 导入此包以触发所有编码器的自动注册
from embedding_converter import encoders

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 性能优化：在支持的硬件（如Ampere）上启用TF32
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    logger.info("检测到安培或更新架构的GPU，启用TF32以加速矩阵运算。")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class FeaturePipeline:
    MAGB_DATASETS = {"Grocery", "Toys", "Movies", "RedditS", "RedditM"}
    
    def __init__(self, config: Dict[str, Any]):
        """使用配置字典初始化特征提取管道。"""
        self.config = config
        
        pipeline_cfg = config.get("pipeline_settings", {})
        encoder_cfg = config.get("encoder_settings", {})
        dataset_cfg = config.get("dataset_settings", {})

        self.force_reprocess = pipeline_cfg.get("force_reprocess", False)
        self.batch_size = pipeline_cfg.get("batch_size", 8)

        self.encoder_type = encoder_cfg.get("encoder_type")
        self.encoder_model = encoder_cfg.get("model_name")
        self.device = encoder_cfg.get("device")
        self.target_dimensions = encoder_cfg.get("target_dimensions", {})
        
        self.dataset_root_path = dataset_cfg.get("dataset_root_path")
        self.datasets_to_process = dataset_cfg.get("datasets_to_process")
        self.modalities_to_process = dataset_cfg.get("modalities_to_process", ["text", "image"])
        
        self.storage_manager = StorageManager(self.dataset_root_path)
        
        self.cache_dir = encoder_cfg.get("cache_dir")
        
        self.quality_checker = QualityChecker()
        self._encoders: Dict[str, BaseEncoder] = {}

    def _get_encoder(self, modality: str, target_dim: Optional[int] = None) -> BaseEncoder:
        if not self.encoder_type or not self.encoder_model:
            raise ValueError("配置文件中必须提供 'encoder_type' 和 'model_name'")

        base_encoder_type = self.encoder_type
        
        if target_dim is not None:
            encoder_name_to_create = f"{base_encoder_type}_with_dim"
            instance_key = f"{encoder_name_to_create}_{self.encoder_model}_{target_dim}"
            constructor_kwargs = {'target_dimension': target_dim}
        else:
            encoder_name_to_create = base_encoder_type
            instance_key = f"{base_encoder_type}_{self.encoder_model}_native"
            constructor_kwargs = {}

        if instance_key not in self._encoders:
            logger.info(f"缓存中未找到实例 '{instance_key}'，将从工厂创建 '{encoder_name_to_create}'。")
            full_kwargs = {
                'model_name': self.encoder_model,
                'cache_dir': self.cache_dir,
                'device': self.device,
                **constructor_kwargs
            }
            self._encoders[instance_key] = EncoderFactory.create_encoder(name=encoder_name_to_create, **full_kwargs)
        
        return self._encoders[instance_key]

    def _load_dataset_metadata(self, dataset_name: str) -> Dict:
        dataset_path = self.storage_manager.get_dataset_path(dataset_name)
        metadata_files = ['metadata.json', 'dataset_info.json', 'info.json']
        for metadata_file in metadata_files:
            metadata_path = dataset_path / metadata_file
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        logger.warning(f"未找到数据集 {dataset_name} 的元数据文件，创建基础元数据")
        return {'dataset_name': dataset_name}

    def _scan_image_directory(self, image_dir: Path) -> Dict[str, Path]:
        image_path_map = {}
        logger.info(f"正在递归扫描图像目录: {image_dir}")
        for image_path in image_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                image_path_map[image_path.stem] = image_path
        return image_path_map

    def _load_image_data_by_id(self, dataset_path: Path) -> Dict[str, Path]:
        image_tar_files = (
            list(dataset_path.glob("*-images.tar")) +
            list(dataset_path.glob("*Images.tar.gz"))
        )
        
        if image_tar_files:
            tar_file = image_tar_files[0]
            
            if tar_file.name.endswith(".tar.gz"):
                dir_name = tar_file.name[:-len(".tar.gz")]
            else:
                dir_name = tar_file.stem
            extract_dir = dataset_path / f"{dir_name}_extracted"
            logger.info(f"找到TAR压缩包: {tar_file}, 将在 {extract_dir} 中寻找/解压图像。")
            if not extract_dir.exists():
                logger.info(f"提取图像从 {tar_file} 到 {extract_dir}")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_file, 'r:*') as tar:
                    tar.extractall(path=extract_dir)
            else:
                logger.info(f"图像已从缓存目录加载: {extract_dir}")
            return self._scan_image_directory(extract_dir)

        possible_dirs = list(dataset_path.glob("*_extracted")) + list(dataset_path.glob("*_images"))
        for image_dir in possible_dirs:
            if image_dir.is_dir():
                logger.info(f"未找到.tar文件，直接从已提取的目录加载图像: {image_dir}")
                return self._scan_image_directory(image_dir)

        logger.warning(f"在 {dataset_path} 中既未找到 *images.tar* 文件，也未找到已提取的图像目录。")
        return {}

    def _load_text_data_by_id(self, dataset_path: Path) -> Dict[str, str]:
        dataset_name = dataset_path.name
        target_jsonl_path = dataset_path / f"{dataset_name}-raw-text.jsonl"
        
        if dataset_name in self.MAGB_DATASETS and not target_jsonl_path.exists():
            logger.info(f"检测到MAGB数据集 '{dataset_name}' 且标准化的JSONL文件不存在，开始自动转换...")
            csv_path = dataset_path / f"{dataset_name}.csv"
            if not csv_path.exists():
                logger.error(f"自动转换失败：在 {dataset_path} 中未找到源文件 {csv_path.name}")
                return {}
            try:
                convert_csv_to_jsonl(csv_path, target_jsonl_path)
                logger.info(f"自动转换成功，已生成: {target_jsonl_path}")
            except Exception as e:
                logger.error(f"自动转换过程中发生错误: {e}", exc_info=True)
                return {}

        if not target_jsonl_path.exists():
            jsonl_files = list(dataset_path.glob("*.jsonl"))
            if not jsonl_files:
                logger.warning(f"在 {dataset_path} 中未找到任何 .jsonl 文件。")
                return {}
            target_jsonl_path = jsonl_files[0]
            logger.info(f"未找到标准命名格式的jsonl，使用找到的第一个文件: {target_jsonl_path}")
        
        with open(target_jsonl_path, 'r', encoding='utf-8') as f:
            text_data_map = {}
            for line in f:
                try:
                    record = json.loads(line.strip())
                    node_id = str(record.get("id") or record.get("asin", "UnknownID"))
                    raw_text = record.get("raw_text", "")
                    text_content = " ".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
                    if node_id != "UnknownID":
                        text_data_map[node_id] = text_content.strip()
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"跳过格式错误的JSONL行: {line.strip()} - {e}")
            return text_data_map

    def run(self):
        logger.info("特征提取管道启动...")
        datasets = self.datasets_to_process or self.storage_manager.list_datasets()
        logger.info(f"目标数据集: {datasets}")
        logger.info(f"目标模态: {self.modalities_to_process}")
        all_results = {}
        for dataset_name in datasets:
            try:
                logger.info(f"======== 开始处理数据集: {dataset_name} ========")
                results = self._process_single_dataset(dataset_name)
                all_results[dataset_name] = results
                logger.info(f"======== 数据集 {dataset_name} 处理完成 ========")
            except Exception as e:
                logger.error(f"处理数据集 {dataset_name} 时发生严重错误: {e}", exc_info=True)
                all_results[dataset_name] = {"error": str(e)}
        logger.info("所有指定的数据集处理完成。")
        logger.info(f"最终结果: \n{json.dumps(all_results, indent=2)}")
        return all_results

    def _process_single_dataset(self, dataset_name: str) -> Dict[str, str]:
        dataset_path = self.storage_manager.get_dataset_path(dataset_name)
        results = {}
        text_data_map = self._load_text_data_by_id(dataset_path)
        image_data_map = self._load_image_data_by_id(dataset_path)
        all_node_ids = sorted(list(set(text_data_map.keys()) | set(image_data_map.keys())))
        if not all_node_ids:
            logger.warning(f"数据集 {dataset_name} 未找到任何文本或图像数据，跳过处理。")
            return {}
        
        id_file_path = dataset_path / "node_ids.json"
        with open(id_file_path, 'w', encoding='utf-8') as f: json.dump(all_node_ids, f)
        logger.info(f"找到 {len(all_node_ids)} 个唯一节点ID，权威ID列表已保存至: {id_file_path}")

        for modality in self.modalities_to_process:
            logger.info(f"--- 处理模态: {modality} ---")
            target_dim = self.target_dimensions.get(modality)
            feature_path = self.storage_manager.get_feature_path(dataset_name, modality, self.encoder_model, target_dim)
            if feature_path.exists() and not self.force_reprocess:
                logger.info(f"特征文件已存在，跳过: {feature_path}")
                results[modality] = str(feature_path)
                continue
            
            encoder = self._get_encoder(modality, target_dim)
            
            if modality == ModalityType.TEXT.value:
                data_list = [text_data_map.get(node_id, "") for node_id in all_node_ids]
                embeddings = encoder.encode(ModalityType.TEXT, texts=data_list, batch_size=self.batch_size)
            elif modality == ModalityType.IMAGE.value:
                data_list = [str(image_data_map.get(node_id, "")) for node_id in all_node_ids]
                embeddings = encoder.encode(ModalityType.IMAGE, image_paths=data_list, batch_size=self.batch_size)
            elif modality == ModalityType.MULTIMODAL.value:
                texts_list = [text_data_map.get(node_id, "") for node_id in all_node_ids]
                images_list = [str(image_data_map.get(node_id, "")) for node_id in all_node_ids]
                embeddings = encoder.encode(ModalityType.MULTIMODAL, texts=texts_list, image_paths=images_list, batch_size=self.batch_size)
            else:
                logger.warning(f"不支持的模态: {modality}，跳过处理。")
                continue
                
            self.storage_manager.save_features(embeddings, feature_path)
            self.quality_checker.check_feature_file(feature_path, target_dim)
            results[modality] = str(feature_path)
            
        self._update_dataset_metadata(dataset_name, results)
        return results

    def _update_dataset_metadata(self, dataset_name: str, processing_results: Dict[str, str]) -> None:
        try:
            metadata_path = self.storage_manager.get_dataset_path(dataset_name) / 'processing_metadata.json'
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

            if 'encoder_runs' not in metadata: metadata['encoder_runs'] = {}
            
            encoder_key = self.encoder_model.replace('/', '_')
            dim_parts = [f"{m}{self.target_dimensions.get(m, 'native')}" for m in self.modalities_to_process]
            dim_key = "_".join(dim_parts)
            feature_key = f"{encoder_key}_{dim_key}"
            
            metadata['encoder_runs'][feature_key] = processing_results
            metadata['node_id_file'] = "node_ids.json"
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"元数据已更新: {metadata_path}")
        except Exception as e:
            logger.error(f"更新元数据失败: {e}", exc_info=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="特征提取管道主程序")
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="指向YAML配置文件的路径。相对路径是相对于项目根目录。",
        default="configs/embedding/qwen2.5-vl-3b-test.yaml"
    )
    args = parser.parse_args()
    
    config_path = project_root / args.config

    try:
        config = load_embedding_config(config_path)
        pipeline = FeaturePipeline(config)
        pipeline.run()
    except Exception as e:
        logger.error(f"管道运行失败: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()