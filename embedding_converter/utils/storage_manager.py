import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Union, Dict
import logging
import json
import tarfile

logger = logging.getLogger(__name__)

class StorageManager:
    """存储管理器，负责管理数据集特征文件的存储路径和命名规范"""
    
    def __init__(self, base_path: Optional[Union[str, Path]] = None):
        """
        初始化存储管理器。

        Args:
            base_path (Optional[Union[str, Path]]):
                可选参数，用于覆盖默认的数据集根路径。
                如果为None，则使用服务器的默认路径 /home/ai/MMAG。
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path("/home/ai/MMAG")
        self._ensure_directory_exists(self.base_path)
        self._node_id_cache: Dict[str, List[str]] = {}
        self._raw_text_cache: Dict[str, Dict[str, str]] = {}
        self._image_path_cache: Dict[str, Dict[str, Path]] = {}

    def _ensure_directory_exists(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
    
    def _scan_image_directory(self, image_dir: Path) -> Dict[str, Path]:
        """递归扫描图像目录，建立 stem -> path 的映射"""
        image_path_map = {}
        print(f"正在扫描图像目录: {image_dir}")
        for image_path in image_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                image_path_map[image_path.stem] = image_path
        return image_path_map

    def load_image_path_map(self, dataset_name: str) -> Dict[str, Path]:
        """加载或从缓存中获取图像ID到路径的映射。"""
        if dataset_name in self._image_path_cache:
            return self._image_path_cache[dataset_name]

        dataset_path = self.get_dataset_path(dataset_name)
        image_tar_files = list(dataset_path.glob("*-images.tar")) + list(dataset_path.glob("*Images.tar.gz"))
        
        if image_tar_files:
            tar_file = image_tar_files[0]
            dir_name = tar_file.name[:-len(".tar.gz")] if tar_file.name.endswith(".tar.gz") else tar_file.stem
            extract_dir = dataset_path / f"{dir_name}_extracted"
            
            if not extract_dir.exists():
                print(f"提取图像从 {tar_file} 到 {extract_dir}")
                extract_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_file, 'r:*') as tar:
                    tar.extractall(path=extract_dir)
            
            image_map = self._scan_image_directory(extract_dir)
            self._image_path_cache[dataset_name] = image_map
            return image_map

        possible_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and (d.name.endswith("_extracted") or d.name.endswith("_images"))]
        if possible_dirs:
            image_map = self._scan_image_directory(possible_dirs[0])
            self._image_path_cache[dataset_name] = image_map
            return image_map

        print(f"警告: 在 {dataset_path} 中未找到图像数据。")
        return {}

    def get_feature_path(self, dataset_name: str, modality: str, encoder_name: str, dimension: Optional[int]) -> Path:
        feature_dir = self.base_path / dataset_name / f"{modality}_features"
        self._ensure_directory_exists(feature_dir)
        safe_encoder_name = encoder_name.replace('/', '_').replace('-', '_').replace('.', '_')
        dim_str = f"{dimension}d" if dimension is not None else "native"
        filename = f"{dataset_name}_{modality}_{safe_encoder_name}_{dim_str}.npy"
        return feature_dir / filename

    def feature_file_exists(self,
                            dataset_name: str,
                            modality: str,
                            encoder_name: str,
                            dimension: Optional[int]) -> bool:
        """检查特征文件是否已存在"""
        filepath = self.get_feature_path(dataset_name, modality, encoder_name, dimension)
        return filepath.exists()

    def get_dataset_metadata_path(self, dataset_name: str) -> Path:
        """获取数据集元数据文件路径"""
        dataset_dir = self.base_path / dataset_name
        self._ensure_directory_exists(dataset_dir)
        return dataset_dir / "metadata.yaml"

    def load_node_ids(self, dataset_name: str) -> List[str]:
        """加载或从缓存中获取node_ids列表。"""
        if dataset_name not in self._node_id_cache:
            node_ids_path = self.get_dataset_path(dataset_name) / "node_ids.json"
            if not node_ids_path.exists(): raise FileNotFoundError(f"节点ID文件未找到: {node_ids_path}")
            with open(node_ids_path, 'r') as f:
                self._node_id_cache[dataset_name] = json.load(f)
        return self._node_id_cache[dataset_name]

    def load_raw_text_map(self, dataset_name: str) -> Dict[str, str]:
        """加载或从缓存中获取原始文本的ID到内容的映射。"""
        if dataset_name not in self._raw_text_cache:
            dataset_path = self.get_dataset_path(dataset_name)
            jsonl_path = dataset_path / f"{dataset_name}-raw-text.jsonl"
            if not jsonl_path.exists(): raise FileNotFoundError(f"原始文本文件未找到: {jsonl_path}")
            text_map = {}
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    node_id = str(record.get("id") or record.get("asin"))
                    raw_text = record.get("raw_text", "")
                    text_content = " ".join(raw_text) if isinstance(raw_text, list) else str(raw_text)
                    text_map[node_id] = text_content.strip()
            self._raw_text_cache[dataset_name] = text_map
        return self._raw_text_cache[dataset_name]

    def get_dataset_path(self, dataset_name: str) -> Path:
        """获取数据集目录路径"""
        dataset_path = self.base_path / dataset_name
        self._ensure_directory_exists(dataset_path)
        return dataset_path
    
    def save_features(self, embeddings: np.ndarray, feature_path: Union[str, Path]) -> None:
        """保存特征向量到文件，并统一数据类型为float32"""
        feature_path = Path(feature_path)
        self._ensure_directory_exists(feature_path.parent)
        
        # 在保存前，将数据类型统一转换为float32
        embeddings_as_float32 = embeddings.astype(np.float32)
        np.save(feature_path, embeddings_as_float32)
        logger.info(f"特征已作为float32类型保存到: {feature_path}")
    
    def load_features(self, feature_path: Union[str, Path]) -> np.ndarray:
        """从文件加载特征向量"""
        return np.load(feature_path)
    
    def list_datasets(self) -> List[str]:
        if not self.base_path.exists(): return []
        return sorted([item.name for item in self.base_path.iterdir() if item.is_dir() and not item.name.startswith('.')])