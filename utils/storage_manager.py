import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Union


class StorageManager:
    """存储管理器，负责管理数据集特征文件的存储路径和命名规范"""
    
    def __init__(self, environment="local", base_path=None):
        self.environment = environment
        
        if base_path:
            self.base_path = base_path
        elif environment == "server":
            self.base_path = "/home/ai/ylzuo/UniMAG/hugging_face"
            self.model_path = "/home/ai/huggingface"
        else:
            self.base_path = "hugging_face"
            self.model_path = None
            
        self._ensure_directory_exists(self.base_path)
    
    def _ensure_directory_exists(self, path):
        """确保目录存在，不存在则创建"""
        if not os.path.exists(path):
            os.makedirs(path)
    
    def get_feature_path(self,
                         dataset_name: str,
                         modality: str,
                         encoder_name: str,
                         dimension: Optional[int]) -> Path:
        """根据命名规范生成特征文件的完整路径"""
        feature_dir = Path(self.base_path) / dataset_name / f"{modality}_features"
        feature_dir.mkdir(parents=True, exist_ok=True)

        # 处理编码器名称中的特殊字符
        safe_encoder_name = encoder_name.replace('/', '_').replace('-', '_').replace('.', '_')
        
        # 构建文件名
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
        dataset_dir = Path(self.base_path) / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir / "metadata.yaml"

    def get_dataset_path(self, dataset_name: str) -> Path:
        """获取数据集目录路径"""
        dataset_path = Path(self.base_path) / dataset_name
        dataset_path.mkdir(parents=True, exist_ok=True)
        return dataset_path
    
    def save_features(self, embeddings: np.ndarray, feature_path: Union[str, Path]) -> None:
        """保存特征向量到文件"""
        feature_path = Path(feature_path)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature_path, embeddings)
    
    def load_features(self, feature_path: Union[str, Path]) -> np.ndarray:
        """从文件加载特征向量"""
        return np.load(feature_path)
    
    def list_datasets(self) -> List[str]:
        """列出所有可用的数据集"""
        base_path = Path(self.base_path)
        if not base_path.exists():
            return []
        
        datasets = []
        for item in base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                datasets.append(item.name)
        
        return sorted(datasets)


if __name__ == "__main__":
    # 测试存储管理器
    storage = StorageManager()
    
    # 测试路径生成
    bert_path = storage.get_feature_path(
        dataset_name="books-nc",
        modality="text",
        encoder_name="bert-base-uncased",
        dimension=768
    )
    print(f"BERT特征文件路径: {bert_path}")
    
    clip_path = storage.get_feature_path(
        dataset_name="books-nc",
        modality="image",
        encoder_name="openai/clip-vit-large-patch14",
        dimension=512
    )
    print(f"CLIP特征文件路径: {clip_path}")