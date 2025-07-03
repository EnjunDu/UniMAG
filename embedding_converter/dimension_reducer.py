import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from typing import Optional, Literal
import logging

logger = logging.getLogger(__name__)

class DimensionReducer:
    """
    维度缩减器
    
    支持线性变换和PCA降维
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 method: Literal["linear", "pca"] = "linear",
                 device: Optional[str] = None):
        """
        初始化维度缩减器
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            method: 降维方法 ('linear' 或 'pca')
            device: 计算设备
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.is_fitted = False
        self.reducer = None
        
        if method == "linear":
            self._init_linear_reducer()
        elif method == "pca":
            self._init_pca_reducer()
        else:
            raise ValueError(f"不支持的降维方法: {method}")
    
    def _init_linear_reducer(self):
        """初始化线性降维器"""
        self.reducer = nn.Linear(self.input_dim, self.output_dim, bias=False)
        # 使用Xavier初始化
        nn.init.xavier_uniform_(self.reducer.weight)
        self.reducer.to(self.device)
        self.is_fitted = True
        logger.info(f"初始化线性降维器: {self.input_dim} -> {self.output_dim}")
    
    def _init_pca_reducer(self):
        """初始化PCA降维器"""
        self.reducer = PCA(n_components=self.output_dim)
        logger.info(f"初始化PCA降维器: {self.input_dim} -> {self.output_dim}")
    
    def fit(self, embeddings: np.ndarray) -> 'DimensionReducer':
        """
        拟合降维器（主要用于PCA）
        
        Args:
            embeddings: 训练数据 [num_samples, input_dim]
            
        Returns:
            self
        """
        if self.method == "pca":
            if embeddings.shape[1] != self.input_dim:
                raise ValueError(f"输入维度不匹配: 期望{self.input_dim}, 实际{embeddings.shape[1]}")
            
            logger.info("开始拟合PCA降维器...")
            self.reducer.fit(embeddings)
            self.is_fitted = True
            
            # 打印解释方差比
            explained_ratio = np.sum(self.reducer.explained_variance_ratio_)
            logger.info(f"PCA降维完成，保留方差比例: {explained_ratio:.4f}")
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        应用维度变换
        
        Args:
            embeddings: 输入嵌入 [num_samples, input_dim]
            
        Returns:
            变换后的嵌入 [num_samples, output_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("降维器尚未拟合，请先调用fit()方法")
        
        if embeddings.shape[1] != self.input_dim:
            raise ValueError(f"输入维度不匹配: 期望{self.input_dim}, 实际{embeddings.shape[1]}")
        
        if self.method == "linear":
            return self._transform_linear(embeddings)
        elif self.method == "pca":
            return self._transform_pca(embeddings)
    
    def _transform_linear(self, embeddings: np.ndarray) -> np.ndarray:
        """线性变换"""
        with torch.no_grad():
            embeddings_tensor = torch.from_numpy(embeddings).float().to(self.device)
            transformed = self.reducer(embeddings_tensor)
            return transformed.cpu().numpy()
    
    def _transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA变换"""
        return self.reducer.transform(embeddings)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        拟合并变换（主要用于PCA的首次使用）
        
        Args:
            embeddings: 输入嵌入 [num_samples, input_dim]
            
        Returns:
            变换后的嵌入 [num_samples, output_dim]
        """
        return self.fit(embeddings).transform(embeddings)
    
    def save_state(self, filepath: str) -> None:
        """
        保存降维器状态
        
        Args:
            filepath: 保存路径
        """
        state = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        
        if self.method == "linear":
            state['linear_weight'] = self.reducer.weight.detach().cpu().numpy()
        elif self.method == "pca":
            if self.is_fitted:
                state['pca_components'] = self.reducer.components_
                state['pca_mean'] = self.reducer.mean_
                state['pca_explained_variance'] = self.reducer.explained_variance_
                state['pca_explained_variance_ratio'] = self.reducer.explained_variance_ratio_
        
        np.savez(filepath, **state)
        logger.info(f"降维器状态已保存到: {filepath}")
    
    @classmethod
    def load_state(cls, filepath: str, device: Optional[str] = None) -> 'DimensionReducer':
        """
        从文件加载降维器状态
        
        Args:
            filepath: 状态文件路径
            device: 计算设备
            
        Returns:
            加载的降维器实例
        """
        state = np.load(filepath, allow_pickle=True)
        
        reducer = cls(
            input_dim=int(state['input_dim']),
            output_dim=int(state['output_dim']),
            method=str(state['method']),
            device=device
        )
        
        if state['method'] == "linear":
            reducer.reducer.weight.data = torch.from_numpy(state['linear_weight']).to(reducer.device)
        elif state['method'] == "pca":
            if state['is_fitted']:
                reducer.reducer.components_ = state['pca_components']
                reducer.reducer.mean_ = state['pca_mean']
                reducer.reducer.explained_variance_ = state['pca_explained_variance']
                reducer.reducer.explained_variance_ratio_ = state['pca_explained_variance_ratio']
                reducer.reducer.n_components_ = len(state['pca_components'])
        
        reducer.is_fitted = bool(state['is_fitted'])
        logger.info(f"降维器状态已从 {filepath} 加载")
        
        return reducer