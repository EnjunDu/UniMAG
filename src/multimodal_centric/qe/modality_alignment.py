# -*- coding: utf-8 -*-
"""
模态对齐 (Modality Alignment)

此模块实现了“多模态质量评估”任务中的“模态对齐”部分。
它旨在通过以下两个阶段，对视觉和文本模态的细粒度对齐程度进行评估：

1.  **数据预处理**:
    - 从文本中提取关键名词短语。
    - 使用外部视觉定位模型（如GroundingDINO）为短语找到图像中的边界框。
    - 存储这些 (image_path, text, [(phrase, box), ...]) 作为基准真值。

2.  **评估**:
    - **传统方法**: 直接在图像和短语上评估我们自己编码器的对齐能力。
    - **MAG特定方法**: 利用图结构，在邻域增强的特征图上进行评估。
"""
import sys
import os
import glob
from tqdm import tqdm

# 将项目根目录添加到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import spacy
from PIL import Image
from typing import List, Tuple, Dict, Any, Optional
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from torchvision.ops import roi_align
import numpy as np
import os
import json
from pathlib import Path

# 导入项目内的模块
from utils.embedding_manager import EmbeddingManager
from utils.graph_loader import GraphLoader
from src.model.models import GCN, GAT, GraphSAGE

class ModalityAligner:
    """
    负责执行模态对齐预处理和评估的核心类。
    """
    def __init__(
        self,
        grounding_model_id: str = "IDEA-Research/grounding-dino-base",
        device: Optional[str] = None
    ):
        """
        初始化ModalityAligner。

        Args:
            grounding_model_id (str): 用于视觉定位的Hugging Face模型ID。
            device (Optional[str]): 计算设备 (例如 "cuda", "cpu")。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载用于名词短语提取的spaCy模型
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("未找到spaCy的'en_core_web_sm'模型，正在尝试下载...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        # 加载视觉定位模型
        self.grounding_processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(self.device)
        
        print(f"ModalityAligner 初始化完成，使用设备: {self.device}")

    def extract_noun_phrases(self, text: str) -> List[str]:
        """
        从给定文本中提取名词短语。

        Args:
            text (str): 输入的文本。

        Returns:
            List[str]: 提取出的名词短语列表。
        """
        doc = self.nlp(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def generate_grounding_truth(
        self,
        image_path: str,
        text: str
    ) -> Dict[str, Any]:
        """
        为单个图文对生成基准真值。

        Args:
            image_path (str): 图像的文件路径。
            text (str): 与图像配对的文本。

        Returns:
            Dict[str, Any]: 包含图像路径、文本和定位结果的字典。
                           格式: {"image_path": ..., "text": ..., "grounding": [(phrase, box), ...]}
        """
        image = Image.open(image_path).convert("RGB")
        phrases = self.extract_noun_phrases(text)
        if not phrases:
            return {"image_path": image_path, "text": text, "grounding": []}

        # 将短语用'.'连接，以符合GroundingDINO的输入格式
        text_for_grounding = ". ".join(phrases)
        
        inputs = self.grounding_processor(images=image, text=text_for_grounding, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )
        
        grounding_results = []
        for phrase, box in zip(results[0]["labels"], results[0]["boxes"]):
            grounding_results.append((phrase, box.cpu().numpy().tolist()))
            
        return {
            "image_path": image_path,
            "text": text,
            "grounding": grounding_results
        }

    def _align_features(
        self,
        feature_map: torch.Tensor,
        phrase_embedding: torch.Tensor,
        box: List[float]
    ) -> float:
        """
        计算单个短语和其对应图像区域之间的对齐分数。

        Args:
            feature_map (torch.Tensor): 图像的特征图 (1, C, H, W)。
            phrase_embedding (torch.Tensor): 短语的嵌入向量 (1, D)。
            box (List[float]): 边界框 [xmin, ymin, xmax, ymax]。

        Returns:
            float: 对齐分数 (余弦相似度)。
        """
        # RoIAlign需要一个批次的边界框
        box_tensor = torch.tensor([box], dtype=feature_map.dtype, device=self.device)
        
        # roi_align的输出尺寸可以根据需要调整
        # 这里我们假设输出一个固定大小的区域特征，例如 7x7
        # 输出形状将是 (1, C, 7, 7)
        region_features = roi_align(feature_map, [box_tensor], output_size=(7, 7), spatial_scale=1.0)
        
        # 将区域特征池化为一个向量
        # (1, C, 7, 7) -> (1, C)
        pooled_region_feature = torch.nn.functional.adaptive_avg_pool2d(region_features, (1, 1)).flatten(1)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(pooled_region_feature, phrase_embedding)
        
        return similarity.item()

    def evaluate_alignment(
        self,
        ground_truth: Dict[str, Any],
        embedding_manager: EmbeddingManager,
        encoder_type: str,
        encoder_name: str,
        dimension: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        评估传统方法下的模态对齐。

        Args:
            ground_truth (Dict[str, Any]): 单个基准真值条目。
            embedding_manager (EmbeddingManager): 用于生成我们自己嵌入的管理器。
            encoder_type (str): 我们自己编码器的类型。
            encoder_name (str): 我们自己编码器的名称。
            dimension (Optional[int]): 嵌入维度。

        Returns:
            List[Dict[str, Any]]: 每个短语的评估结果列表。
        """
        image_path = ground_truth["image_path"]
        
        # 1. 获取我们自己编码器的图像特征图
        # generate_embedding 返回一个列表，我们取第一个元素
        feature_maps = embedding_manager.generate_embedding(
            data=[image_path],
            modality="image",
            encoder_type=encoder_type,
            encoder_name=encoder_name,
            dimension=dimension,
            output_feature_map=True
        )
        if feature_maps.size == 0:
            print("错误: 无法生成特征图。")
            return []
        
        # 将numpy数组转为torch张量
        feature_map = torch.from_numpy(feature_maps).to(self.device)
        if feature_map.ndim == 3:
            feature_map = feature_map.unsqueeze(0)
        
        results = []
        for phrase, box in ground_truth["grounding"]:
            # 2. 获取我们自己编码器的短语嵌入
            phrase_embedding_np = embedding_manager.generate_embedding(
                data=[phrase],
                modality="text",
                encoder_type=encoder_type,
                encoder_name=encoder_name,
                dimension=dimension
            )
            if phrase_embedding_np is None:
                continue
            
            phrase_embedding = torch.from_numpy(phrase_embedding_np).to(self.device)
            
            # 3. 计算对齐分数
            alignment_score = self._align_features(feature_map, phrase_embedding, box)
            
            results.append({
                "phrase": phrase,
                "box": box,
                "alignment_score": alignment_score
            })
            
        return results

    def evaluate_mag_alignment(
        self,
        node_id: str,
        dataset_name: str,
        ground_truth: Dict[str, Any],
        embedding_manager: EmbeddingManager,
        graph_loader: GraphLoader,
        gnn_model: torch.nn.Module,
        encoder_type: str,
        encoder_name: str,
        dimension: Optional[int] = 768
    ) -> List[Dict[str, Any]]:
        """
        评估MAG特定方法下的模态对齐。

        Args:
            node_id (str): 目标节点的ID。
            dataset_name (str): 数据集名称。
            ground_truth (Dict[str, Any]): 基准真值。
            embedding_manager (EmbeddingManager): 嵌入管理器。
            graph_loader (GraphLoader): 图加载器。
            gnn_model (torch.nn.Module): 一个已经实例化的GNN模型。
            encoder_type (str): 编码器类型。
            encoder_name (str): 编码器名称。
            dimension (Optional[int]): 嵌入维度。

        Returns:
            List[Dict[str, Any]]: 每个短语的评估结果列表。
        """
        # 1. 加载图和节点ID
        graph_data = graph_loader.load_graph(dataset_name)
        edge_index = graph_data.edge_index.to(self.device)
        
        dataset_path = Path(graph_loader.data_root) / dataset_name
        node_ids_path = dataset_path / "node_ids.json"
        if not node_ids_path.exists():
            raise FileNotFoundError(f"节点ID文件未找到: {node_ids_path}")
        with open(node_ids_path, 'r') as f:
            all_node_ids = json.load(f)
        
        # 2. 构建图像路径列表
        image_paths = []
        dataset_root_str = str(dataset_path)
        # 预先扫描所有图片，构建一个从ID到路径的映射
        all_images = glob.glob(os.path.join(dataset_root_str, '**', f"*.jpg"), recursive=True)
        image_map = {Path(p).stem: p for p in all_images}

        for nid in tqdm(all_node_ids, desc="构建图像路径"):
            image_paths.append(image_map.get(nid, ""))

        # 3. 批量获取所有节点的特征图
        all_feature_maps_list = embedding_manager.generate_embedding(
            data=image_paths,
            modality="image",
            encoder_type=encoder_type,
            encoder_name=encoder_name,
            dimension=dimension,
            output_feature_map=True
        )
        
        # 将特征图列表处理成一个 (N, C, H, W) 的张量
        # 注意：这假设所有特征图大小相同，对于某些模型可能是这样
        all_feature_maps = torch.from_numpy(np.concatenate(all_feature_maps_list, axis=0)).to(self.device)
        
        # 4. 使用GNN聚合特征
        num_nodes, C, H, W = all_feature_maps.shape
        
        # 直接使用传入的gnn_model
        gnn_model.to(self.device)
        
        flat_features = all_feature_maps.view(num_nodes, -1)
        # 注意：确保传入的gnn_model的in_dim与flat_features的维度匹配
        enhanced_flat_features, _, _ = gnn_model(flat_features, edge_index)
        
        # 假设GNN模型的输出维度可以重新塑形回 (C, H, W)
        # 这需要GNN的hidden_dim等于 C*H*W，或者进行相应调整
        try:
            enhanced_feature_maps = enhanced_flat_features.view(num_nodes, C, H, W)
        except RuntimeError:
            print(f"警告: GNN输出维度 {enhanced_flat_features.shape[1]} 与原始特征图维度 {C*H*W} 不匹配。")
            print("将直接使用扁平化的增强特征进行后续计算，这可能不是预期行为。")
            # 在这种情况下，需要一个从 enhanced_flat_features 到区域特征的映射，这里简化处理
            # 实际应用中可能需要更复杂的解码器结构
            enhanced_feature_maps = all_feature_maps # 作为后备
        
        # 5. 提取目标节点的增强特征图
        try:
            target_node_index = all_node_ids.index(node_id)
        except ValueError:
            raise ValueError(f"节点ID '{node_id}' 不在数据集中。")
            
        target_enhanced_feature_map = enhanced_feature_maps[target_node_index].unsqueeze(0)

        # 6. 评估
        results = []
        for phrase, box in ground_truth["grounding"]:
            phrase_embedding_np = embedding_manager.generate_embedding(
                data=[phrase],
                modality="text",
                encoder_type=encoder_type,
                encoder_name=encoder_name,
                dimension=dimension
            )
            if phrase_embedding_np is None:
                continue
            
            phrase_embedding = torch.from_numpy(phrase_embedding_np).to(self.device)
            
            alignment_score = self._align_features(target_enhanced_feature_map, phrase_embedding, box)
            
            results.append({
                "phrase": phrase,
                "box": box,
                "mag_alignment_score": alignment_score
            })
            
        return results

if __name__ == '__main__':
    print("=== 模态对齐模块使用示例 ===")

    # --- 固定配置 ---
    DATASET_NAME = "Grocery"
    ENCODER_TYPE = "qwen_vl"
    ENCODER_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
    DIMENSION = 768
    TARGET_NODE_ID = "1"  # 使用字符串ID
    # 服务器上的固定数据根目录
    DATA_ROOT = "/home/ai/MMAG"

    # --- 初始化 ---
    aligner = ModalityAligner()
    manager = EmbeddingManager()
    loader = GraphLoader()

    # --- 加载真实数据进行测试 ---
    print(f"--- 准备从 '{DATASET_NAME}' 数据集加载节点 '{TARGET_NODE_ID}' 的真实数据 ---")
    
    # 构造图像路径
    dataset_dir = Path(DATA_ROOT) / DATASET_NAME
    search_pattern = os.path.join(str(dataset_dir), '**', f"{TARGET_NODE_ID}.*")
    found_files = glob.glob(search_pattern, recursive=True)
    image_path = Path(found_files[0]) if found_files else None

    # 构造文本路径并加载文本
    text_path = dataset_dir / f"{DATASET_NAME}-raw-text.jsonl"
    node_text = ""
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                # 兼容 'id' 和 'asin' 两种节点标识符
                current_id = record.get("id") or record.get("asin")
                if str(current_id) == TARGET_NODE_ID:
                    raw_text = record.get("raw_text", "")
                    # 确保文本是字符串，如果是列表则拼接
                    if isinstance(raw_text, list):
                        node_text = " ".join(raw_text)
                    else:
                        node_text = str(raw_text)
                    break
    except FileNotFoundError:
        print(f"错误: 文本文件未找到: {text_path}")
        sys.exit(1)
    except Exception as e:
        print(f"读取或解析文本文件时出错: {e}")
        sys.exit(1)

    if not image_path or not image_path.exists() or not node_text:
        print(f"错误: 无法为节点 '{TARGET_NODE_ID}' 加载完整的图文数据。")
        print(f"  - 检查图像是否存在: {image_path} ({'存在' if image_path.exists() else '不存在'})")
        print(f"  - 检查文本是否已加载: {'是' if node_text else '否'}")
        sys.exit(1)
        
    print(f"成功加载节点 '{TARGET_NODE_ID}' 的数据。")
    print(f"  - 图像路径: {image_path}")
    print(f"  - 文本: '{node_text[:100]}...'")


    # 1. 生成基准真值
    print("\n--- 1. 生成基准真值 ---")
    ground_truth_data = aligner.generate_grounding_truth(str(image_path), node_text)
    print(f"成功生成基准真值，找到 {len(ground_truth_data.get('grounding', []))} 个短语。")

    if ground_truth_data and ground_truth_data.get('grounding'):
        # 2. 评估传统方法的对齐
        print("\n--- 2. 评估传统对齐 ---")
        traditional_results = aligner.evaluate_alignment(
            ground_truth=ground_truth_data,
            embedding_manager=manager,
            encoder_type=ENCODER_TYPE,
            encoder_name=ENCODER_NAME,
            dimension=DIMENSION
        )
        print(f"传统对齐评估完成，处理了 {len(traditional_results)} 个短语。")
        if traditional_results:
            print(f"  - 示例结果: {traditional_results[0]}")

        # 3. 评估MAG特定方法的对齐
        print("\n--- 3. 评估MAG特定对齐 ---")
        try:
            # 实例化GCN模型
            # 注意：这里的维度设置需要非常小心
            # 假设特征图可以被展平
            # C, H, W 需要从实际的特征图获取，这里使用占位符
            # 实际应用中，可能需要先获取一个样本特征图来确定维度
            placeholder_C, placeholder_H, placeholder_W = 256, 7, 7
            gcn_in_dim = placeholder_C * placeholder_H * placeholder_W

            gcn_model = GCN(
                in_dim=gcn_in_dim,
                hidden_dim=gcn_in_dim, # 保持维度以便恢复形状
                num_layers=2,
                dropout=0.5
            )

            mag_results = aligner.evaluate_mag_alignment(
                node_id=TARGET_NODE_ID,
                dataset_name=DATASET_NAME,
                ground_truth=ground_truth_data,
                embedding_manager=manager,
                graph_loader=loader,
                gnn_model=gcn_model,
                encoder_type=ENCODER_TYPE,
                encoder_name=ENCODER_NAME,
                dimension=DIMENSION
            )
            print(f"MAG对齐评估完成，处理了 {len(mag_results)} 个短语。")
            if mag_results:
                print(f"  - 示例结果: {mag_results[0]}")
        except Exception as e:
            print(f"MAG对齐评估失败: {e}")
    else:
        print("\n未能生成有效的基准真值，跳过对齐评估。")