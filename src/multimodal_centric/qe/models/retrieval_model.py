# -*- coding: utf-8 -*-
"""
用于模态检索的双塔模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    """
    一个简单的双塔模型，用于模态检索。
    每个塔是一个多层感知机 (MLP)，将输入嵌入投影到共享的语义空间。
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float):
        """
        初始化双塔模型。

        Args:
            input_dim (int): 输入嵌入的维度 (来自GNN的增强嵌入)。
            hidden_dim (int): MLP隐藏层的维度。
            output_dim (int): 输出嵌入的维度 (最终的检索空间维度)。
            num_layers (int): 每个塔的MLP层数。
            dropout (float): Dropout比率。
        """
        super().__init__()
        self.text_tower = self._create_tower(input_dim, hidden_dim, output_dim, num_layers, dropout)
        self.image_tower = self._create_tower(input_dim, hidden_dim, output_dim, num_layers, dropout)

    def _create_tower(self, input_dim, hidden_dim, output_dim, num_layers, dropout) -> nn.Sequential:
        """
        创建一个MLP塔。
        """
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        前向传播。

        Args:
            text_embeds (torch.Tensor): 文本塔的输入嵌入。
            image_embeds (torch.Tensor): 图像塔的输入嵌入。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 投影后的文本和图像嵌入。
        """
        projected_text = self.text_tower(text_embeds)
        projected_image = self.image_tower(image_embeds)
        return projected_text, projected_image

    def encode_text(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """只编码文本嵌入。"""
        return self.text_tower(text_embeds)

    def encode_image(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """只编码图像嵌入。"""
        return self.image_tower(image_embeds)