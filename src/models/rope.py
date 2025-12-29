"""
对每个位置的 Q/K 向量做旋转变换，位置不同，旋转角度不同。
旋转变换的数学性质保证：两个向量旋转后的点积，只和它们的相对旋转角度有关。
相对旋转角度 = 绝对位置差 × 角度系数，因此点积只和相对位置有关。
"""
import torch
import torch.nn as nn
import math
from src.utils.logger import logger


class DynamicRoPE(nn.Module):
    def __init__(self, dim:int, num_heads:int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.angle_coeff = 10000.0 ** (-2.0 * torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)


        # 生成频率参数theta
        self.theta = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        logger.info(f"初始化RoPE：维度={dim}，头数={num_heads}")


    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """应用RoPE到Q/K向量"""
        pos_theta = torch.outer(pos.float(), self.theta).to(x.device)
        sin = torch.sin(pos_theta).repeat_interleave(2, dim=-1)
        cos = torch.cos(pos_theta).repeat_interleave(2, dim=-1)

        # 旋转计算
        x1 = x[..., ::2]  # 偶数位
        x2 = x[..., 1::2] # 奇数位
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot


    