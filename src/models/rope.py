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

def apply_rope(q,k,seq_len):
    """
    对Q/K向量应用RoPE
    :param q: Q向量，形状为(batch_size, num_heads, seq_len, head_dim)
    :param k: K向量，形状为(batch_size, num_heads, seq_len, head_dim)
    :param seq_len: 序列长度
    :return: 旋转后的Q/K向量
    """
    batch_size, num_heads, _, head_dim = q.shape
    assert head_dim % 2 == 0, "head_dim必须是偶数"

    # 步骤1: 计算角度系数a_k
    k_indices = torch.arange(0, head_dim, 2, dtype=q.device)
    alpha = 10000.0 ** (-2 * k_indices / head_dim)  # [head_dim/2]
    m = torch.arange(seq_len, device=q.device)[:, None]  # [seq_len, 1]

    # m 扩展为 [3,2]（列扩展）
    # m_expanded = [[0, 0],
    #               [1, 1],
    #               [2, 2]]
    #
    # # alpha 扩展为 [3,2]（行扩展）
    # alpha_expanded = [[1.0, 0.0001],
    #                   [1.0, 0.0001],
    #                   [1.0, 0.0001]]

    # theta = m_expanded * alpha_expanded
        # =   [[0×1.0, 0×0.0001],
        #     [1×1.0, 1×0.0001],
        #     [2×1.0, 2×0.0001]]
        # =   [[0.0, 0.0],
        #     [1.0, 0.0001],
        #     [2.0, 0.0002]]
    theta = m * alpha  # [seq_len, head_dim/2]

