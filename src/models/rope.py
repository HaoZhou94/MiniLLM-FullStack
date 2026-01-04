"""
对每个位置的 Q/K 向量做旋转变换，位置不同，旋转角度不同。
旋转变换的数学性质保证：两个向量旋转后的点积，只和它们的相对旋转角度有关。
相对旋转角度 = 绝对位置差 × 角度系数，因此点积只和相对位置有关。
"""
import os
import sys
import torch
import torch.nn as nn
import math

# 获取项目根目录（MiniLLM-FullStack）的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 将项目根目录加入Python搜索路径
sys.path.append(project_root)

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
    k_indices = torch.arange(0, head_dim, 2, device=q.device)
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


    # 步骤3：预计算cos和sin
    cos_theta = torch.cos(theta).unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,head_dim/2]
    sin_theta = torch.sin(theta).unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,head_dim/2]

    # 步骤4：向量化拆分与旋转
    q1, q2 = q[..., ::2], q[..., 1::2]  # [batch, heads, seq_len, head_dim/2]
    k1, k2 = k[..., ::2], k[..., 1::2]

    q1_rot = q1 * cos_theta - q2 * sin_theta
    q2_rot = q1 * sin_theta + q2 * cos_theta
    k1_rot = k1 * cos_theta - k2 * sin_theta
    k2_rot = k1 * sin_theta + k2 * cos_theta

    # 步骤5：拼接
    q_rot = torch.stack([q1_rot, q2_rot], dim=-1).reshape_as(q)
    k_rot = torch.stack([k1_rot, k2_rot], dim=-1).reshape_as(k)

    return q_rot, k_rot

if __name__ == "__main__":
    batch_size = 1
    num_heads = 2
    seq_len = 3
    head_dim = 4

    # 构造Q/K向量（随机但固定种子，数值可复现）
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    print("===== 原始输入 =====")
    print(f"Q向量形状: {q.shape}")
    print("Q向量具体值:\n", q)
    print(f"\nK向量形状: {k.shape}")
    print("K向量具体值:\n", k)

    # ---------------------- 2. 调用RoPE函数 ----------------------
    q_rot, k_rot = apply_rope(q, k, seq_len)

    print("\n===== RoPE输出 =====")
    print(f"旋转后Q向量形状: {q_rot.shape}")
    print("旋转后Q向量具体值:\n", q_rot)
    print(f"\n旋转后K向量形状: {k_rot.shape}")
    print("旋转后K向量具体值:\n", k_rot)

    # ---------------------- 3. 验证RoPE核心性质 ----------------------
    # 核心验证：相对位置相同的Q-K点积应相等
    print("\n===== 验证相对位置性质 =====")
    # 取第一个注意力头的结果（head=0）
    head_idx = 0
    q_head = q_rot[0, head_idx]  # [3,4]
    k_head = k_rot[0, head_idx]  # [3,4]

    # 计算相对位置Δ=1的点积（位置0→1，位置1→2）
    dot1 = torch.dot(q_head[0], k_head[1])  # 位置0的Q × 位置1的K
    dot2 = torch.dot(q_head[1], k_head[2])  # 位置1的Q × 位置2的K

    # 计算相对位置Δ=2的点积（位置0→2）
    dot3 = torch.dot(q_head[0], k_head[2])

    print(f"相对位置Δ=1 (0→1) 的点积: {dot1:.6f}")
    print(f"相对位置Δ=1 (1→2) 的点积: {dot2:.6f}")
    print(f"相对位置Δ=2 (0→2) 的点积: {dot3:.6f}")
    print(f"Δ=1的两个点积差值: {abs(dot1 - dot2):.6f}")