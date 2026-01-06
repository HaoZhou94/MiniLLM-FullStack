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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备


# def apply_rope(q,k,seq_len):
#     """
#     对Q/K向量应用RoPE
#     :param q: Q向量，形状为(batch_size, num_heads, seq_len, head_dim)
#     :param k: K向量，形状为(batch_size, num_heads, seq_len, head_dim)
#     :param seq_len: 序列长度
#     :return: 旋转后的Q/K向量
#     """
#     batch_size, num_heads, _, head_dim = q.shape
#     assert head_dim % 2 == 0, "head_dim必须是偶数"
#
#     # 步骤1: 计算角度系数a_k
#     k_indices = torch.arange(0, head_dim, 2, device=q.device)
#     alpha = 10000.0 ** (-2 * k_indices / head_dim)  # [head_dim/2]
#     m = torch.arange(seq_len, device=q.device)[:, None]  # [seq_len, 1]
#
#     # m 扩展为 [3,2]（列扩展）
#     # m_expanded = [[0, 0],
#     #               [1, 1],
#     #               [2, 2]]
#     #
#     # # alpha 扩展为 [3,2]（行扩展）
#     # alpha_expanded = [[1.0, 0.0001],
#     #                   [1.0, 0.0001],
#     #                   [1.0, 0.0001]]
#
#     # theta = m_expanded * alpha_expanded
#         # =   [[0×1.0, 0×0.0001],
#         #     [1×1.0, 1×0.0001],
#         #     [2×1.0, 2×0.0001]]
#         # =   [[0.0, 0.0],
#         #     [1.0, 0.0001],
#         #     [2.0, 0.0002]]
#     theta = m * alpha  # [seq_len, head_dim/2]
#
#
#     # 步骤3：预计算cos和sin
#     cos_theta = torch.cos(theta).unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,head_dim/2]
#     sin_theta = torch.sin(theta).unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,head_dim/2]
#
#     # 步骤4：向量化拆分与旋转
#     q1, q2 = q[..., ::2], q[..., 1::2]  # [batch, heads, seq_len, head_dim/2]
#     k1, k2 = k[..., ::2], k[..., 1::2]
#
#     q1_rot = q1 * cos_theta - q2 * sin_theta
#     q2_rot = q1 * sin_theta + q2 * cos_theta
#     k1_rot = k1 * cos_theta - k2 * sin_theta
#     k2_rot = k1 * sin_theta + k2 * cos_theta
#
#     # 步骤5：拼接
#     q_rot = torch.stack([q1_rot, q2_rot], dim=-1).reshape_as(q)
#     k_rot = torch.stack([k1_rot, k2_rot], dim=-1).reshape_as(k)
#
#     return q_rot, k_rot
#
# if __name__ == "__main__":
#     batch_size = 1
#     num_heads = 2
#     seq_len = 3
#     head_dim = 4
#
#     # 构造Q/K向量（随机但固定种子，数值可复现）
#     q = torch.randn(batch_size, num_heads, seq_len, head_dim)
#     k = torch.randn(batch_size, num_heads, seq_len, head_dim)
#
#     print("===== 原始输入 =====")
#     print(f"Q向量形状: {q.shape}")
#     print("Q向量具体值:\n", q)
#     print(f"\nK向量形状: {k.shape}")
#     print("K向量具体值:\n", k)
#
#     # ---------------------- 2. 调用RoPE函数 ----------------------
#     q_rot, k_rot = apply_rope(q, k, seq_len)
#
#     print("\n===== RoPE输出 =====")
#     print(f"旋转后Q向量形状: {q_rot.shape}")
#     print("旋转后Q向量具体值:\n", q_rot)
#     print(f"\n旋转后K向量形状: {k_rot.shape}")
#     print("旋转后K向量具体值:\n", k_rot)
#
#     # ---------------------- 3. 验证RoPE核心性质 ----------------------
#     # 核心验证：相对位置相同的Q-K点积应相等
#     print("\n===== 验证相对位置性质 =====")
#     # 取第一个注意力头的结果（head=0）
#     head_idx = 0
#     q_head = q_rot[0, head_idx]  # [3,4]
#     k_head = k_rot[0, head_idx]  # [3,4]
#
#     # 计算相对位置Δ=1的点积（位置0→1，位置1→2）
#     dot1 = torch.dot(q_head[0], k_head[1])  # 位置0的Q × 位置1的K
#     dot2 = torch.dot(q_head[1], k_head[2])  # 位置1的Q × 位置2的K
#
#     # 计算相对位置Δ=2的点积（位置0→2）
#     dot3 = torch.dot(q_head[0], k_head[2])
#
#     import pdb; pdb.set_trace()
#     print(f"相对位置Δ=1 (0→1) 的点积: {dot1:.6f}")
#     print(f"相对位置Δ=1 (1→2) 的点积: {dot2:.6f}")
#     print(f"相对位置Δ=2 (0→2) 的点积: {dot3:.6f}")
#     print(f"Δ=1的两个点积差值: {abs(dot1 - dot2):.6f}")

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
        self.theta = self.theta.to(x.device)
        # 计算位置-二维对的旋转角度（外积等价于广播乘法）
        pos_theta = torch.outer(pos.float(), self.theta).to(x.device)
        # 扩展sin/cos到与x最后一维同维度（每个二维对的sin/cos重复2次）
        sin = torch.sin(pos_theta)  # [5,4]（无需重复）
        cos = torch.cos(pos_theta)  # [5,4]（无需重复）
        # 旋转计算
        x1 = x[..., ::2]  # 偶数位 #[batch, seq_len, 总维度/2]
        x2 = x[..., 1::2] # 奇数位
        # import pdb; pdb.set_trace()

        """
        把每个维度想象成 “嵌套盒子”，层级对应维度索引（从外到内：维度 0→维度 1→维度 2→维度 3）：
        x1 的盒子结构：
        外层有 2 个大盒子（batch=2）→ 每个大盒子里有 8 个中盒子（num_heads=8）→ 每个中盒子里有 5 个小盒子（seq_len=5）→ 每个小盒子里有 4 个小球（head_dim/2=4）。
        广播后 cos 的盒子结构：
        外层有 1 个大盒子 → 每个大盒子里有 1 个中盒子 → 每个中盒子里有 5 个小盒子 → 每个小盒子里有 4 个小球。
        
        广播的过程就是：
        把 cos 的 1 个大盒子复制 1 份，变成 2 个（匹配 x1 的 batch=2）；
        把每个大盒子里的 1 个中盒子复制 7 份，变成 8 个（匹配 x1 的 num_heads=8）；
        中盒子里的小盒子（5 个）、小盒子里的小球（4 个）完全不用动，因为数量和 x1 一致。
        最终，cos 的盒子结构和 x1 完全一样，每个小球都能和 x1 的小球一一对应，这就是 “完全匹配” 的本质。
        五、重要补充：广播是 “逻辑扩展”，不是 “物理复制”
        PyTorch 的广播不会真的复制数据（否则会浪费显存），而是在运算时临时逻辑上扩展，比如 cos 的[1,1,5,4]在和 x1 相乘时，会针对每个 batch 和 num_heads，重复使用相同的 cos 值（因为位置相关的 θ 不随 batch/head 变化），既保证运算正确，又节省内存。
        总结
        “广播后 cos 的维度 [1,1,5,4] 与 x1 的 [2,8,5,4] 完全匹配” 的完整含义是：
        从后往前逐维度对比，cos 的维度满足 “相等” 或 “为 1” 的广播条件；
        cos 中维度为 1 的位置（batch/num_heads 维度）会被逻辑扩展为 2 和 8；
        扩展后 cos 的形状与 x1 完全一致，因此可以逐元素相乘（每个位置的元素一一对应）。
        简单说：广播让 “形状看似不同” 的两个张量，变成了 “运算时形状完全相同” 的张量，这是 PyTorch 实现向量化运算的核心技巧。
                
        """
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot


if __name__ == "__main__":
    # ===================== 测试参数配置 =====================
    # 核心参数（确保dim是num_heads的整数倍，head_dim为偶数）
    DIM = 64  # Q/K总维度
    NUM_HEADS = 8  # 注意力头数
    BATCH_SIZE = 2  # 批次大小
    SEQ_LEN = 5  # 序列长度（token数量）
    logger.info(f"测试环境：设备={DEVICE}，PyTorch版本={torch.__version__}")

    # ===================== 步骤1：实例化DynamicRoPE =====================
    rope = DynamicRoPE(dim=DIM, num_heads=NUM_HEADS).to(DEVICE)
    # 验证初始化参数
    assert rope.head_dim == DIM // NUM_HEADS, "单头维度计算错误"
    assert rope.head_dim % 2 == 0, "单头维度必须为偶数（RoPE要求）"
    logger.info(f"✅ DynamicRoPE实例化成功，单头维度={rope.head_dim}")

    # ===================== 步骤2：构造测试输入 =====================
    # 构造位置索引pos（0到SEQ_LEN-1）
    pos = torch.arange(SEQ_LEN, device=DEVICE)  # 形状 [SEQ_LEN] = [5]
    logger.info(f"位置索引pos形状：{pos.shape}，值：{pos.tolist()}")

    # 构造测试Q向量（格式1：[batch_size, seq_len, dim]，Transformer标准输入）
    # x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM, device=DEVICE)  # 形状 [2,5,64]
    x = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, rope.head_dim, device=DEVICE)  # [2,8,5,8]

    logger.info(f"原始Q向量形状：{x.shape}")

    # ===================== 步骤3：调用forward执行RoPE =====================
    x_rot = rope(x, pos)
    # 验证输出维度与输入一致
    assert x_rot.shape == x.shape, "RoPE输出维度与输入不一致！"
    logger.info(f"✅ RoPE执行成功，旋转后Q向量形状：{x_rot.shape}")

    # ===================== 步骤4：验证RoPE核心特性（相对位置点积相等） =====================
    logger.info("\n===== 验证RoPE核心特性：相对位置相同，点积相等 =====")
    # 取第一个样本的Q向量（简化验证）
    x_rot_sample0 = x_rot[0]  # 形状 [SEQ_LEN, DIM] = [5,64]

