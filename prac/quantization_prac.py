
# #
# # """
# # FP32 → INT8 量化演示脚本
# # 用最简单的单层网络，手把手展示量化每一步的数学过程。
# # """
# # import sys
# #
# # import torch
# # import torch.nn as nn
# # import numpy as np
# #
# # # ═══════════════════════════════════════════════════════════════════
# # # 第 0 步：构造一个极简模型（单层 Linear）
# # # ═══════════════════════════════════════════════════════════════════
# #
# # torch.manual_seed(42)
# #
# #
# # class TinyModel(nn.Module):
# #     def __init__(self, in_features=4, out_features=3):
# #         super().__init__()
# #         self.fc = nn.Linear(in_features, out_features, bias=False)
# #         # 故意设一些好看的权重，方便观察
# #         with torch.no_grad():
# #             self.fc.weight.copy_(torch.tensor([
# #                 [0.50, -0.30, 1.20, -0.80],
# #                 [-1.50, 0.20, -0.10, 0.60],
# #                 [0.10, 0.90, -1.10, 0.40],
# #             ]))
# #
# #     def forward(self, x):
# #         return self.fc(x)
# #
# #
# # model = TinyModel()
# # model.eval()
# #
# # # 提取权重
# # W_fp32 = model.fc.weight.data.clone()  # [3, 4]
# # print("=" * 60)
# # print("【原始模型】FP32 权重矩阵 W")
# # print("=" * 60)
# # # print(W_fp32.tolist())
# # print(f"权重范围: [{W_fp32.min():.4f}, {W_fp32.max():.4f}]")
# # print(f"权重占用: {W_fp32.element_size() * W_fp32.numel()} bytes\n")
# #
# #
# # # ═══════════════════════════════════════════════════════════════════
# # # 第 1 步：计算 Scale（对称量化）
# # # ═══════════════════════════════════════════════════════════════════
# #
# # def compute_scale(tensor, bits=8):
# #     """对称量化：scale = max(|x|) / (2^(bits-1) - 1)"""
# #     alpha = torch.max(torch.abs(tensor.min()), torch.abs(tensor.max()))
# #     scale = alpha / (2 ** (bits - 1) - 1)
# #     return scale.item()
# #
# #
# # w_scale = compute_scale(W_fp32, bits=8)
# # print("=" * 60)
# # print("【Step 1】计算 Scale")
# # print("=" * 60)
# # print(f"  绝对值最大 alpha = {torch.max(torch.abs(W_fp32)).item():.4f}")
# # print(f"  INT8 正向边界 = 127")
# # print(f"  scale = alpha / 127 = {w_scale:.8f}")
# # print(f"  含义: FP32 中每 {w_scale:.6f} 对应 INT8 中的 1 个单位\n")
# #
# #
# # # ═══════════════════════════════════════════════════════════════════
# # # 第 2 步：量化 FP32 → INT8
# # # ═══════════════════════════════════════════════════════════════════
# #
# # def quantize_symmetric(tensor, scale):
# #     """
# #     量化过程：
# #       1. tensor / scale      → 映射到整数域
# #       2. round()             → 四舍五入到最近的整数格子
# #       3. clamp(-128, 127)    → 截断，防止溢出
# #       4. to(torch.int8)      → 转为 8 位整数存储
# #     """
# #     scaled = tensor / scale
# #     rounded = torch.round(scaled)
# #     clamped = torch.clamp(rounded, -128, 127)
# #     int8 = clamped.to(torch.int8)
# #     return int8, scaled, rounded, clamped
# #
# #
# # W_int8, W_scaled, W_rounded, W_clamped = quantize_symmetric(W_fp32, w_scale)
# # #
# # print("=" * 60)
# # print("【Step 2】量化: W_fp32 → W_int8")
# # print("=" * 60)
# #
# # # 逐元素展示量化过程
# # print("\n逐元素拆解（以第一行为例）:")
# # for i in range(4):
# #     orig = W_fp32[0, i].item()
# #     div = W_scaled[0, i].item()
# #     rnd = W_rounded[0, i].item()
# #     clp = W_clamped[0, i].item()
# #     final = W_int8[0, i].item()
# #     print(f"  {orig:>7.4f} / {w_scale:.6f} = {div:>8.2f} "
# #           f"→ round → {rnd:>6.0f} → clip → {clp:>4.0f} → int8 → {final:>4}")
# #
# # print(f"\n完整 INT8 权重矩阵:")
# # # print(W_int8.tolist())
# # print(f"INT8 范围: [{W_int8.min()}, {W_int8.max()}]")
# # print(f"INT8 占用: {W_int8.element_size() * W_int8.numel()} bytes (压缩了 4 倍)\n")
# #
# #
# # # ═══════════════════════════════════════════════════════════════════
# # # 第 3 步：反量化 INT8 → FP32（近似恢复）
# # # ═══════════════════════════════════════════════════════════════════
# #
# # def dequantize_symmetric(int8_tensor, scale):
# #     """反量化：int8 * scale"""
# #     return int8_tensor.float() * scale
# #
# #
# # W_deq = dequantize_symmetric(W_int8, w_scale)
# #
# # print("=" * 60)
# # print("【Step 3】反量化: W_int8 → Ŵ_fp32（近似值）")
# # print("=" * 60)
# # # print(W_deq.tolist())
# #
# # # 误差分析
# # abs_err = torch.abs(W_fp32 - W_deq)
# # rel_err = abs_err / (torch.abs(W_fp32) + 1e-12)
# #
# # print(f"\n误差分析:")
# # print(f"  理论最大误差 ≤ scale/2 = {w_scale / 2:.8f}")
# # print(f"  实际最大绝对误差: {abs_err.max().item():.8f}")
# # print(f"  实际平均绝对误差: {abs_err.mean().item():.8f}")
# # print(f"  实际最大相对误差: {rel_err.max().item():.2%}")
# #
# # # ═══════════════════════════════════════════════════════════════════
# # # 第 4 步：INT8 推理（模拟真实推理流程）
# # # ═══════════════════════════════════════════════════════════════════
# #
# # print("\n" + "=" * 60)
# # print("【Step 4】推理对比: FP32 vs INT8")
# # print("=" * 60)
# #
# # # 构造一个输入
# # x_fp32 = torch.tensor([[1.0, -0.5, 0.3, -1.2]])
# #
# # # 4.1 FP32 推理（黄金标准）
# # with torch.no_grad():
# #     y_fp32 = model(x_fp32)
# # print(f"\n输入 x: {x_fp32.tolist()}")
# # print(f"FP32 输出: {y_fp32.tolist()}")
# #
# # # 4.2 把输入也量化为 INT8
# # x_scale = compute_scale(x_fp32, bits=8)
# # x_int8 = quantize_symmetric(x_fp32, x_scale)[0]
# # print(f"\n输入量化:")
# # print(f"  x_scale = {x_scale:.6f}")
# # print(f"  x_int8 = {x_int8.tolist()}")
# #
# # # 4.3 INT8 矩阵乘（提升到 INT32 累加，防止溢出）
# # # 公式: Y ≈ (W_int8 @ X_int8) * (w_scale * x_scale)
# # print(f"\nINT8 矩阵乘过程:")
# # print(f"  W_int8 shape: {W_int8.shape}")
# # print(f"  x_int8 shape: {x_int8.shape}")
# #
# # # 矩阵乘：W_int8 [3,4] @ x_int8.T [4,1] = [3,1]
# # # 先提升到 int32 再做乘法（真实硬件中累加器是 int32）
# # y_int32 = torch.matmul(W_int8.to(torch.int32), x_int8.t().to(torch.int32))
# # print(f"  INT32 累加结果: {y_int32.flatten().tolist()}")
# #
# # # 反量化
# # combined_scale = w_scale * x_scale
# # y_int8 = y_int32.float() * combined_scale
# # print(f"  组合 scale = w_scale * x_scale = {combined_scale:.8f}")
# # print(f"INT8 输出: {y_int8.t().tolist()}")
# #
# # # 4.4 对比
# # diff = torch.abs(y_fp32 - y_int8.t())
# # print(f"\n输出误差: {diff.tolist()}")
# # print(f"平均误差: {diff.mean().item():.6f}")
# #
# # # ═══════════════════════════════════════════════════════════════════
# # # 第 5 步：位级可视化（理解为什么能压缩）
# # # ═══════════════════════════════════════════════════════════════════
# #
# # print("\n" + "=" * 60)
# # print("【Step 5】位级存储对比")
# # print("=" * 60)
# #
# #
# # def bits_of_float(f):
# #     """把 float32 转成 32 位二进制字符串"""
# #     import struct
# #     packed = struct.pack('>f', f)
# #     return ''.join(f'{b:08b}' for b in packed)
# #
# #
# # def bits_of_int8(i):
# #     """把 int8 转成 8 位二进制（补码）"""
# #     if i < 0:
# #         i = i & 0xFF
# #     return f'{i:08b}'
# #
# #
# # # 取第一个权重做对比
# # w0_fp32 = W_fp32[0, 0].item()
# # w0_int8 = W_int8[0, 0].item()
# #
# # print(f"\n以权重 W[0,0] = {w0_fp32:.4f} 为例:")
# # print(f"  FP32 存储: {bits_of_float(w0_fp32)}  (32 bit = 4 bytes)")
# # print(f"  INT8 存储: {bits_of_int8(w0_int8)}          (8 bit = 1 byte)")
# # print(f"  压缩比: 4:1")
# #
# # print(f"\n完整统计:")
# # print(f"  FP32 总字节: {W_fp32.element_size() * W_fp32.numel()}")
# # print(f"  INT8 总字节: {W_int8.element_size() * W_int8.numel()}")
# # print(f"  节省: {(1 - (W_int8.element_size() * W_int8.numel()) / (W_fp32.element_size() * W_fp32.numel())) * 100:.0f}%")
# #
# # # ═══════════════════════════════════════════════════════════════════
# # # 第 6 步：直观理解量化步长
# # # ═══════════════════════════════════════════════════════════════════
# #
# # print("\n" + "=" * 60)
# # print("【Step 6】量化步长可视化")
# # print("=" * 60)
# # print(f"scale = {w_scale:.6f} 就是相邻两个 INT8 格子之间的 FP32 距离")
# # print(f"INT8 的 256 个格子，每个格子代表 {w_scale:.6f} 的 FP32 区间")
# #
# # # 画几个格子
# # print(f"\nINT8 值  →  对应的 FP32 区间（中点）")
# # for q in [-2, -1, 0, 1, 2]:
# #     center = q * w_scale
# #     low = (q - 0.5) * w_scale
# #     high = (q + 0.5) * w_scale
# #     print(f"  {q:>4}  →  [{low:>10.6f}, {high:>10.6f}]  中点: {center:.6f}")



# """
# 完整量化流程演示：一个只有 Linear 层的小网络
# FP32 推理 vs INT8 量化推理，逐层对比
# """
#
# import torch
# import torch.nn as nn
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 1. 定义极简模型（两层 Linear，无 bias，方便观察）
# # ═══════════════════════════════════════════════════════════════════
#
# class TinyNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(4, 3, bias=False)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(3, 2, bias=False)
#
#         # 手工填入好看的权重，方便肉眼验证
#         with torch.no_grad():
#             self.fc1.weight.copy_(torch.tensor([
#                 [ 0.50, -0.30,  1.20, -0.80],
#                 [-1.50,  0.20, -0.10,  0.60],
#                 [ 0.10,  0.90, -1.10,  0.40],
#             ]))
#             self.fc2.weight.copy_(torch.tensor([
#                 [ 0.20, -0.50,  0.30],
#                 [-0.40,  0.10, -0.20],
#             ]))
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x
#
#
# model = TinyNet()
# model.eval()
#
# # ═══════════════════════════════════════════════════════════════════
# # 2. 量化工具函数
# # ═══════════════════════════════════════════════════════════════════
#
# def compute_scale(tensor, bits=8):
#     """对称量化 scale = max(|x|) / (2^(bits-1) - 1)"""
#     alpha = torch.max(torch.abs(tensor.min()), torch.abs(tensor.max()))
#     return (alpha / (2 ** (bits - 1) - 1)).item()
#
#
# def quantize_symmetric(tensor, scale):
#     """FP32 -> INT8：除以 scale → round → clamp → int8"""
#     q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
#     return q
#
#
# def dequantize_symmetric(q, scale):
#     """INT8 -> FP32：int8 转 float 后乘 scale"""
#     return q.float() * scale
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 3. FP32 推理（黄金标准）
# # ═══════════════════════════════════════════════════════════════════
#
# x = torch.tensor([[1.0, -0.5, 0.3, -1.2],
#                   [0.2,  0.8, -0.1, 0.5]])  # [batch=2, in=4]
#
# print("=" * 60)
# print("【FP32 推理】黄金标准")
# print("=" * 60)
#
# with torch.no_grad():
#     # 拆开每一步，方便对比
#     h0_fp32 = x                                 # 输入
#     h1_fp32 = model.fc1(h0_fp32)                # 第一层线性
#     h1_relu_fp32 = model.relu(h1_fp32)          # ReLU
#     y_fp32 = model.fc2(h1_relu_fp32)            # 第二层线性
#
# print(f"输入 x:\n{h0_fp32.tolist()}")
# print(f"\n第一层输出 (fc1):\n{h1_fp32.tolist()}")
# print(f"\nReLU 后:\n{h1_relu_fp32.tolist()}")
# print(f"\n最终输出 y (FP32):\n{y_fp32.tolist()}")
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 4. 量化权重（离线完成，模型发布后权重就固定为 INT8）
# # ═══════════════════════════════════════════════════════════════════
#
# print("\n" + "=" * 60)
# print("【量化权重】把模型权重永久转成 INT8")
# print("=" * 60)
#
# w1_scale = compute_scale(model.fc1.weight.data, bits=8)
# w1_int8 = quantize_symmetric(model.fc1.weight.data, w1_scale)
#
# w2_scale = compute_scale(model.fc2.weight.data, bits=8)
# w2_int8 = quantize_symmetric(model.fc2.weight.data, w2_scale)
#
# print(f"fc1 权重 scale = {w1_scale:.8f}")
# print(f"fc1 权重 INT8:\n{w1_int8.tolist()}")
#
# print(f"\nfc2 权重 scale = {w2_scale:.8f}")
# print(f"fc2 权重 INT8:\n{w2_int8.tolist()}")
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 5. INT8 推理（模拟真实推理流程）
# # ═══════════════════════════════════════════════════════════════════
#
# print("\n" + "=" * 60)
# print("【INT8 推理】逐层模拟")
# print("=" * 60)
#
#
# def int8_linear(x_fp32, w_int8, w_scale):
#     """
#     模拟 INT8 推理的 Linear 层：
#       1. 把输入激活量化为 INT8
#       2. INT8 权重 × INT8 激活，累加到 INT32
#       3. 反量化回 FP32
#     """
#     # 1. 输入也量化为 INT8（真实推理中每层输入都要量化）
#     x_scale = compute_scale(x_fp32, bits=8)
#     x_int8 = quantize_symmetric(x_fp32, x_scale)
#
#     # 2. 矩阵乘：提升到 INT32，防止溢出
#        y_int32 = W_int8 @ x_int8.T
#     y_int32 = torch.matmul(
#         w_int8.to(torch.int32),
#         x_int8.t().to(torch.int32)
#     )  # shape: [d_out, batch]
#
#     # 3. 反量化：乘回 (w_scale * x_scale)
#     y_fp32 = y_int32.float() * (w_scale * x_scale)
#
#     # 4. 转置回 [batch, d_out]
#     return y_fp32.t(), x_scale
#
#
# # --- 第一层：fc1 ---
# h1_int8, x_scale = int8_linear(h0_fp32, w1_int8, w1_scale)
# print(f"输入 scale = {x_scale:.8f}")
# print(f"fc1 INT8 输出（反量化后）:\n{h1_int8.tolist()}")
#
# # ReLU（在 FP32 域做，因为 ReLU 不是线性操作，不能直接在 INT8 做）
# h1_relu_int8 = torch.relu(h1_int8)
# print(f"\nReLU 后:\n{h1_relu_int8.tolist()}")
#
# # --- 第二层：fc2 ---
# y_int8, h1_scale = int8_linear(h1_relu_int8, w2_int8, w2_scale)
# print(f"中间激活 scale = {h1_scale:.8f}")
# print(f"\n最终输出 y (INT8 推理):\n{y_int8.tolist()}")
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 6. 结果对比
# # ═══════════════════════════════════════════════════════════════════
#
# print("\n" + "=" * 60)
# print("【结果对比】FP32 vs INT8")
# print("=" * 60)
#
# diff = torch.abs(y_fp32 - y_int8)
# print(f"FP32 输出:\n{y_fp32.tolist()}")
# print(f"INT8 输出:\n{y_int8.tolist()}")
# print(f"绝对误差:\n{diff.tolist()}")
# print(f"平均误差: {diff.mean().item():.6f}")
# print(f"最大误差: {diff.max().item():.6f}")
#
#
# # ═══════════════════════════════════════════════════════════════════
# # 7. 空间对比
# # ═══════════════════════════════════════════════════════════════════
#
# print("\n" + "=" * 60)
# print("【空间对比】")
# print("=" * 60)
#
# # 原始 FP32 模型
# w1_fp32_bytes = model.fc1.weight.element_size() * model.fc1.weight.numel()
# w2_fp32_bytes = model.fc2.weight.element_size() * model.fc2.weight.numel()
# total_fp32 = w1_fp32_bytes + w2_fp32_bytes
#
# # 量化后 INT8 模型
# w1_int8_bytes = w1_int8.element_size() * w1_int8.numel()
# w2_int8_bytes = w2_int8.element_size() * w2_int8.numel()
# total_int8 = w1_int8_bytes + w2_int8_bytes
#
# print(f"fc1 权重: {w1_fp32_bytes} bytes → {w1_int8_bytes} bytes")
# print(f"fc2 权重: {w2_fp32_bytes} bytes → {w2_int8_bytes} bytes")
# print(f"总权重:   {total_fp32} bytes → {total_int8} bytes")
# print(f"压缩率:   {total_fp32 / total_int8:.0f}x")
#
#





"""
FP32 → INT8 量化完整流程练习
目标：亲手实现一个微型网络的量化推理，理解每一步的数学原理。

建议顺序：模块 1 → 模块 2 → 模块 3 → 模块 4 → 运行测试
"""

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════
# 模块 0：极简模型（已提供，无需修改）
# ═══════════════════════════════════════════════════════════════════
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=False)

        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([
                [ 0.50, -0.30,  1.20, -0.80],
                [-1.50,  0.20, -0.10,  0.60],
                [ 0.10,  0.90, -1.10,  0.40],
            ]))
            self.fc2.weight.copy_(torch.tensor([
                [ 0.20, -0.50,  0.30],
                [-0.40,  0.10, -0.20],
            ]))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════════
# 模块 1：计算 Scale（对称量化）
# 难度：★☆☆☆☆
# ═══════════════════════════════════════════════════════════════════
#
# 对称量化公式：
#   alpha = max( |tensor_min|, |tensor_max| )
#   scale = alpha / (2^(bits-1) - 1)
#
# 当 bits=8 时，分母 = 127
def compute_scale(tensor: torch.Tensor, bits: int = 8) -> float:
    """
    计算对称量化的 scale。

    Args:
        tensor: 任意形状的 FP32 张量
        bits:   量化位宽（本练习中固定为 8）
    Returns:
        scale: 正浮点数
    """
    # TODO: 先求 tensor 的最小值和最大值的绝对值，取较大的那个作为 alpha
    # 提示：tensor.min().abs() 和 tensor.max().abs()
    alpha = torch.max(tensor.min().abs(), tensor.max().abs())

    # TODO: 计算 scale = alpha / (2^(bits-1) - 1)
    # 当 bits=8 时，就是 alpha / 127
    scale = alpha / (2**(bits-1) -1)
    return scale.item()


# ═══════════════════════════════════════════════════════════════════
# 模块 2：FP32 → INT8 量化
# 难度：★★☆☆☆
# ═══════════════════════════════════════════════════════════════════
#
# 量化四步走：
#   1. tensor / scale          → 把 FP32 映射到整数域
#   2. torch.round(...)       → 四舍五入到最近的整数格子
#   3. torch.clamp(..., -128, 127) → 截断，防止溢出
#   4. .to(torch.int8)         → 转为 8 位整数存储
def quantize_int8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """
    FP32 张量 → INT8 张量（对称量化）。

    Args:
        tensor: FP32 张量
        scale:  由 compute_scale 计算得到的 scale
    Returns:
        INT8 张量，形状与输入相同
    """
    # TODO: 依次完成 "除 scale → round → clamp → 转 int8" 四步
    # 提示：注意使用 torch.round 而不是 Python 内置 round
    q = torch.clamp(torch.round(tensor / scale), -128,127).to(torch.int8)
    return q


# ═══════════════════════════════════════════════════════════════════
# 模块 3：INT8 → FP32 反量化
# 难度：★☆☆☆☆
# ═══════════════════════════════════════════════════════════════════
#
# 反量化公式：
#   x_hat = q.float() * scale
def dequantize_int8(q: torch.Tensor, scale: float) -> torch.Tensor:
    """
    INT8 张量 → FP32 张量。

    Args:
        q:     INT8 张量
        scale: 量化时使用的同一个 scale
    Returns:
        FP32 张量
    """
    # TODO: 把 q 转成 float，再乘以 scale
    x_hat = q.float() * scale
    return x_hat


# ═══════════════════════════════════════════════════════════════════
# 模块 4：INT8 线性层推理
# 难度：★★★☆☆
# ═══════════════════════════════════════════════════════════════════
#
# 真实推理流程：
#   1. 输入激活量化为 INT8（用输入自己的 scale）
#   2. 权重(INT8) × 激活(INT8)，累加到 INT32（防止溢出）
#   3. 反量化回 FP32：乘回 (w_scale * x_scale)
#   4. 转置维度为 [batch, d_out]
#
# 公式：
#   Y_fp32 ≈ (W_int8 @ X_int8.T) * (w_scale * x_scale)
def int8_linear(
    x_fp32: torch.Tensor,
    w_int8: torch.Tensor,
    w_scale: float,
) -> torch.Tensor:
    """
    用 INT8 权重做一次线性层前向（无偏置）。

    Args:
        x_fp32:  输入激活，FP32，形状 [batch, d_in]
        w_int8:  量化后的权重，INT8，形状 [d_out, d_in]
        w_scale: 权重的 scale
    Returns:
        输出 FP32，形状 [batch, d_out]
    """
    # TODO Step 1: 计算输入激活的 scale，并把 x_fp32 量化为 INT8
    # 提示：直接调用上面写好的 compute_scale 和 quantize_int8
    x_scale = compute_scale(x_fp32, bits=8)

    x_int8 = quantize_int8(x_fp32, x_scale)

    # TODO Step 2: 矩阵乘
    # 提示：
    #   - w_int8 形状是 [d_out, d_in]
    #   - x_int8 形状是 [batch, d_in]，需要转置为 [d_in, batch]
    #   - 先把两者都提升到 torch.int32，再做 torch.matmul
    #   - 结果形状应为 [d_out, batch]
    y_int32 = torch.matmul(w_int8.to(torch.int32), x_int8.t().to(torch.int32))


    # TODO Step 3: 反量化
    # 提示：y_int32 先转 float，再乘以 (w_scale * x_scale)
    y_fp32 = y_int32.float() * (w_scale * x_scale)
    # TODO Step 4: 转置
    # 提示：当前 y_fp32 是 [d_out, batch]，需要 .t() 变成 [batch, d_out]
    # y_fp32 = y_fp32.t()

    return y_fp32.t()


# ═══════════════════════════════════════════════════════════════════
# 测试（全部通过 = 实现正确）
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    model = TinyNet()
    model.eval()

    # 构造输入 [batch=2, in=4]
    x = torch.tensor([[1.0, -0.5, 0.3, -1.2],
                      [0.2,  0.8, -0.1, 0.5]])

    # 提取权重
    w1_fp32 = model.fc1.weight.data.clone()
    w2_fp32 = model.fc2.weight.data.clone()

    print("=" * 60)
    print("量化练习测试")
    print("=" * 60)

    # ── 测试 1：compute_scale ─────────────────────────────────────
    print("\n[1] compute_scale")
    s1 = compute_scale(w1_fp32, bits=8)
    s2 = compute_scale(w2_fp32, bits=8)
    # import pdb;pdb.set_trace()
    assert isinstance(s1, float) and s1 > 0, "scale 应为正浮点数"
    assert isinstance(s2, float) and s2 > 0, "scale 应为正浮点数"

    # 手动验算：w1 最大绝对值是 1.5，所以 scale 应该 ≈ 1.5/127
    expected_s1 = 1.5 / 127.0
    assert abs(s1 - expected_s1) < 1e-6, f"fc1 scale 计算错误，期望 {expected_s1}，得到 {s1}"
    print(f"  通过：fc1 scale={s1:.8f}, fc2 scale={s2:.8f}")

    # ── 测试 2：quantize_int8 ─────────────────────────────────────
    print("\n[2] quantize_int8")
    w1_int8 = quantize_int8(w1_fp32, s1)
    w2_int8 = quantize_int8(w2_fp32, s2)

    assert w1_int8.dtype == torch.int8, f"期望 int8，得到 {w1_int8.dtype}"
    assert w1_int8.shape == w1_fp32.shape, "量化后形状不能变"
    assert w1_int8.min().item() >= -128 and w1_int8.max().item() <= 127, "超出 INT8 范围"

    # 手动抽查：0.5 / s1 ≈ 42.33，round 后应为 42
    assert w1_int8[0, 0].item() == 42, f"w1[0,0] 量化结果应为 42，得到 {w1_int8[0, 0].item()}"
    print(f"  通过：w1 量化范围 [{w1_int8.min()}, {w1_int8.max()}]，抽查 w1[0,0]={w1_int8[0,0]}")

    # ── 测试 3：dequantize_int8 ────────────────────────────────────
    print("\n[3] dequantize_int8")
    w1_deq = dequantize_int8(w1_int8, s1)

    assert w1_deq.dtype == torch.float32, f"反量化后应为 float32，得到 {w1_deq.dtype}"
    rmse = torch.sqrt(torch.mean((w1_fp32 - w1_deq) ** 2)).item()
    assert rmse < s1, f"RMSE 应小于 1 个 scale 步长，得到 {rmse:.6f}"
    print(f"  通过：反量化 RMSE={rmse:.6f}（< scale={s1:.6f}）")

    # ── 测试 4：int8_linear（核心）────────────────────────────────
    print("\n[4] int8_linear")
    with torch.no_grad():
        h1_fp32 = model.fc1(x)          # [2, 3]
        h1_relu = torch.relu(h1_fp32)
        y_fp32 = model.fc2(h1_relu)     # [2, 2]

    # 用你写的 int8_linear 模拟
    h1_int8 = int8_linear(x, w1_int8, s1)
    assert h1_int8.shape == h1_fp32.shape, f"fc1 输出形状错误：{h1_int8.shape} vs {h1_fp32.shape}"

    h1_int8_relu = torch.relu(h1_int8)
    y_int8 = int8_linear(h1_int8_relu, w2_int8, s2)
    assert y_int8.shape == y_fp32.shape, f"fc2 输出形状错误：{y_int8.shape} vs {y_fp32.shape}"

    mean_err = (y_fp32 - y_int8).abs().mean().item()
    assert mean_err < 0.1, f"INT8 推理误差过大：{mean_err:.4f}"
    print(f"  通过：INT8 推理平均误差={mean_err:.6f}")

    # ── 测试 5：空间对比 ─────────────────────────────────────────
    print("\n[5] 空间对比")
    fp32_bytes = w1_fp32.element_size() * w1_fp32.numel() + w2_fp32.element_size() * w2_fp32.numel()
    int8_bytes = w1_int8.element_size() * w1_int8.numel() + w2_int8.element_size() * w2_int8.numel()



    assert fp32_bytes == 72, f"FP32 总字节应为 72，得到 {fp32_bytes}"
    assert int8_bytes == 18, f"INT8 总字节应为 18，得到 {int8_bytes}"
    assert fp32_bytes == int8_bytes * 4, "压缩率应为 4x"

    print(f"  通过：FP32 {fp32_bytes} bytes → INT8 {int8_bytes} bytes（{fp32_bytes // int8_bytes}x）")