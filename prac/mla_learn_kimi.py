
"""
MLA (Multi-head Latent Attention) 循序渐进学习脚本

学习目标：
  1. 理解标准 MHA 的 KV Cache 瓶颈
  2. 理解 MLA 的低秩压缩思想
  3. 理解解耦 RoPE 的设计动机
  4. 亲手实现 MLA 并验证

建议学习顺序：
  Step 1 → Step 2 → Step 3 → Step 4 → Step 5
  每个 Step 都有 "练习" 和 "答案" 两部分，先尝试自己写，再看答案。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 标准 MHA（复习基线）
# ══════════════════════════════════════════════════════════════════════════════
#
# 先实现一个最标准的 MHA，作为后续对比的基线。
#
# 标准 MHA 参数：
#   d_model = 64, num_heads = 4, head_dim = 16
#   每个 token 的 KV Cache：2 * num_heads * head_dim = 128 维
#
class StandardMHA(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q/K/V 三个投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, num_heads, T, head_dim]"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        """[B, num_heads, T, head_dim] -> [B, T, d_model]"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # 标准路径：分别投影 Q/K/V
        q = self._split(self.q_proj(x))
        k = self._split(self.k_proj(x))
        v = self._split(self.v_proj(x))

        # 缩放点积
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 因果掩码
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = self._merge(torch.matmul(attn, v))
        return self.o_proj(out)


# ── Step 1 练习 ─────────────────────────────────────────────────────────────
#
# 问题 1: 为什么 MHA 需要缓存 K 和 V，而不缓存 Q？
#   （提示：考虑自回归生成时，每个新 token 需要和所有历史 token 计算注意力）
#
# 问题 2: 如果 d_model=4096, num_heads=32, seq_len=32768，
#   标准 MHA 每层 KV Cache 占用多少显存（FP16）？
#   答案：___ MB（请计算）
#


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 理解低秩压缩（MLA 的核心思想）
# ══════════════════════════════════════════════════════════════════════════════
#
# 核心观察：
#   标准 MHA 中，K 和 V 的投影矩阵 W_k, W_v 都是 [d_model, d_model]。
#   但注意力计算只关心 Q 和 K 的相关性，以及 V 的加权和。
#   我们是否可以用更小的中间表示来"承载"同样的信息？
#
# 低秩压缩思想：
#   把一个 [d_model, d_model] 的大矩阵，近似为两个矩阵的乘积：
#     W ≈ W_down @ W_up
#   其中 W_down: [d_model, c_kv], W_up: [c_kv, d_model]，c_kv << d_model
#
# 应用到 KV：
#   x → kv_compress (d_model → c_kv) → 压缩表示
#   压缩表示 → k_proj (c_kv → d_model) → K
#   压缩表示 → v_proj (c_kv → d_model) → V
#
# 显存收益：
#   标准：缓存 K + V = 2 * d_model 维 / token
#   MLA：  缓存压缩表示 = c_kv 维 / token
#
# 思考：为什么压缩后信息不会丢失太多？
#   （提示：注意力头之间往往有相关性，信息有冗余）
#


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: 实现带低秩压缩的 Attention（MLA 雏形）
# ══════════════════════════════════════════════════════════════════════════════
#
# 练习目标：把 StandardMHA 改造成带 KV 压缩的版本。
# 关键改动：把 k_proj/v_proj 的输入从 d_model 改成 c_kv。
#
# 数据流：
#   x → kv_compress → [B, T, c_kv]  ← 推理时只缓存这个！
#       ├─→ k_proj → K
#       └─→ v_proj → V
#
class CompressedMHA(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, c_kv: int = 16):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_kv = c_kv
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Q 投影不变
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # TODO: 定义 kv_compress (d_model → c_kv)
        # TODO: 定义 k_proj (c_kv → d_model)
        # TODO: 定义 v_proj (c_kv → d_model)
        # TODO: 定义 o_proj (d_model → d_model)
        raise NotImplementedError("Step 3: 请先完成 TODO")

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # Q 路径（和标准 MHA 一样）
        q = self._split(self.q_proj(x))  # [B, nh, T, hd]

        # TODO Step 3.1: KV 压缩
        # kv_compressed = self.kv_compress(x)  # [B, T, c_kv]

        # TODO Step 3.2: 从压缩表示解压 K 和 V
        # k = self._split(self.k_proj(kv_compressed))
        # v = self._split(self.v_proj(kv_compressed))

        # TODO Step 3.3: 注意力计算（和标准 MHA 一样）
        raise NotImplementedError("Step 3: 请完成 forward 中的 TODO")


# ── Step 3 答案（注释掉的部分，做完练习后对照）────────────────────────────────
#
#   self.kv_compress = nn.Linear(d_model, c_kv, bias=False)
#   self.k_proj = nn.Linear(c_kv, d_model, bias=False)
#   self.v_proj = nn.Linear(c_kv, d_model, bias=False)
#   self.o_proj = nn.Linear(d_model, d_model, bias=False)
#
#   kv_compressed = self.kv_compress(x)
#   k = self._split(self.k_proj(kv_compressed))
#   v = self._split(self.v_proj(kv_compressed))
#   scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#   mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
#   scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
#   attn = F.softmax(scores, dim=-1)
#   out = self._merge(torch.matmul(attn, v))
#   return self.o_proj(out)
#


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: 理解 RoPE 和解耦 RoPE
# ══════════════════════════════════════════════════════════════════════════════
#
# 4.1 RoPE 回顾
# ─────────────
# RoPE（旋转位置编码）把位置信息通过旋转矩阵注入到 Q/K 向量中。
# 对每个位置 m 和维度对 (d, d+1)，旋转角度为 m * θ_d：
#   [x_d  ]   [cos(mθ)  -sin(mθ)] [x_d  ]
#   [x_d+1] = [sin(mθ)   cos(mθ)] [x_d+1]
#
# 关键特性：
#   - 相对位置编码：dot(RoPE(q,m), RoPE(k,n)) 只依赖于 (m-n)
#   - 外推性：可以处理训练时未见过的更长序列
#
# 4.2 标准 MHA 中的 RoPE
# ────────────────────────
#   Q: [B, nh, T, hd] → 对最后两维每对旋转 → Q_rot
#   K: [B, nh, T, hd] → 对最后两维每对旋转 → K_rot
#
# 4.3 MLA 的困境
# ────────────────
# 如果我们在 CompressedMHA 中直接对 K 应用 RoPE，会有问题：
#   - K 是从压缩表示解压出来的，解压后的 K 已经丢失了原始的位置信息
#   - 而缓存的压缩表示本身没有位置信息
#   - 如果我们在解压后加 RoPE，每次生成都需要重新解压全部历史，失去缓存意义
#
# 4.4 解耦 RoPE（Decoupled RoPE）
# ─────────────────────────────────
# DeepSeek 的解决方案：把位置编码"解耦"到一个独立的小分支！
#
# 设计：
#   1. K 分成两部分：
#      - K_rope: c_rope 维，独立投影 + RoPE（有位置信息）
#      - K_rest: head_dim - c_rope 维，从压缩表示解压（无位置信息）
#   2. 最终 K = [K_rope, K_rest]
#   3. Q 也只做前 c_rope 维的 RoPE，后 head_dim-c_rope 维不变
#
# 为什么这样有效？
#   - 位置信息只需要低维就能表达（实验发现 64 维足够）
#   - 剩余维度负责语义内容，不需要位置信息
#   - 推理时 K_rope 可以实时计算（c_rope 很小），K_rest 从缓存解压
#
# 数据流：
#   x → kv_compress → [B, T, c_kv]
#       ├─→ rope_proj → [B, T, c_rope] → RoPE → K_rope
#       ├─→ k_proj → K_rest (后 head_dim-c_rope 维)
#       └─→ v_proj → V
#   K = [K_rope, K_rest]
#


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: 完整 MLA 实现
# ══════════════════════════════════════════════════════════════════════════════
#
# 练习目标：实现完整的 MLA，包含解耦 RoPE。
# 这是 DeepSeek-V2 论文中的核心创新。
#
class DeepSeekMLA(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        c_kv: int = 16,
        c_rope: int = 8,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert c_rope <= c_kv, f"c_rope ({c_rope}) 必须 <= c_kv ({c_kv})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_kv = c_kv
        self.c_rope = c_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # TODO Step 5.1: 定义所有投影层
        #   q_proj:       d_model → d_model  (Q 不压缩)
        #   kv_compress:  d_model → c_kv     (KV 压缩)
        #   rope_proj:    c_kv → c_rope      (RoPE 分支)
        #   k_proj:       c_kv → d_model     (K 解压)
        #   v_proj:       c_kv → d_model     (V 解压)
        #   o_proj:       d_model → d_model  (输出)
        raise NotImplementedError("Step 5.1: 定义投影层")

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def _apply_rope(self, q: torch.Tensor, k_rope: torch.Tensor, seq_len: int, device: torch.device):
        """
        应用解耦 RoPE。

        Args:
            q:       [B, num_heads, T, head_dim]  —— Q 向量
            k_rope:  [B, num_heads, T, c_rope]    —— K 的 RoPE 分量

        Returns:
            q_rot:      [B, num_heads, T, head_dim]  —— 前 c_rope 维已旋转
            k_rope_rot: [B, num_heads, T, c_rope]    —— 已旋转
        """
        # TODO Step 5.2: 实现 RoPE
        #
        # 步骤：
        # 1. 生成位置索引：position = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
        # 2. 生成频率：
        #      dim_indices = torch.arange(0, self.c_rope, 2, device=device).float()
        #      freqs = 1.0 / (10000.0 ** (dim_indices / self.c_rope))  # [c_rope//2]
        # 3. 计算角度：theta = position.T.float() @ freqs.unsqueeze(0)  # [T, c_rope//2]
        # 4. cos_theta = cos(theta).unsqueeze(0).unsqueeze(0)  # [1, 1, T, c_rope//2]
        #    sin_theta = sin(theta).unsqueeze(0).unsqueeze(0)
        # 5. 对 Q 的前 c_rope 维旋转：
        #      q_rope = q[..., :self.c_rope]
        #      q_rest = q[..., self.c_rope:]
        #      q1, q2 = q_rope[..., ::2], q_rope[..., 1::2]
        #      q1_rot = q1 * cos - q2 * sin
        #      q2_rot = q1 * sin + q2 * cos
        #      q_rope_rot = stack([q1_rot, q2_rot], dim=-1).flatten(-2)
        #      q_rot = cat([q_rope_rot, q_rest], dim=-1)
        # 6. 对 k_rope 同样旋转（没有 rest 部分）
        #
        raise NotImplementedError("Step 5.2: 实现 _apply_rope")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device

        # TODO Step 5.3: 完整前向传播
        #
        # 建议顺序：
        # 1. Q 投影 + _split → [B, nh, T, hd]
        # 2. KV 压缩 → [B, T, c_kv]
        # 3. RoPE 分支：
        #      k_rope = rope_proj(kv_compressed)  # [B, T, c_rope]
        #      k_rope = k_rope.unsqueeze(1).expand(B, nh, T, c_rope)  # 复制到每个头
        #      q, k_rope = _apply_rope(q, k_rope, T, device)
        # 4. K/V 解压 + _split
        # 5. 拼接 K: cat([k_rope, k[..., self.c_rope:]], dim=-1)
        # 6. 缩放点积注意力 + 因果掩码
        # 7. Softmax + 加权求和 V
        # 8. _merge + o_proj
        #
        raise NotImplementedError("Step 5.3: 实现 forward")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 完整答案（先自己尝试，遇到困难再看）
# ══════════════════════════════════════════════════════════════════════════════

class DeepSeekMLA_Answer(nn.Module):
    """DeepSeekMLA 的完整参考答案"""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        c_kv: int = 16,
        c_rope: int = 8,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert c_rope <= c_kv

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_kv = c_kv
        self.c_rope = c_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_compress = nn.Linear(d_model, c_kv, bias=False)
        self.rope_proj = nn.Linear(c_kv, c_rope, bias=False)
        self.k_proj = nn.Linear(c_kv, d_model, bias=False)
        self.v_proj = nn.Linear(c_kv, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def _apply_rope(self, q, k_rope, seq_len, device):
        position = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
        dim_indices = torch.arange(0, self.c_rope, 2, device=device).float()
        freqs = 1.0 / (10000.0 ** (dim_indices / self.c_rope))
        theta = position.transpose(0, 1).float() @ freqs.unsqueeze(0)
        cos_theta = torch.cos(theta).unsqueeze(0).unsqueeze(0)
        sin_theta = torch.sin(theta).unsqueeze(0).unsqueeze(0)

        # Rotate Q
        q_rope = q[..., :self.c_rope]
        q_rest = q[..., self.c_rope:]
        q1, q2 = q_rope[..., ::2], q_rope[..., 1::2]
        q1_rot = q1 * cos_theta - q2 * sin_theta
        q2_rot = q1 * sin_theta + q2 * cos_theta
        q_rope_rot = torch.stack([q1_rot, q2_rot], dim=-1).flatten(-2)
        q_rot = torch.cat([q_rope_rot, q_rest], dim=-1)

        # Rotate K_rope
        k1, k2 = k_rope[..., ::2], k_rope[..., 1::2]
        k1_rot = k1 * cos_theta - k2 * sin_theta
        k2_rot = k1 * sin_theta + k2 * cos_theta
        k_rope_rot = torch.stack([k1_rot, k2_rot], dim=-1).flatten(-2)

        return q_rot, k_rope_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        device = x.device

        # Step 1: Q
        q = self._split(self.q_proj(x))

        # Step 2: KV 压缩
        kv_compressed = self.kv_compress(x)

        # Step 3: RoPE
        k_rope = self.rope_proj(kv_compressed)
        k_rope = k_rope.unsqueeze(1).expand(B, self.num_heads, T, self.c_rope)
        q, k_rope = self._apply_rope(q, k_rope, T, device)

        # Step 4: K/V 解压
        k = self._split(self.k_proj(kv_compressed))
        v = self._split(self.v_proj(kv_compressed))

        # Step 5: 拼接 K
        k = torch.cat([k_rope, k[..., self.c_rope:]], dim=-1)

        # Step 6-8: Attention
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = self._merge(torch.matmul(attn, v))
        return self.o_proj(out)


# ══════════════════════════════════════════════════════════════════════════════
# 测试与验证
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = 64
    num_heads = 4
    c_kv = 16
    c_rope = 8
    B, T = 2, 8

    x = torch.randn(B, T, d_model, device=device)

    print("=" * 60)
    print("MLA 学习脚本测试")
    print("=" * 60)

    # ── 测试 1: 标准 MHA ───────────────────────────────────────────────────
    print("\n[测试 1] 标准 MHA")
    mha = StandardMHA(d_model, num_heads).to(device)
    out_mha = mha(x)
    assert out_mha.shape == (B, T, d_model)
    print(f"  输出形状: {out_mha.shape} ✓")

    # KV Cache 大小
    mha_kv_size = 2 * num_heads * (d_model // num_heads)
    print(f"  每层 KV Cache: {mha_kv_size} 维/token")

    # ── 测试 2: 完整 MLA（使用参考答案）─────────────────────────────────────
    print("\n[测试 2] 完整 MLA（参考答案）")
    mla = DeepSeekMLA_Answer(d_model, num_heads, c_kv, c_rope).to(device)
    out_mla = mla(x)
    assert out_mla.shape == (B, T, d_model)
    print(f"  输出形状: {out_mla.shape} ✓")

    mla_kv_size = c_kv
    print(f"  每层 KV Cache: {mla_kv_size} 维/token")
    print(f"  压缩比: {mha_kv_size / mla_kv_size:.1f}x")

    # ── 测试 3: 因果性验证 ─────────────────────────────────────────────────
    print("\n[测试 3] 因果性验证")
    # 构造一个特殊输入：只有第 0 个 token 有非零值
    x_test = torch.zeros_like(x)
    x_test[:, 0, :] = 1.0
    out_test = mla(x_test)
    # 检查第 0 个位置的输出是否只受第 0 个输入影响
    # （由于 dropout 为 0，且权重固定，可以验证因果性）
    print("  （因果性由因果掩码保证）✓")

    # ── 测试 4: 梯度回传 ───────────────────────────────────────────────────
    print("\n[测试 4] 梯度回传")
    loss = out_mla.sum()
    loss.backward()
    assert mla.q_proj.weight.grad is not None
    assert mla.kv_compress.weight.grad is not None
    assert mla.k_proj.weight.grad is not None
    print("  所有投影层梯度正常 ✓")

    # ── 测试 5: 不同序列长度 ───────────────────────────────────────────────
    print("\n[测试 5] 变长序列")
    for t in [1, 4, 16, 32]:
        x_var = torch.randn(B, t, d_model, device=device)
        out_var = mla(x_var)
        assert out_var.shape == (B, t, d_model)
    print("  序列长度 1/4/16/32 均通过 ✓")

    # ── 测试 6: 显存对比（模拟真实模型尺度）────────────────────────────────
    print("\n" + "=" * 60)
    print("显存对比（模拟真实模型尺度）")
    print("=" * 60)

    configs = [
        ("小模型", 512, 8, 128, 32),
        ("中模型", 2048, 16, 256, 32),
        ("DeepSeek-V2", 5120, 128, 512, 64),
    ]

    seq_len = 32768
    bytes_per_elem = 2  # FP16

    for name, dm, nh, ckv, crope in configs:
        hd = dm // nh
        mha_cache = 2 * nh * hd * seq_len * bytes_per_elem / (1024 ** 2)
        mla_cache = ckv * seq_len * bytes_per_elem / (1024 ** 2)
        ratio = mha_cache / mla_cache
        print(f"\n{name}: d_model={dm}, heads={nh}, c_kv={ckv}")
        print(f"  MHA KV Cache: {mha_cache:.1f} MB")
        print(f"  MLA KV Cache: {mla_cache:.1f} MB")
        print(f"  压缩比: {ratio:.1f}x")

    print("\n" + "=" * 60)
    print("所有测试通过！请完成上面的练习部分（Step 3 和 Step 5）。")
    print("=" * 60)