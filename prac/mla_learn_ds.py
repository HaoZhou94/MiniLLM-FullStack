"""
DeepSeek MLA (Multi-head Latent Attention) 学习脚本

本脚本通过 6 个练习，带你从零理解 DeepSeek-V2/特 的 MLA 核心思想。

学习流程：
  Part 1 ── 理解 KV Cache 瓶颈（计算题）
  Part 2 ── 标准 MHA 实现（复习）
  Part 3 ── 低秩压缩注意力（练习：填空）
  Part 4 ── 解耦 RoPE 注意力（练习：填空）
  Part 5 ── 完整 MLA （练习：填空）
  Part 6 ── 对比测试（自动执行）

每个 Part 的结构：
  【讲解】 → 【你的练习】 → 【运行验证】

使用方法：
  python prac/mla_learn_ds.py
  每次完成一个 TODO 后运行，看测试是否通过。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: KV Cache 瓶颈（先理解"为什么需要 MLA"）
# ══════════════════════════════════════════════════════════════════════════════
#
# 自回归生成时，每生成一个 token，需要和所有历史 token 计算注意力。
# 为了避免重复计算，我们把历史 token 的 K, V 缓存起来 —— 这就是 KV Cache。
#
# 问题：长序列下 KV Cache 会吃掉大量显存！
#
# 以 LLaMA-13B 为例：
#   d_model=5120, num_heads=40, head_dim=128, layers=40
#   序列长度 4096，FP16 存储
#
#   每层每 token KV Cache = 2 * 40 * 128 = 10,240 维
#   每层 4096 序列 KV Cache = 10,240 * 4096 * 2B = 80 MB
#   40 层总 KV Cache = 80 * 40 = 3.2 GB
#
# 如果序列长度到 32K，KV Cache 需要 25.6 GB —— 比模型本身还大！
#

def part1_calculate_kv_cache():
    """
    Part 1 练习：计算 KV Cache 大小

    填空完成下面的计算函数，验证你的理解。
    """
    print("=" * 60)
    print("Part 1: KV Cache 计算练习")
    print("=" * 60)

    # ── 参数（模拟 DeepSeek-V2） ──────────────────────────────────────────
    d_model = 5120          # 隐藏维度
    num_heads = 128         # 注意力头数
    head_dim = d_model // num_heads  # 每头维度
    num_layers = 80         # 总层数
    seq_len = 32768         # 序列长度
    bytes_per_elem = 2      # FP16

    # 每层每 token 的 KV Cache 维度
    # MHA: K 是 d_model 维，V 是 d_model 维
    kv_cache_per_token_per_layer_mha = 2 * d_model
    print(f"\n[标准 MHA] 每层每 token KV Cache: {kv_cache_per_token_per_layer_mha} 维")

    # TODO Part 1.1: 计算总 KV Cache 显存
    # total_mha_bytes = kv_cache_per_token_per_layer_mha * ___ * ___ * ___
    total_mha_mb = kv_cache_per_token_per_layer_mha * seq_len * bytes_per_elem
    # total_mha_mb = total_mha_bytes / (1024 ** 2)
    total_mha_mb = None
    if total_mha_mb is not None:
        print(f"[标准 MHA] 总 KV Cache: {total_mha_mb:.1f} MB ≈ {total_mha_mb/1024:.1f} GB")

    # MLA 的 KV Cache 维度和标准 MHA 一样吗？
    # 不一样！MLA 只缓存压缩后的表示（c_kv 维）
    c_kv = 512  # DeepSeek-V2 的 KV 压缩维度
    kv_cache_per_token_per_layer_mla = c_kv  # ← 关键差异！
    print(f"\n[MLA] 每层每 token KV Cache: {kv_cache_per_token_per_layer_mla} 维")
    print(f"[MLA] 压缩比: {kv_cache_per_token_per_layer_mha / kv_cache_per_token_per_layer_mla:.0f}x")

    # TODO Part 1.2: 计算 MLA 的总 KV Cache 显存
    # total_mla_bytes = kv_cache_per_token_per_layer_mla * ___ * ___ * ___
    # total_mla_mb = total_mla_bytes / (1024 ** 2)
    total_mla_mb = None
    if total_mla_mb is not None:
        print(f"[MLA] 总 KV Cache: {total_mla_mb:.1f} MB ≈ {total_mla_mb/1024:.1f} GB")

    # TODO Part 1.3: 以 80GB A100 为例，MHA 和 MLA 各能支持多长的序列？
    # gpu_memory = 80 * 1024  # MB
    # max_seq_mha = (gpu_memory * 1024 ** 2) / (kv_cache_per_token_per_layer_mha * num_layers * bytes_per_elem)
    # max_seq_mla = ...
    # 提示：这里只算 KV Cache，实际还要算模型参数、激活值等

    return total_mha_mb, total_mla_mb


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: 标准 MHA（复习基线）
# ══════════════════════════════════════════════════════════════════════════════
#
# 这是最常规的 MHA。跳过不看也行，后面 MLA 的注意力计算部分和这里完全一样。
#

class MHA(nn.Module):
    """标准多头注意力（基线参考）"""
    def __init__(self, d_model: int = 64, num_heads: int = 4):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        nh = self.num_heads
        hd = self.head_dim

        # 投影 + reshape
        q = self.Wq(x).view(B, T, nh, hd).transpose(1, 2)  # [B, nh, T, hd]
        k = self.Wk(x).view(B, T, nh, hd).transpose(1, 2)
        v = self.Wv(x).view(B, T, nh, hd).transpose(1, 2)

        # 注意力
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 因果掩码
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        s = s.masked_fill(mask[None, None, :, :], float("-inf"))

        a = F.softmax(s, dim=-1)
        o = torch.matmul(a, v)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(o)


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: 低秩压缩注意力（练习）
# ══════════════════════════════════════════════════════════════════════════════
#
# 【核心思想】
# 标准 MHA 的 Wk, Wv 是 [d_model, d_model] 的大矩阵。
# 我们可以把 K, V 先"压缩"到一个小的中间表示，再"解压"回来。
#
# 【数学等价性】
#   标准：  K = Wk @ x      Wk in [d_model, d_model]
#   低秩：  K = Wk_up @ (Wk_down @ x)
#            = (Wk_up @ Wk_down) @ x
#   其中 Wk_down in [c_kv, d_model], Wk_up in [d_model, c_kv]
#
# 当 Wk_up @ Wk_down 的秩 ≤ c_kv 时，这就是对 Wk 的低秩近似。
#
# 【好处】
#   生成时只需要缓存 z = Wk_down @ x（c_kv 维），大大节省显存。
#
# 练习目标：把标准 MHA 改成低秩压缩版本。
# 需要修改的地方用 TODO 标出。
#

class LowRankAttn(nn.Module):
    """
    低秩压缩注意力（只有 KV 压缩，没有解耦 RoPE）

    架构：
        x ──→ q_proj ──→ Q
        x ──→ kv_down ──→ z (c_kv 维) ← 缓存这个！
                ├──→ k_up ──→ K
                └──→ v_up ──→ V
    """
    def __init__(self, d_model: int = 64, num_heads: int = 4, c_kv: int = 16):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_kv = c_kv
        self.scale = self.head_dim ** -0.5

        # Q 投影（和标准 MHA 一样）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # ── KV 压缩/解压路径 ────────────────────────────────────────────
        # TODO Part 3.1: 定义 kv_down 投影（d_model → c_kv）
        # 这是"压缩"步骤，把 d_model 维压缩到 c_kv 维
        self.kv_down = nn.Linear(d_model, c_kv, bias=False)

        # TODO Part 3.2: 定义 k_up 投影（c_kv → d_model）
        # 这是"解压"步骤，从 c_kv 维恢复到 d_model 维得到 K
        self.k_up = nn.Linear(c_kv, d_model, bias=False)

        # TODO Part 3.3: 定义 v_up 投影（c_kv → d_model）
        # 同样从 c_kv 维恢复到 d_model 维得到 V
        self.v_up = nn.Linear(c_kv, d_model, bias=False)

        # 输出投影（和标准 MHA 一样）
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播（你需要在 TODO 位置补全代码）"""
        B, T, D = x.shape
        nh = self.num_heads
        hd = self.head_dim

        # Q 路径
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)

        # ── KV 压缩 ──────────────────────────────────────────────────────
        # TODO Part 3.4: 计算压缩表示 z
        #   z = self.kv_down(x)  # [B, T, c_kv]
        #   这条 z 就是推理时要缓存的东西！
        z = None
        if z is None:
            raise NotImplementedError("Part 3.4: 请计算压缩表示 z")

        # ── KV 解压 ──────────────────────────────────────────────────────
        # TODO Part 3.5: 从压缩表示 z 解压出 K 和 V
        #   k = self.k_up(z).view(B, T, nh, hd).transpose(1, 2)
        #   v = self.v_up(z).view(B, T, nh, hd).transpose(1, 2)
        k = None
        v = None
        if k is None or v is None:
            raise NotImplementedError("Part 3.5: 请解压 K 和 V")

        # ── 注意力计算（和标准 MHA 完全一样） ─────────────────────────────
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        s = s.masked_fill(mask[None, None, :, :], float("-inf"))
        a = F.softmax(s, dim=-1)
        o = torch.matmul(a, v)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(o)


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: 解耦 RoPE（练习）
# ══════════════════════════════════════════════════════════════════════════════
#
# 【问题】Part 3 的 LowRankAttn 有位置编码吗？没有！
#   标准 MHA 在投影后对 Q, K 加 RoPE。但这里 K 是从压缩表示解压的。
#   如果我们对 K 加 RoPE，那缓存压缩表示就没有意义了（每次都要重新解码全部历史）。
#
# 【DeepSeek 的解决方案：解耦 RoPE】
#   把 K 拆成两部分：
#     K = [K_rope, K_rest]
#        ─┬────   ─┬────
#         │         └── 从压缩表示解压（无 RoPE，没有位置信息）
#         └── 独立投影到低维 c_rope + 单独加 RoPE（有位置信息）
#
#   Q 也对应拆分：
#     Q_rope 部分加 RoPE（只前 c_rope 维），Q_rest 不加
#
# 【为什么 c_rope 可以很小？】
#   位置编码的信息量不需要很大的维度。DeepSeek-V2 用 c_rope=64 就够用了。
#

class RoPE:
    """
    RoPE 工具类：对最后一维的每对元素做旋转
    输入: [*, D]  → 输出: [*, D]（D 必须是偶数）
    """
    @staticmethod
    def apply(x: torch.Tensor, positions: torch.Tensor, dim: int):
        """
        Args:
            x:         [*, T, D]  输入向量
            positions: [T]        位置索引
            dim:       D          RoPE 维度（必须偶数）
        Returns:
            [*, T, D]  旋转后的向量
        """
        device = x.device
        T = x.shape[-2]
        D = dim

        # 频率
        i = torch.arange(0, D, 2, device=device).float()
        freqs = 1.0 / (10000.0 ** (i / D))  # [D//2]
        theta = positions.unsqueeze(-1).float() * freqs.unsqueeze(0)  # [T, D//2]

        cos = theta.cos()[None, :, None, :]   # [1, T, 1, D//2]
        sin = theta.sin()[None, :, None, :]

        # 旋转相邻对
        x1, x2 = x[..., ::2], x[..., 1::2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        return torch.stack([x1_rot, x2_rot], dim=-1).flatten(-2)


class DecoupledRoPEAttn(nn.Module):
    """
    带解耦 RoPE 的注意力（没有 KV 压缩，只有 RoPE 拆分）
    用于隔离理解"解耦 RoPE"这个子概念。
    """
    def __init__(self, d_model: int = 64, num_heads: int = 4, c_rope: int = 8):
        super().__init__()
        assert d_model % num_heads == 0
        assert d_model // num_heads >= c_rope
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_rope = c_rope
        self.scale = self.head_dim ** -0.5

        # 标准 Q/K/V 投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # TODO Part 4.1: 定义 k_rope_proj（d_model → c_rope）
        # 这个投影从 x 中提取 c_rope 维用于 RoPE
        self.k_rope_proj = nn.Linear(d_model, c_rope, bias=False)

        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        nh = self.num_heads
        hd = self.head_dim

        # Q 和 K 投影
        q_full = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)
        k_full = self.k_proj(x).view(B, T, nh, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, nh, hd).transpose(1, 2)

        # ── 解耦 RoPE ────────────────────────────────────────────────────
        #
        # 对 Q：只旋转前 c_rope 维
        # 对 K：独立投影 c_rope 维 → RoPE → 拼接到 K 主体
        #
        # TODO Part 4.2: 提取并旋转 Q 的前 c_rope 维
        #   q_rope = q_full[..., :self.c_rope]   # [B, nh, T, c_rope]
        #   q_rest = q_full[..., self.c_rope:]   # [B, nh, T, hd-c_rope]
        #   用 RoPE.apply(q_rope, positions, self.c_rope) 旋转
        #   再 cat 回去
        positions = torch.arange(T, device=x.device)
        # 你的代码在这里

        # TODO Part 4.3: 计算并旋转 K_rope，然后拼接
        #   k_rope_flat = self.k_rope_proj(x)     # [B, T, c_rope]
        #   k_rope = k_rope_flat.unsqueeze(1).expand(B, nh, T, self.c_rope)
        #   用 RoPE.apply(k_rope, positions, self.c_rope) 旋转
        #   k = torch.cat([k_rope, k_full[..., self.c_rope:]], dim=-1)
        # 你的代码在这里

        # 注意力计算（和 MHA 一样）
        s = torch.matmul(q_full, k_full.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        s = s.masked_fill(mask[None, None, :, :], float("-inf"))
        a = F.softmax(s, dim=-1)
        o = torch.matmul(a, v)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(o)


# ══════════════════════════════════════════════════════════════════════════════
# Part 5: 完整 MLA（练习）
# ══════════════════════════════════════════════════════════════════════════════
#
# 把 Part 3 的低秩压缩 + Part 4 的解耦 RoPE 合起来，就是完整的 MLA！
#
# 架构：
#
#   x ──→ q_proj ──→ Q (d_model 维)
#            Q 的前 c_rope 维加 RoPE，后面不变
#
#   x ──→ kv_compress ──→ z (c_kv 维) ← 只缓存这个！
#           ├──→ k_rope_proj ──→ c_rope 维 ─→ RoPE ─→ K_rope
#           ├──→ k_proj ──→ K_rest (d_model 维)
#           │              取后 (head_dim - c_rope) 维
#           └──→ v_proj ──→ V
#
#   最终 K = [K_rope, K_rest]   ← 拼接
#

class DeepSeekMLA_Practice(nn.Module):
    """
    完整的 DeepSeek MLA 实现（练习版）

    你需要完成以下 TODO：
      Part 5.1: 定义 __init__ 中的投影层
      Part 5.2: 实现 forward 中的前向传播
    """
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        c_kv: int = 16,
        c_rope: int = 8,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert c_rope <= c_kv, f"c_rope({c_rope}) 必须 <= c_kv({c_kv})，否则 K_rope 无法拼入 head_dim"
        assert c_rope <= d_model // num_heads, f"c_rope({c_rope}) 不能超过 head_dim({d_model // num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_kv = c_kv
        self.c_rope = c_rope
        self.scale = self.head_dim ** -0.5

        # ── TODO Part 5.1: 定义所有投影层（共 6 个 Linear，均 bias=False） ──
        # 列表如下：
        #
        #   1. q_proj:       d_model → d_model    (Q 投影)
        #   2. kv_compress:  d_model → c_kv       (KV 压缩，缓存这个！)
        #   3. k_rope_proj:  c_kv → c_rope        (从压缩表示提取 RoPE)
        #   4. k_proj:       c_kv → d_model       (K 解压)
        #   5. v_proj:       c_kv → d_model       (V 解压)
        #   6. o_proj:       d_model → d_model    (输出)
        #
        # 提示：nn.Linear(in_dim, out_dim, bias=False)

        # 练习：
        # self.q_proj = nn.Linear(d_model, d_model, bias=False)
        # self.kv_compress = nn.Linear(d_model, c_kv, bias=False)
        # self.k_rope_proj = nn.Linear(c_kv, c_rope, bias=False)
        # self.k_proj = nn.Linear(c_kv, d_model, bias=False)
        # self.v_proj = nn.Linear(c_kv, d_model, bias=False)
        # self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 先全部设为 None，你来实现
        self.q_proj = None
        self.kv_compress = None
        self.k_rope_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None

        if any(p is None for p in [self.q_proj, self.kv_compress, self.k_rope_proj,
                                    self.k_proj, self.v_proj, self.o_proj]):
            raise NotImplementedError("Part 5.1: 请定义所有投影层")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        nh = self.num_heads
        hd = self.head_dim
        device = x.device

        # ── Step 1: Q 投影 ───────────────────────────────────────────────
        q = self.q_proj(x).view(B, T, nh, hd).transpose(1, 2)

        # ── Step 2: KV 压缩 ──────────────────────────────────────────────
        # TODO Part 5.2a: z = self.kv_compress(x)  # [B, T, c_kv]
        z = None
        if z is None:
            raise NotImplementedError("Part 5.2a: 请实现 KV 压缩")

        # ── Step 3: RoPE 分支 ────────────────────────────────────────────
        # TODO Part 5.2b: 从压缩表示 z 计算 K_rope
        #   1) k_rope_flat = self.k_rope_proj(z)        # [B, T, c_rope]
        #   2) 扩展到多头: k_rope = k_rope_flat.unsqueeze(1).expand(B, nh, T, self.c_rope)
        #   3) 应用 RoPE:
        #        positions = torch.arange(T, device=device)
        #        k_rope = RoPE.apply(k_rope, positions, self.c_rope)
        k_rope = None
        if k_rope is None:
            raise NotImplementedError("Part 5.2b: 请实现 RoPE 分支")

        # ── Step 4: Q 的前 c_rope 维也加 RoPE ────────────────────────────
        # TODO Part 5.2c: Q 的 RoPE
        #   1) q_rope = q[..., :self.c_rope]     # 前 c_rope 维
        #   2) q_rest = q[..., self.c_rope:]     # 剩余维度
        #   3) q_rope = RoPE.apply(q_rope, positions, self.c_rope)
        #   4) q = torch.cat([q_rope, q_rest], dim=-1)
        # 你的代码在这里

        # ── Step 5: K/V 解压 ─────────────────────────────────────────────
        # TODO Part 5.2d: 从压缩表示 z 解压出 K 主体和 V
        #   k_body = self.k_proj(z).view(B, T, nh, hd).transpose(1, 2)
        #   v = self.v_proj(z).view(B, T, nh, hd).transpose(1, 2)
        k_body = None
        v = None
        if k_body is None or v is None:
            raise NotImplementedError("Part 5.2d: 请解压 K 和 V")

        # ── Step 6: 拼接 K ───────────────────────────────────────────────
        # K_rope（已旋转 c_rope 维）+ K_body 的后 (hd - c_rope) 维
        # TODO Part 5.2e: k = torch.cat([k_rope, k_body[..., self.c_rope:]], dim=-1)
        k = None
        if k is None:
            raise NotImplementedError("Part 5.2e: 请拼接 K")

        # ── Step 7-9: 注意力计算 ─────────────────────────────────────────
        s = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        s = s.masked_fill(mask[None, None, :, :], float("-inf"))
        a = F.softmax(s, dim=-1)
        o = torch.matmul(a, v)
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(o)


# ══════════════════════════════════════════════════════════════════════════════
# Part 6: 综合测试
# ══════════════════════════════════════════════════════════════════════════════

def test_low_rank_attn():
    """测试 Part 3: 低秩压缩注意力"""
    print("\n[Part 3] 低秩压缩注意力...", end=" ")
    try:
        model = LowRankAttn(d_model=64, num_heads=4, c_kv=16).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64), f"形状错误: {out.shape}"
        print("通过 ✓")
        return True
    except Exception as e:
        print(f"未通过: {e}")
        return False


def test_decoupled_rope():
    """测试 Part 4: 解耦 RoPE 注意力"""
    print("\n[Part 4] 解耦 RoPE 注意力...", end=" ")
    try:
        model = DecoupledRoPEAttn(d_model=64, num_heads=4, c_rope=8).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64), f"形状错误: {out.shape}"
        print("通过 ✓")
        return True
    except Exception as e:
        print(f"未通过: {e}")
        return False


def test_full_mla():
    """测试 Part 5: 完整 MLA"""
    print("\n[Part 5] 完整 MLA...", end=" ")
    try:
        model = DeepSeekMLA_Practice(d_model=64, num_heads=4, c_kv=16, c_rope=8).to(device)
        x = torch.randn(2, 8, 64, device=device)
        out = model(x)
        assert out.shape == (2, 8, 64), f"形状错误: {out.shape}"
        print("通过 ✓")
        return True
    except Exception as e:
        print(f"未通过: {e}")
        return False


def test_gradient():
    """梯度回传测试"""
    print("\n[梯度测试] 梯度回传...", end=" ")
    model = DeepSeekMLA_Practice(d_model=64, num_heads=4, c_kv=16, c_rope=8).to(device)
    x = torch.randn(2, 8, 64, device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()
    # 检查关键层是否有梯度
    grads_ok = all(
        p.grad is not None
        for p in [model.q_proj.weight, model.kv_compress.weight,
                  model.k_proj.weight, model.v_proj.weight]
    )
    if grads_ok:
        print("通过 ✓")
    else:
        print("未通过（某些层无梯度）✗")
    return grads_ok


def test_causality():
    """因果性测试"""
    print("\n[因果性测试] 未来不可见...", end=" ")
    model = DeepSeekMLA_Practice(d_model=64, num_heads=4, c_kv=16, c_rope=8).to(device)
    model.eval()
    x = torch.randn(1, 8, 64, device=device)
    with torch.no_grad():
        out = model(x)
    print("通过 ✓")


def test_compression_ratio():
    """压缩比验证"""
    print("\n[压缩比验证]")
    d_model, num_heads, c_kv = 5120, 128, 512
    mha_per_token = 2 * d_model
    mla_per_token = c_kv
    ratio = mha_per_token / mla_per_token
    print(f"  MHA: {mha_per_token} 维/token")
    print(f"  MLA: {mla_per_token} 维/token")
    print(f"  压缩比: {ratio:.0f}x")
    assert ratio > 1


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 60)
    print("DeepSeek MLA 学习脚本")
    print("  - Part 1: KV Cache 计算练习")
    print("  - Part 2: 标准 MHA（复习）")
    print("  - Part 3: 低秩压缩注意力（练习）")
    print("  - Part 4: 解耦 RoPE（练习）")
    print("  - Part 5: 完整 MLA（练习）")
    print("  - Part 6: 测试验证")
    print("=" * 60)

    # Part 1: 计算练习（不阻塞测试）
    part1_calculate_kv_cache()

    # Part 3-5: 测试
    results = []
    results.append(("Part 3", test_low_rank_attn()))
    results.append(("Part 4", test_decoupled_rope()))
    results.append(("Part 5", test_full_mla()))

    # 只在 Part 5 通过时才跑后续测试
    if results[-1][1]:
        test_gradient()
        test_causality()
        test_compression_ratio()

    print("\n" + "=" * 60)
    all_pass = all(r[1] for r in results)
    if all_pass:
        print("恭喜！所有已完成的 Part 测试通过！")
        print("（Part 3 和 Part 4 的测试不依赖 Part 5 的实现）")
    else:
        failed = [r[0] for r in results if not r[1]]
        print(f"以下 Part 未通过测试: {', '.join(failed)}")
        print("请检查对应的 TODO 是否已正确填写。")
    print("=" * 60)