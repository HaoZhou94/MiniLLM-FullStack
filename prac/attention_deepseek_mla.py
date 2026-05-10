"""
DeepSeek-V2 多头潜在注意力（Multi-head Latent Attention，MLA）实现

# ═══════════════════════════════════════════════════════════════════
# 一、MLA 核心创新：低秩 KV 压缩
# ═══════════════════════════════════════════════════════════════════
#
# 标准 MHA 的 KV Cache 瓶颈：
#   每个 token 的 K/V 需要存储：2 * num_heads * head_dim 维度
#   例如 LLaMA-7B：2 * 32 * 128 = 8192 维 / token
#   长序列推理时，KV Cache 占用显存巨大（几十 GB）
#
# DeepSeek MLA 的解决方案：
#   1. 把 K/V 先压缩到低秩潜在空间（c_kv 维，如 512）
#   2. 只缓存压缩后的表示（512 维 / token）
#   3. 推理时再从压缩表示解压到多头空间
#
# 压缩比：c_kv / (num_heads * head_dim)
#   DeepSeek-V2：512 / (128 * 128) ≈ 3.1%（压缩 32 倍）
#
# ═══════════════════════════════════════════════════════════════════
# 二、MLA 架构图解
# ═══════════════════════════════════════════════════════════════════
#
#  输入 x: [B, T, d_model]
#       │
#       ├─────────────────────────────────────────────────────┐
#       │                                                       │
#       ▼ Q 分支（标准投影）                                    ▼ KV 分支（低秩压缩）
#  ┌─────────────────────┐                            ┌─────────────────────┐
#  │ q_proj              │                            │ kv_compress         │
#  │ [d_model → d_model] │                            │ [d_model → c_kv]    │
#  └─────────────────────┘                            └─────────────────────┘
#       │                                                       │
#       │ split_heads                                           │ 压缩表示（缓存这个！）
#       │ [B,T,d] → [B,nh,T,hd]                                │
#       │                                                       ├─ RoPE 分支
#       │                                                       │  [c_kv → c_rope]
#       │                                                       │  应用 RoPE
#       │                                                       │
#       │                                                       ├─ K 解压分支
#       │                                                       │  [c_kv → nh*hd]
#       │                                                       │  split_heads
#       │                                                       │
#       │                                                       └─ V 解压分支
#       │                                                          [c_kv → nh*hd]
#       │                                                          split_heads
#       │
#       ▼ Q [B,nh,T,hd]                                        ▼ K,V [B,nh,T,hd]
#  ┌─────────────────────────────────────────────────────────────────┐
#  │  Scaled Dot-Product Attention                                   │
#  │  scores = (Q @ K^T) / sqrt(hd)                                  │
#  │  attn = softmax(scores + causal_mask)                           │
#  │  output = attn @ V                                              │
#  └─────────────────────────────────────────────────────────────────┘
#       │ merge_heads
#       │ [B,nh,T,hd] → [B,T,d_model]
#       ▼
#  ┌─────────────────────┐
#  │ o_proj              │
#  │ [d_model → d_model] │
#  └─────────────────────┘
#       │
#       ▼ 输出 [B, T, d_model]
#
# ═══════════════════════════════════════════════════════════════════
# 三、关键参数说明
# ═══════════════════════════════════════════════════════════════════
#
# d_model:    模型主干维度（如 4096）
# num_heads:  注意力头数（如 128）
# head_dim:   每个头的维度 = d_model / num_heads（如 128）
# c_kv:       KV 压缩维度（如 512，核心超参数）
# c_rope:     RoPE 应用的维度（如 64，通常远小于 c_kv）
#
# 显存对比（以 DeepSeek-V2 67B 为例，序列长度 32K）：
#   标准 MHA KV Cache：2 * 128 * 128 * 32K * 2字节 ≈ 2.1 GB / layer
#   MLA KV Cache：      512 * 32K * 2字节 ≈ 32 MB / layer
#   压缩比：65 倍
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DeepSeekMLA(nn.Module):
    """
    DeepSeek-V2 多头潜在注意力（MLA）实现。

    核心特性：
    1. 低秩 KV 压缩：KV 先压缩到 c_kv 维潜在空间，大幅减少 KV Cache
    2. 解耦 RoPE：位置编码在独立的低维空间（c_rope）应用
    3. 高效推理：KV Cache 只存压缩表示，推理时动态解压

    参数量对比（以 d_model=4096, num_heads=128 为例）：
      标准 MHA：3 * d_model * d_model = 50M 参数
      MLA：     d_model * (d_model + 2*c_kv + c_rope) ≈ 20M 参数（c_kv=512）
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 128,
        c_kv: int = 512,
        c_rope: int = 64,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        bias: bool = False,
    ):
        """
        Args:
            d_model:           模型主干维度
            num_heads:         注意力头数
            c_kv:              KV 压缩维度（核心超参数，越小显存越省）
            c_rope:            RoPE 应用的维度（通常 64 或 128）
            attention_dropout: 注意力权重 dropout 概率
            hidden_dropout:    输出投影 dropout 概率
            bias:              线性层是否带偏置
        """
        super().__init__()

        # ── 维度验证 ──────────────────────────────────────────────────────────
        assert d_model % num_heads == 0, f"d_model {d_model} 必须能被 num_heads {num_heads} 整除"
        assert c_rope <= c_kv, f"c_rope {c_rope} 不能大于 c_kv {c_kv}"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_kv = c_kv
        self.c_rope = c_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # ── Q 投影（标准 MHA 风格）────────────────────────────────────────────
        # Q 不压缩，直接投影到完整的多头空间
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)

        # ── KV 压缩投影（MLA 核心）────────────────────────────────────────────
        # 把输入从 d_model 维压缩到 c_kv 维潜在空间
        # 这是 KV Cache 的存储形式，推理时只缓存这个压缩表示
        self.kv_compress = nn.Linear(d_model, c_kv, bias=bias)

        # ── RoPE 投影（解耦位置编码）──────────────────────────────────────────
        # 从压缩表示中提取 c_rope 维用于 RoPE
        # 为什么解耦？RoPE 只需要低维信息，不需要完整的 c_kv 维
        self.rope_proj = nn.Linear(c_kv, c_rope, bias=bias)

        # ── K/V 解压投影 ──────────────────────────────────────────────────────
        # 从压缩表示 c_kv 解压到多头空间 num_heads * head_dim
        # 推理时：从缓存的压缩表示动态解压，不存储解压后的 K/V
        self.k_proj = nn.Linear(c_kv, d_model, bias=bias)
        self.v_proj = nn.Linear(c_kv, d_model, bias=bias)

        # ── 输出投影 ──────────────────────────────────────────────────────────
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # ── Dropout ───────────────────────────────────────────────────────────
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.hidden_dropout = nn.Dropout(hidden_dropout)

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        拆分多头：[B, T, d_model] → [B, num_heads, T, head_dim]

        与标准 MHA 完全一致，只是输入来自不同的投影路径。
        """
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        合并多头：[B, num_heads, T, head_dim] → [B, T, d_model]
        """
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k_rope: torch.Tensor,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用旋转位置编码（RoPE）到 Q 和 K 的 RoPE 分量。

        MLA 的 RoPE 应用方式：
        1. Q 的前 c_rope 维应用 RoPE（后面维度不变）
        2. K 的 RoPE 分量（单独投影的 c_rope 维）应用 RoPE
        3. 推理时把旋转后的 K_rope 拼接到 K 的主体部分

        Args:
            q:      [B, num_heads, T, head_dim]  Q 向量
            k_rope: [B, num_heads, T, c_rope]    K 的 RoPE 分量
            seq_len: 序列长度
            device:  设备

        Returns:
            q_rot:      [B, num_heads, T, head_dim]  旋转后的 Q
            k_rope_rot: [B, num_heads, T, c_rope]    旋转后的 K_rope
        """
        # 生成位置索引
        position = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]

        # 计算旋转角度（只对 c_rope 维度）
        # 频率：10000^(-2i/c_rope)，i = 0, 1, ..., c_rope//2 - 1
        dim_indices = torch.arange(0, self.c_rope, 2, device=device).float()
        freqs = 1.0 / (10000.0 ** (dim_indices / self.c_rope))  # [c_rope//2]

        # 位置 × 频率 → 旋转角度
        # position: [1, T], freqs: [c_rope//2] → theta: [T, c_rope//2]
        theta = position.transpose(0, 1).float() @ freqs.unsqueeze(0)  # [T, c_rope//2]

        # 扩展维度以匹配输入：[T, c_rope//2] → [1, 1, T, c_rope//2]
        cos_theta = torch.cos(theta).unsqueeze(0).unsqueeze(0)
        sin_theta = torch.sin(theta).unsqueeze(0).unsqueeze(0)

        # ── 旋转 Q 的前 c_rope 维 ────────────────────────────────────────────
        # 提取 Q 的前 c_rope 维和剩余维度
        q_rope = q[..., :self.c_rope]      # [B, nh, T, c_rope]
        q_rest = q[..., self.c_rope:]      # [B, nh, T, head_dim - c_rope]

        # 拆分奇偶维度进行旋转
        q1, q2 = q_rope[..., ::2], q_rope[..., 1::2]  # 各 [B, nh, T, c_rope//2]
        q1_rot = q1 * cos_theta - q2 * sin_theta
        q2_rot = q1 * sin_theta + q2 * cos_theta

        # 拼接旋转后的维度
        q_rope_rot = torch.stack([q1_rot, q2_rot], dim=-1).flatten(-2)  # [B, nh, T, c_rope]
        q_rot = torch.cat([q_rope_rot, q_rest], dim=-1)  # [B, nh, T, head_dim]

        # ── 旋转 K 的 RoPE 分量 ──────────────────────────────────────────────
        k1, k2 = k_rope[..., ::2], k_rope[..., 1::2]
        k1_rot = k1 * cos_theta - k2 * sin_theta
        k2_rot = k1 * sin_theta + k2 * cos_theta
        k_rope_rot = torch.stack([k1_rot, k2_rot], dim=-1).flatten(-2)  # [B, nh, T, c_rope]

        return q_rot, k_rope_rot

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码：[1, 1, T, T]

        与标准 MHA 完全一致。
        """
        mask = torch.ones(seq_len, seq_len, device=device)
        mask = torch.tril(mask)
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, 0.0)
        return mask.unsqueeze(0).unsqueeze(0)

    # ── 前向传播 ──────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        MLA 前向传播。

        数据流：
        1. Q 分支：x → q_proj → split_heads → 应用 RoPE（前 c_rope 维）
        2. KV 分支：
           a. x → kv_compress（压缩到 c_kv 维，这是缓存的形式）
           b. 压缩表示 → rope_proj → 应用 RoPE → K_rope
           c. 压缩表示 → k_proj → split_heads → 拼接 K_rope → K
           d. 压缩表示 → v_proj → split_heads → V
        3. 注意力计算：标准 scaled dot-product attention
        4. 输出投影：o_proj

        Args:
            hidden_states:    [B, T, d_model]
            attention_mask:   [B, 1, 1, T] padding 掩码（可选）
            output_attentions: 是否返回注意力权重

        Returns:
            output:           [B, T, d_model]
            attention_probs:  [B, num_heads, T, T]（仅 output_attentions=True）
        """
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device

        # ── Step 1: Q 投影（标准路径）─────────────────────────────────────────
        q = self.q_proj(hidden_states)  # [B, T, d_model]
        q = self._split_heads(q, batch_size)  # [B, num_heads, T, head_dim]

        # ── Step 2: KV 压缩（MLA 核心）────────────────────────────────────────
        # 这是 KV Cache 的存储形式！推理时只缓存 kv_compressed
        kv_compressed = self.kv_compress(hidden_states)  # [B, T, c_kv]

        # ── Step 3: RoPE 分支 ─────────────────────────────────────────────────
        # 从压缩表示中提取 RoPE 信息
        k_rope = self.rope_proj(kv_compressed)  # [B, T, c_rope]
        # 扩展到多头空间（每个头共享相同的 RoPE 信息）
        k_rope = k_rope.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.c_rope)

        # 应用 RoPE 到 Q 和 K_rope
        q, k_rope = self._apply_rope(q, k_rope, seq_len, device)
        # q:      [B, num_heads, T, head_dim]（前 c_rope 维已旋转）
        # k_rope: [B, num_heads, T, c_rope]（已旋转）

        # ── Step 4: K/V 解压 ──────────────────────────────────────────────────
        # 从压缩表示解压到多头空间
        k = self.k_proj(kv_compressed)  # [B, T, d_model]
        v = self.v_proj(kv_compressed)  # [B, T, d_model]

        k = self._split_heads(k, batch_size)  # [B, num_heads, T, head_dim]
        v = self._split_heads(v, batch_size)  # [B, num_heads, T, head_dim]

        # ── Step 5: 拼接 K 的 RoPE 分量和主体部分 ─────────────────────────────
        # MLA 的 K 由两部分组成：
        #   1. 旋转后的 RoPE 分量（c_rope 维）
        #   2. 解压后的主体部分（head_dim - c_rope 维）
        # 注意：这里假设 K 的前 c_rope 维被 k_rope 替换
        k = torch.cat([k_rope, k[..., self.c_rope:]], dim=-1)  # [B, num_heads, T, head_dim]

        # ── Step 6: 缩放点积注意力 ────────────────────────────────────────────
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [B, nh, T, T]
        attention_scores = attention_scores * self.scale

        # 叠加因果掩码
        causal_mask = self._create_causal_mask(seq_len, device)
        attention_scores = attention_scores + causal_mask

        # 叠加 padding 掩码（可选）
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax 归一化
        attention_probs = F.softmax(attention_scores, dim=-1)  # [B, nh, T, T]
        attention_probs = self.attention_dropout(attention_probs)

        # ── Step 7: 加权聚合 V ────────────────────────────────────────────────
        context_layer = torch.matmul(attention_probs, v)  # [B, nh, T, head_dim]

        # ── Step 8: 合并多头 ──────────────────────────────────────────────────
        context_layer = self._merge_heads(context_layer, batch_size)  # [B, T, d_model]

        # ── Step 9: 输出投影 + Dropout ────────────────────────────────────────
        output = self.o_proj(context_layer)
        output = self.hidden_dropout(output)

        if output_attentions:
            return output, attention_probs
        return output, None


# ══════════════════════════════════════════════════════════════════════════════
# 测试：对比 MLA 和标准 MHA 的显存占用
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 配置参数（模拟 DeepSeek-V2 的小尺度版本）──────────────────────────────
    d_model = 2048       # 隐藏层维度（DeepSeek-V2 原版 5120）
    num_heads = 16       # 注意力头数（DeepSeek-V2 原版 128）
    c_kv = 256           # KV 压缩维度（DeepSeek-V2 原版 512）
    c_rope = 32          # RoPE 维度（DeepSeek-V2 原版 64）
    batch_size = 2
    seq_len = 128

    print("=" * 70)
    print("DeepSeek MLA 测试")
    print("=" * 70)

    # ── 实例化 MLA ────────────────────────────────────────────────────────────
    mla = DeepSeekMLA(
        d_model=d_model,
        num_heads=num_heads,
        c_kv=c_kv,
        c_rope=c_rope,
        attention_dropout=0.1,
        hidden_dropout=0.1,
    ).to(device)

    # ── 前向传播测试 ──────────────────────────────────────────────────────────
    hidden_states = torch.randn(batch_size, seq_len, d_model, device=device)
    output, attn_probs = mla(hidden_states, output_attentions=True)

    print(f"\n输入形状:       {hidden_states.shape}")   # [2, 128, 2048]
    print(f"输出形状:       {output.shape}")          # [2, 128, 2048]
    print(f"注意力权重形状: {attn_probs.shape}")      # [2, 16, 128, 128]

    # ── KV Cache 显存对比 ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KV Cache 显存对比（单层，序列长度 32K）")
    print("=" * 70)

    seq_len_long = 32768  # 32K 序列
    bytes_per_param = 2   # FP16

    # 标准 MHA 的 KV Cache
    mha_kv_size = 2 * num_heads * (d_model // num_heads) * seq_len_long * bytes_per_param
    mha_kv_mb = mha_kv_size / (1024 ** 2)

    # MLA 的 KV Cache（只存压缩表示）
    mla_kv_size = c_kv * seq_len_long * bytes_per_param
    mla_kv_mb = mla_kv_size / (1024 ** 2)

    compression_ratio = mha_kv_size / mla_kv_size

    print(f"\n标准 MHA KV Cache: {mha_kv_mb:.2f} MB")
    print(f"MLA KV Cache:      {mla_kv_mb:.2f} MB")
    print(f"压缩比:            {compression_ratio:.1f}x")
    print(f"节省显存:          {mha_kv_mb - mla_kv_mb:.2f} MB ({(1 - mla_kv_mb/mha_kv_mb)*100:.1f}%)")

    # ── 参数量对比 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("参数量对比")
    print("=" * 70)

    # MLA 参数量
    mla_params = sum(p.numel() for p in mla.parameters())

    # 标准 MHA 参数量（3 个 QKV 投影 + 1 个输出投影）
    mha_params = 4 * d_model * d_model

    print(f"\nMLA 参数量:        {mla_params / 1e6:.2f}M")
    print(f"标准 MHA 参数量:   {mha_params / 1e6:.2f}M")
    print(f"参数减少:          {(mha_params - mla_params) / 1e6:.2f}M ({(1 - mla_params/mha_params)*100:.1f}%)")

    print("\n所有测试通过！")