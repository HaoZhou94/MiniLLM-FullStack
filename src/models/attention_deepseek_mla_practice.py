"""
DeepSeek-V2 MLA（Multi-head Latent Attention）练习文件

目标：读懂每个模块的注释后，自己填写 TODO 块，不看 attention_deepseek_mla.py。
完成后用 `python attention_deepseek_mla_practice.py` 跑测试，全部通过即完成。

建议顺序：__init__ → _split_heads/_merge_heads → _apply_rope → forward

核心要理解的问题：
  1. MLA 相比标准 MHA，KV Cache 的存储形式是什么？
  2. K 的最终形态是怎么拼出来的？（RoPE 分量 + 解压分量）
  3. RoPE 为什么要"解耦"到低维空间 c_rope？
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# DeepSeekMLA
# 难度：★★★★☆
# ══════════════════════════════════════════════════════════════════════════════
#
# MLA 和标准 MHA 的唯一区别：KV 的生成路径不同。
#
# 标准 MHA：
#   x → k_proj(d_model→d_model) → K
#   x → v_proj(d_model→d_model) → V
#   KV Cache 存储：K + V = 2 * d_model 维 / token
#
# MLA（低秩 KV 压缩）：
#   x → kv_compress(d_model→c_kv) → 压缩表示（只缓存这个！）
#   压缩表示 → k_proj(c_kv→d_model) → K 主体
#   压缩表示 → v_proj(c_kv→d_model) → V
#   压缩表示 → rope_proj(c_kv→c_rope) → 应用 RoPE → K_rope
#   K = [K_rope, K主体的后(head_dim-c_rope)维]  ← 拼接！
#   KV Cache 存储：压缩表示 = c_kv 维 / token（大幅缩减）
#
# 参数说明：
#   d_model: 模型主干维度
#   c_kv:    KV 压缩维度（如 512，越小越省显存）
#   c_rope:  RoPE 维度（如 64）
#   head_dim = d_model / num_heads
#
class DeepSeekMLA(nn.Module):
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
            c_kv:              KV 压缩维度（核心超参数，越小 KV Cache 越小）
            c_rope:            RoPE 维度（通常 64 或 128）
            attention_dropout: 注意力权重 dropout
            hidden_dropout:    输出投影 dropout
            bias:              线性层偏置
        """
        super().__init__()

        assert d_model % num_heads == 0
        assert c_rope <= c_kv

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.c_kv = c_kv
        self.c_rope = c_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # TODO: 定义以下 6 个线性层（均不带偏置，bias=False）
        #
        # 1. self.q_proj      —— Q 投影：d_model → d_model
        #    （Q 不压缩，和标准 MHA 一样）
        #
        # 2. self.kv_compress —— KV 压缩：d_model → c_kv
        #    （这是 KV Cache 的存储形式！推理时只缓存这个）
        #
        # 3. self.rope_proj   —— RoPE 投影：c_kv → c_rope
        #    （从压缩表示中提取 RoPE 所需的低维信息）
        #
        # 4. self.k_proj      —— K 解压：c_kv → d_model
        #    （推理时从缓存的压缩表示动态解压）
        #
        # 5. self.v_proj      —— V 解压：c_kv → d_model
        #
        # 6. self.o_proj      —— 输出投影：d_model → d_model
        #
        raise NotImplementedError

        # TODO: 定义两个 Dropout 层
        #   self.attention_dropout = nn.Dropout(attention_dropout)
        #   self.hidden_dropout    = nn.Dropout(hidden_dropout)
        raise NotImplementedError


    # ── 辅助方法 1：拆分多头 ──────────────────────────────────────────────────
    #
    # 标准 MHA 中也有此操作，MLA 中完全一样。
    # 变换：[B, T, d_model] → [B, num_heads, T, head_dim]
    #
    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x:          [B, T, d_model]
            batch_size: B
        Returns:
            [B, num_heads, T, head_dim]
        """
        # TODO: view + transpose
        # 提示：
        #   x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        #   return x.transpose(1, 2).contiguous()
        raise NotImplementedError


    # ── 辅助方法 2：合并多头 ──────────────────────────────────────────────────
    #
    # _split_heads 的逆操作。
    # 变换：[B, num_heads, T, head_dim] → [B, T, d_model]
    #
    def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            x:          [B, num_heads, T, head_dim]
            batch_size: B
        Returns:
            [B, T, d_model]
        """
        # TODO: transpose + contiguous + view
        raise NotImplementedError


    # ── 辅助方法 3：应用 RoPE ─────────────────────────────────────────────────
    #
    # MLA 的 RoPE 是"解耦"的：
    #   - Q 只旋转前 c_rope 维，剩余维度不变
    #   - K_rope 是独立投影到 c_rope 维后旋转
    #
    # 旋转公式（对每对相邻维度 (x1, x2)）：
    #   x1_rot = x1 * cos - x2 * sin
    #   x2_rot = x1 * sin + x2 * cos
    #
    def _apply_rope(
        self,
        q: torch.Tensor,        # [B, num_heads, T, head_dim]
        k_rope: torch.Tensor,   # [B, num_heads, T, c_rope]
        seq_len: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            q_rot:      [B, num_heads, T, head_dim]  （前 c_rope 维已旋转）
            k_rope_rot: [B, num_heads, T, c_rope]    （已旋转）
        """
        # TODO Step 1: 计算旋转角度
        #   位置索引：position = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
        #   频率：freqs = 1.0 / (10000.0 ** (dim_indices / self.c_rope))
        #         其中 dim_indices = torch.arange(0, self.c_rope, 2, device=device).float()
        #   theta = position.transpose(0,1).float() @ freqs.unsqueeze(0)  # [T, c_rope//2]
        #   cos_theta = cos(theta).unsqueeze(0).unsqueeze(0)  # [1, 1, T, c_rope//2]
        #   sin_theta = sin(theta).unsqueeze(0).unsqueeze(0)

        # TODO Step 2: 旋转 Q 的前 c_rope 维
        #   q_rope = q[..., :self.c_rope]      # 前 c_rope 维
        #   q_rest = q[..., self.c_rope:]      # 剩余维度（不旋转）
        #   q1, q2 = q_rope[..., ::2], q_rope[..., 1::2]   # 拆奇偶
        #   旋转：q1_rot = q1 * cos - q2 * sin
        #         q2_rot = q1 * sin + q2 * cos
        #   拼回：q_rope_rot = torch.stack([q1_rot, q2_rot], dim=-1).flatten(-2)
        #         q_rot = torch.cat([q_rope_rot, q_rest], dim=-1)

        # TODO Step 3: 旋转 K_rope（同 Step 2，但没有 q_rest 部分）

        raise NotImplementedError


    # ── 辅助方法 4：因果掩码 ──────────────────────────────────────────────────
    #
    # 和标准 MHA 完全一样，上三角 -inf，下三角 0。
    # 输出形状：[1, 1, T, T]
    #
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # TODO: tril + masked_fill + unsqueeze × 2
        # 提示：
        #   mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        #   mask = mask.masked_fill(mask == 0, float("-inf"))
        #   mask = mask.masked_fill(mask == 1, 0.0)
        #   return mask.unsqueeze(0).unsqueeze(0)
        raise NotImplementedError


    # ── 前向传播（核心）──────────────────────────────────────────────────────
    #
    # 9 步走，前 5 步是 MLA 独有，后 4 步和标准 MHA 一样：
    #
    #  Step 1: Q 投影 → split_heads → [B, nh, T, hd]
    #  Step 2: KV 压缩 → kv_compressed [B, T, c_kv]   ← 推理时缓存这个
    #  Step 3: RoPE 分支：kv_compressed → rope_proj → 扩展多头 → _apply_rope
    #  Step 4: K/V 解压：kv_compressed → k_proj/v_proj → split_heads
    #  Step 5: 拼接 K：[K_rope, K主体后(hd-c_rope)维] → K [B, nh, T, hd]
    #  Step 6: 缩放点积注意力分数 → 叠加因果掩码 → 叠加 padding 掩码
    #  Step 7: softmax → attention_probs
    #  Step 8: attention_probs @ V → context_layer → merge_heads
    #  Step 9: o_proj → hidden_dropout → return
    #
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            hidden_states:     [B, T, d_model]
            attention_mask:    [B, 1, 1, T] padding 掩码（可选，-inf / 0）
            output_attentions: 是否返回注意力权重矩阵

        Returns:
            output:            [B, T, d_model]
            attention_probs:   [B, num_heads, T, T]（output_attentions=True 时）
        """
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device

        # TODO Step 1: Q 投影 + split_heads
        # q = self.q_proj(hidden_states)          # [B, T, d_model]
        # q = self._split_heads(q, batch_size)    # [B, num_heads, T, head_dim]

        # TODO Step 2: KV 压缩（MLA 核心，推理时只存这个）
        # kv_compressed = self.kv_compress(hidden_states)  # [B, T, c_kv]

        # TODO Step 3: RoPE 分支
        #   a. k_rope = self.rope_proj(kv_compressed)          # [B, T, c_rope]
        #   b. 扩展到多头：k_rope = k_rope.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.c_rope)
        #   c. q, k_rope = self._apply_rope(q, k_rope, seq_len, device)

        # TODO Step 4: K/V 解压
        #   k = self.k_proj(kv_compressed)   # [B, T, d_model]
        #   v = self.v_proj(kv_compressed)   # [B, T, d_model]
        #   k = self._split_heads(k, batch_size)  # [B, num_heads, T, head_dim]
        #   v = self._split_heads(v, batch_size)

        # TODO Step 5: 拼接 K
        #   K = [K_rope（已旋转，c_rope 维）, K主体后(head_dim-c_rope)维]
        #   k = torch.cat([k_rope, k[..., self.c_rope:]], dim=-1)  # [B, nh, T, head_dim]

        # TODO Step 6: 注意力分数
        #   attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        #   attention_scores = attention_scores + self._create_causal_mask(seq_len, device)
        #   if attention_mask is not None: attention_scores = attention_scores + attention_mask

        # TODO Step 7: Softmax + dropout
        #   attention_probs = F.softmax(attention_scores, dim=-1)
        #   attention_probs = self.attention_dropout(attention_probs)

        # TODO Step 8: 加权聚合 + merge_heads
        #   context_layer = torch.matmul(attention_probs, v)  # [B, nh, T, hd]
        #   context_layer = self._merge_heads(context_layer, batch_size)  # [B, T, d_model]

        # TODO Step 9: 输出投影
        #   output = self.o_proj(context_layer)
        #   output = self.hidden_dropout(output)
        #   if output_attentions: return output, attention_probs
        #   return output, None

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 测试（全部通过 = 实现正确）
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = 2048
    num_heads = 16
    c_kv = 256
    c_rope = 32
    B, T = 2, 128

    print("=" * 60)
    print("DeepSeek MLA 练习文件测试")
    print("=" * 60)

    mla = DeepSeekMLA(
        d_model=d_model,
        num_heads=num_heads,
        c_kv=c_kv,
        c_rope=c_rope,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    ).to(device)

    hidden_states = torch.randn(B, T, d_model, device=device)

    print("\n[1] 前向传播（无注意力输出）")
    output, attn = mla(hidden_states, output_attentions=False)
    assert output.shape == (B, T, d_model), f"期望 {(B, T, d_model)}，得到 {output.shape}"
    assert attn is None, "output_attentions=False 时应返回 None"
    print(f"  通过：输出形状 {output.shape}")

    print("\n[2] 前向传播（含注意力输出）")
    output, attn = mla(hidden_states, output_attentions=True)
    assert output.shape == (B, T, d_model)
    assert attn.shape == (B, num_heads, T, T), f"期望 {(B, num_heads, T, T)}，得到 {attn.shape}"
    print(f"  通过：注意力权重形状 {attn.shape}")

    print("\n[3] 注意力权重求和验证（因果 softmax 每行应等于 1）")
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"注意力权重行和不为 1：{row_sums[0, 0, :3]}"
    print(f"  通过：注意力权重行和 ≈ 1.0")

    print("\n[4] 因果掩码验证（未来位置的注意力权重应为 0）")
    for i in range(T):
        future_attn = attn[0, 0, i, i+1:]
        assert future_attn.abs().max() < 1e-6, f"位置 {i} 对未来 token 有非零注意力"
    print(f"  通过：未来 token 注意力权重为 0")

    print("\n[5] padding 掩码测试")
    padding_mask = torch.zeros(B, 1, 1, T, device=device)
    padding_mask[0, 0, 0, T//2:] = float("-inf")
    output_masked, _ = mla(hidden_states, attention_mask=padding_mask)
    assert output_masked.shape == (B, T, d_model)
    print(f"  通过：带 padding 掩码的输出形状 {output_masked.shape}")

    print("\n[6] 反向传播")
    loss = output.mean()
    loss.backward()
    assert mla.q_proj.weight.grad is not None, "q_proj 应有梯度"
    assert mla.kv_compress.weight.grad is not None, "kv_compress 应有梯度"
    assert mla.k_proj.weight.grad is not None, "k_proj 应有梯度"
    print(f"  通过：所有投影层梯度正常")

    print("\n[7] KV Cache 压缩比验证")
    mha_kv = 2 * num_heads * (d_model // num_heads) * T
    mla_kv = c_kv * T
    ratio = mha_kv / mla_kv
    print(f"  标准 MHA KV Cache：{mha_kv} 维/序列")
    print(f"  MLA KV Cache：      {mla_kv} 维/序列")
    print(f"  压缩比：            {ratio:.1f}x")
    assert ratio > 1.0, "MLA 应该比 MHA 更节省"
    print(f"  通过：压缩比 {ratio:.1f}x")

    print("\n所有测试通过！")