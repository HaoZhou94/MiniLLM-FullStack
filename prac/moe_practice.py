"""
混合专家模型（Mixture of Experts，MoE）练习文件

目标：读懂每个模块的注释后，自己填写 TODO 块，不看 moe.py。
完成后用 `python moe_practice.py` 跑测试，全部通过即完成。

建议顺序：Expert → TopKRouter → load_balance_loss → MoELayer → MoETransformerBlock
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 模块 1：Expert（单个专家 FFN）
# 难度：★☆☆☆☆
# ══════════════════════════════════════════════════════════════════════════════
#
# 一个专家就是一个两层 FFN：
#
#   x → Linear(d_model → d_ff) → SiLU激活 → Linear(d_ff → d_model) → 输出
#
# 和普通 Transformer FFN 完全一样，MoE 只是把 N 个这样的 FFN 并排放，
# 每个 token 只走其中 k 个。
#
# SiLU 激活函数：f(x) = x * sigmoid(x)，比 ReLU 更平滑，现代 LLM 常用。
#
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: 输入/输出维度
            d_ff:    中间层维度（通常是 d_model 的 4 倍）
        """
        super().__init__()
        # TODO: 定义两个线性层 w1、w2，以及 SiLU 激活函数
        # 提示：nn.Linear(in, out, bias=False)，nn.SiLU()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, d_model]  —— 分配给本专家的 token 集合
        Returns:
            [num_tokens, d_model]
        """
        # TODO: x → w1 → 激活 → w2
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 模块 2：TopKRouter（路由器）
# 难度：★★☆☆☆
# ══════════════════════════════════════════════════════════════════════════════
#
# 路由器的任务：为每个 token 选出 top-k 个专家，并计算权重。
#
# 步骤：
#   1. gate 线性层：[num_tokens, d_model] → [num_tokens, num_experts]  得到 logits
#   2. softmax(logits) → router_probs  （全局概率，用于辅助损失）
#   3. topk(router_probs, k) → topk_weights_raw, topk_indices
#   4. 对 topk_weights_raw 再做归一化 → topk_weights（使 k 个权重之和为 1）
#
# 为什么要做两次 softmax？
#   第一次（全局）：反映所有专家的竞争关系，用于辅助损失的梯度。
#   第二次（局部）：只对选中的 k 个归一化，保证加权求和输出的数值稳定。
#
class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int):
        """
        Args:
            d_model:     输入维度
            num_experts: 专家总数 N
            top_k:       每个 token 激活的专家数 k
        """
        super().__init__()
        assert top_k <= num_experts
        self.num_experts = num_experts
        self.top_k = top_k
        # TODO: 定义路由线性层 gate：d_model → num_experts，不带偏置
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [num_tokens, d_model]
        Returns:
            topk_indices:  [num_tokens, top_k]       选中的专家编号
            topk_weights:  [num_tokens, top_k]       归一化路由权重（行和为 1）
            router_probs:  [num_tokens, num_experts] 全局 softmax 概率
        """
        # TODO Step 1: 计算 logits
        # logits = ?  形状：[num_tokens, num_experts]

        # TODO Step 2: softmax → router_probs

        # TODO Step 3: torch.topk 选出 top-k
        # 提示：torch.topk(tensor, k, dim=-1) 返回 (values, indices)

        # TODO Step 4: 对选中的 k 个权重归一化（行和变为 1）
        # 提示：topk_weights_raw / topk_weights_raw.sum(dim=-1, keepdim=True)

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 模块 3：load_balance_loss（负载均衡辅助损失）
# 难度：★★★☆☆
# ══════════════════════════════════════════════════════════════════════════════
#
# 问题背景：
#   不加约束时，路由器会"偷懒"，总选同一两个专家（collapse），
#   其他专家永远得不到训练。
#
# 解决方案（Switch Transformer）：
#   aux_loss = num_experts * Σ_i( f_i * p_i )
#
#   f_i = 专家 i 实际收到的 token 比例（离散，不可微，用 detach()）
#         = count(被路由到专家i的token数) / (num_tokens * top_k)
#
#   p_i = 路由器给专家 i 的平均概率（连续，可微，梯度经此回传）
#         = mean over all tokens of router_probs[:, i]
#
# 直觉：专家 i 又被选得多（f_i 大）、路由器又给它高概率（p_i 大），
#       乘积大 → 损失大 → 梯度迫使路由器把概率分散出去。
#
def load_balance_loss(
    router_probs: torch.Tensor,   # [num_tokens, num_experts]
    topk_indices: torch.Tensor,   # [num_tokens, top_k]
    num_experts: int,
) -> torch.Tensor:
    """Returns: 标量损失"""
    num_tokens = router_probs.shape[0]

    # TODO Step 1: 计算 f_i
    #   把 topk_indices 展平 → one_hot → 统计每个专家被选次数 → 归一化
    #   提示：
    #     flat_indices = topk_indices.reshape(-1)             # [num_tokens * top_k]
    #     one_hot = F.one_hot(flat_indices, num_classes=num_experts).float()
    #     tokens_per_expert = one_hot.sum(dim=0)              # [num_experts]
    #     f = tokens_per_expert / (num_tokens * top_k)

    # TODO Step 2: 计算 p_i
    #   router_probs 沿 token 维度取平均
    #   p = ?  形状：[num_experts]

    # TODO Step 3: 计算辅助损失
    #   aux_loss = num_experts * (f.detach() * p).sum()
    #   注意：f 要 detach()，因为它不可微；p 保持可微用于梯度传播

    raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 模块 4：MoELayer（整合路由 + 专家分发 + 加权合并）
# 难度：★★★★☆
# ══════════════════════════════════════════════════════════════════════════════
#
# 这是整个 MoE 的核心，难点在 Step 3 的专家分发循环。
#
# 数据流：
#   x [B, T, d_model]
#   → reshape 为 [B*T, d_model]（展平 batch）
#   → Router → topk_indices [B*T, k], topk_weights [B*T, k]
#   → 对每个专家 i：
#       找出选了专家 i 的 token（token_mask）
#       批量送入 experts[i]
#       × 路由权重后累加到 output_flat
#   → output_flat reshape 回 [B, T, d_model]
#   → 计算 aux_loss
#
# 专家分发循环的关键：
#   expert_mask = (topk_indices == expert_id)          # [B*T, k]，哪些槽选了此专家
#   token_mask  = expert_mask.any(dim=-1)              # [B*T]，哪些 token 选了此专家
#   weights     = topk_weights[token_mask][expert_mask[token_mask]]  # [n_i]，对应权重
#
class MoELayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha
        # TODO: 初始化 router（TopKRouter）和 experts（nn.ModuleList，包含 num_experts 个 Expert）
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]
        Returns:
            output:   [B, T, d_model]
            aux_loss: 标量
        """
        B, T, d_model = x.shape

        # TODO Step 1: 展平 → [B*T, d_model]

        # TODO Step 2: 调用 router，得到 topk_indices, topk_weights, router_probs

        # TODO Step 3: 初始化 output_flat = torch.zeros_like(x_flat)
        #   然后 for expert_id in range(self.num_experts):
        #     a. expert_mask = (topk_indices == expert_id)   # [B*T, k]
        #     b. token_mask  = expert_mask.any(dim=-1)       # [B*T]
        #     c. 若无 token 选该专家，continue
        #     d. expert_input = x_flat[token_mask]
        #     e. expert_output = self.experts[expert_id](expert_input)
        #     f. weights = topk_weights[token_mask][expert_mask[token_mask]]
        #     g. output_flat[token_mask] += expert_output * weights.unsqueeze(-1)

        # TODO Step 4: reshape output_flat 回 [B, T, d_model]

        # TODO Step 5: 计算 aux_loss = self.aux_loss_alpha * load_balance_loss(...)

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 模块 5：MoETransformerBlock（完整 Block）
# 难度：★★☆☆☆
# ══════════════════════════════════════════════════════════════════════════════
#
# 结构：
#   x → LayerNorm → MultiHeadAttention → Dropout → 残差
#     → LayerNorm → MoELayer           → Dropout → 残差
#     → (output, aux_loss)
#
# 与标准 Transformer Block 唯一的区别：FFN 换成 MoELayer。
# forward 要返回 aux_loss，调用方把它加到总训练损失里。
#
class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.0,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()
        # TODO: 定义 norm1, attn (nn.MultiheadAttention, batch_first=True), norm2, moe, dropout
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:         [B, T, d_model]
            attn_mask: [T, T] 因果掩码（上三角 -inf）
        Returns:
            output:   [B, T, d_model]
            aux_loss: 标量
        """
        # TODO: 自注意力子层（Pre-Norm 风格：先 norm 再 attention，再残差）
        #   residual = x
        #   x = norm1(x)
        #   x, _ = attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        #   x = dropout(x)
        #   x = residual + x

        # TODO: MoE FFN 子层（同上结构）
        #   residual = x
        #   x = norm2(x)
        #   moe_out, aux_loss = moe(x)
        #   x = dropout(moe_out)
        #   x = residual + x

        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
# 测试（全部通过 = 实现正确）
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = 256
    num_heads = 4
    d_ff = 512
    num_experts = 8
    top_k = 2
    B, T = 2, 16

    print("=" * 60)
    print("MoE 练习文件测试")
    print("=" * 60)

    print("\n[1] Expert")
    expert = Expert(d_model, d_ff).to(device)
    out = expert(torch.randn(10, d_model, device=device))
    assert out.shape == (10, d_model), f"期望 (10, {d_model})，得到 {out.shape}"
    print(f"  通过：输出形状 {out.shape}")

    print("\n[2] TopKRouter")
    router = TopKRouter(d_model, num_experts, top_k).to(device)
    x_flat = torch.randn(B * T, d_model, device=device)
    indices, weights, probs = router(x_flat)
    assert indices.shape == (B * T, top_k)
    assert weights.shape == (B * T, top_k)
    assert probs.shape == (B * T, num_experts)
    weight_sum = weights.sum(dim=-1)
    assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-5), \
        f"权重之和应为 1，实际：{weight_sum[:3]}"
    print(f"  通过：indices {indices.shape}, weights {weights.shape}, probs {probs.shape}")

    print("\n[3] load_balance_loss")
    loss_val = load_balance_loss(probs, indices, num_experts)
    assert loss_val.shape == (), f"应为标量，得到 {loss_val.shape}"
    assert loss_val.item() > 0, "损失应大于 0"
    print(f"  通过：aux_loss = {loss_val.item():.6f}")

    print("\n[4] MoELayer")
    moe = MoELayer(d_model, d_ff, num_experts, top_k).to(device)
    x = torch.randn(B, T, d_model, device=device)
    moe_out, moe_aux = moe(x)
    assert moe_out.shape == x.shape, f"期望 {x.shape}，得到 {moe_out.shape}"
    print(f"  通过：输出形状 {moe_out.shape}, aux_loss = {moe_aux.item():.6f}")

    print("\n[5] MoETransformerBlock")
    block = MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k).to(device)
    causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
    blk_out, blk_aux = block(x, attn_mask=causal_mask)
    assert blk_out.shape == x.shape, f"期望 {x.shape}，得到 {blk_out.shape}"
    print(f"  通过：输出形状 {blk_out.shape}, aux_loss = {blk_aux.item():.6f}")

    print("\n[6] 反向传播")
    total_loss = blk_out.mean() + blk_aux
    total_loss.backward()
    assert block.moe.router.gate.weight.grad is not None, "router 应有梯度"
    print(f"  通过：total_loss = {total_loss.item():.6f}，梯度正常流通")

    print("\n所有测试通过！")