"""
混合专家模型（Mixture of Experts，MoE）实现

# ═══════════════════════════════════════════════════════════════════
# 一、MoE 核心思想
# ═══════════════════════════════════════════════════════════════════
#
# 传统 FFN（前馈网络）：每个 token 都经过同一套参数计算。
#
#   token → FFN(W1, W2) → output          参数量 = O(d_model * d_ff)
#
# MoE 的做法：把一个大 FFN 拆成 N 个"专家"（小 FFN），
# 每个 token 只激活其中 top-k 个专家，其余专家不计算。
#
#   token → Router → 选出 top-k 专家 → 加权求和 → output
#
# 好处：参数量扩大 N 倍（更强的表达能力），
#       但每个 token 的计算量只增加 k/N 倍（计算效率不变）。
#
# 代表模型：
#   - Mixtral-8x7B：8 个专家，每次激活 2 个（top-2）
#   - DeepSeek-V2/V3：细粒度 MoE，专家数更多（64/256），top-k 更小
#   - Switch Transformer（Google）：top-1，极致计算效率
#
# ═══════════════════════════════════════════════════════════════════
# 二、整体数据流
# ═══════════════════════════════════════════════════════════════════
#
#  输入 x: [B, T, d_model]
#       │
#       ▼
#  ┌─────────────────────────────────────────────────────┐
#  │  Router（路由器）                                    │
#  │    线性层: [B*T, d_model] → [B*T, num_experts]       │
#  │    softmax → 每个 token 对各专家的"亲和度"得分        │
#  │    top-k   → 选出得分最高的 k 个专家及其权重          │
#  └─────────────────────────────────────────────────────┘
#       │  indices: [B*T, k]   weights: [B*T, k]
#       ▼
#  ┌─────────────────────────────────────────────────────┐
#  │  Expert Dispatch（专家分发）                          │
#  │    按专家 id 把 token 分组，分别送入对应专家计算       │
#  │    Expert_i: Linear(d_model→d_ff) → ReLU/SiLU        │
#  │             → Linear(d_ff→d_model)                   │
#  └─────────────────────────────────────────────────────┘
#       │  各专家输出: [tokens_for_expert_i, d_model]
#       ▼
#  ┌─────────────────────────────────────────────────────┐
#  │  Weighted Combine（加权合并）                         │
#  │    把 k 个专家的输出 × 路由权重后相加                 │
#  └─────────────────────────────────────────────────────┘
#       │
#       ▼
#  输出 x: [B, T, d_model]
#
# ═══════════════════════════════════════════════════════════════════
# 三、负载均衡损失（Auxiliary Load Balancing Loss）
# ═══════════════════════════════════════════════════════════════════
#
# 问题：如果不加约束，Router 会"坍塌"到总选同一两个专家，
#       其他专家永远得不到训练 → 专家利用率极低。
#
# 解决：在训练损失中加一项辅助损失，惩罚专家负载不均：
#
#   aux_loss = alpha * num_experts * sum_i(f_i * p_i)
#
#   f_i = 实际被分配到专家 i 的 token 比例（硬选择，不可微）
#   p_i = 路由器给专家 i 的平均概率（可微，用于反向传播）
#
# 直觉：如果专家 i 被选了很多次（f_i 大），但路由器给它的概率也大（p_i 大），
#       乘积就大，损失就大，迫使路由器把概率分散到其他专家。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1. 单个专家（Expert FFN）
# ══════════════════════════════════════════════════════════════════════════════

class Expert(nn.Module):
    """
    单个专家 = 一个两层前馈网络（FFN）。

    结构：Linear(d_model → d_ff) → 激活函数 → Linear(d_ff → d_model)

    与 Transformer 标准 FFN 完全一致，MoE 只是把 N 个这样的 FFN 并排放，
    每个 token 只走其中 k 个。

    参数量：2 * d_model * d_ff（每个专家）
    N 个专家总参数量：N * 2 * d_model * d_ff
    但每个 token 只计算 k 个专家 → 计算量 = k/N * 总参数量
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: 输入/输出维度（与主干模型 hidden_size 一致）
            d_ff:    FFN 中间层维度（通常为 d_model 的 4 倍）
        """
        super().__init__()
        # 第一层：升维，学习特征组合
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # 第二层：降维，映射回 d_model
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        # SiLU（Swish）：现代 LLM 常用激活，比 ReLU 更平滑
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_tokens, d_model]  —— 分配给本专家的 token 集合
        Returns:
            [num_tokens, d_model]
        """
        # [N, d_model] → [N, d_ff] → 激活 → [N, d_model]
        return self.w2(self.act(self.w1(x)))


# ══════════════════════════════════════════════════════════════════════════════
# 2. 路由器（Router / Gate）
# ══════════════════════════════════════════════════════════════════════════════

class TopKRouter(nn.Module):
    """
    Top-K 路由器：为每个 token 选出 k 个专家，并计算归一化权重。

    核心步骤：
      1. 线性层把 token 映射到 num_experts 维的 logits
      2. softmax → 概率分布（用于辅助损失）
      3. top-k 选出得分最高的 k 个专家
      4. 对选出的 k 个分数再做 softmax → 路由权重（用于加权求和）

    为什么要做两次 softmax？
      - 第一次（全局）：用于计算辅助损失，需要反映所有专家的竞争关系
      - 第二次（局部）：只对选中的 k 个分数归一化，使权重之和为 1，
        方便加权求和时保持输出的数值稳定
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        """
        Args:
            d_model:     输入维度
            num_experts: 专家总数 N
            top_k:       每个 token 激活的专家数 k（通常 1 或 2）
        """
        super().__init__()
        assert top_k <= num_experts, "top_k 不能超过专家总数"
        self.num_experts = num_experts
        self.top_k = top_k
        # 路由线性层：把 token 映射到每个专家的"亲和度"分数
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [num_tokens, d_model]

        Returns:
            topk_indices:  [num_tokens, top_k]  — 每个 token 选中的专家索引
            topk_weights:  [num_tokens, top_k]  — 对应的路由权重（归一化后）
            router_probs:  [num_tokens, num_experts] — 全局 softmax 概率（用于辅助损失）
        """
        # Step 1: 计算 logits
        # [num_tokens, d_model] → [num_tokens, num_experts]
        logits = self.gate(x)

        # Step 2: 全局 softmax → 每个专家的路由概率
        # 这个概率用于计算负载均衡辅助损失（需要可微）
        router_probs = F.softmax(logits, dim=-1)  # [num_tokens, num_experts]

        # Step 3: top-k 选出分数最高的 k 个专家
        # topk_weights_raw: 被选中的 k 个专家的原始 softmax 概率
        # topk_indices: 被选中的专家编号（0 ~ num_experts-1）
        topk_weights_raw, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        # topk_weights_raw: [num_tokens, top_k]
        # topk_indices:     [num_tokens, top_k]

        # Step 4: 对选中的 k 个权重再做归一化，使其和为 1
        # 动机：top-k 截断后，k 个概率之和 < 1，需要重新归一化
        # 这样加权求和后输出的数值范围与输入一致，训练更稳定
        topk_weights = topk_weights_raw / topk_weights_raw.sum(dim=-1, keepdim=True)
        # topk_weights: [num_tokens, top_k]，每行之和为 1

        return topk_indices, topk_weights, router_probs


# ══════════════════════════════════════════════════════════════════════════════
# 3. 负载均衡辅助损失
# ══════════════════════════════════════════════════════════════════════════════

def load_balance_loss(
    router_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Switch Transformer 风格的负载均衡损失。

    公式：aux_loss = num_experts * sum_i( f_i * p_i )

    变量含义：
        f_i：专家 i 实际被分配的 token 比例（离散，不可微）
             = (被路由到专家 i 的 token 数) / (总 token 数 * top_k)
        p_i：路由器给专家 i 的平均概率（连续，可微，用于梯度传播）
             = mean over all tokens of router_probs[:, i]

    乘以 num_experts 是为了让损失值不随专家数量变化而缩放
    （期望均匀分布时，每项 f_i * p_i ≈ 1/N²，乘 N 后 ≈ 1/N）

    Args:
        router_probs:  [num_tokens, num_experts]  全局路由概率
        topk_indices:  [num_tokens, top_k]        实际选中的专家索引
        num_experts:   专家总数 N

    Returns:
        标量损失值
    """
    num_tokens = router_probs.shape[0]

    # ── 计算 f_i：每个专家实际收到的 token 比例（one-hot 统计）──────────────
    # 把 topk_indices 展平后做 one-hot，统计每个专家被选中的次数
    # topk_indices: [num_tokens, top_k] → [num_tokens * top_k]
    flat_indices = topk_indices.reshape(-1)

    # one_hot: [num_tokens * top_k, num_experts]
    one_hot = F.one_hot(flat_indices, num_classes=num_experts).float()

    # 每个专家被选中的总次数 → 归一化为比例
    # tokens_per_expert: [num_experts]
    tokens_per_expert = one_hot.sum(dim=0)
    # 分母 = 总 token 数 × top_k（每个 token 选 k 个，总选择次数）
    f = tokens_per_expert / (num_tokens * topk_indices.shape[1])

    # ── 计算 p_i：路由器对每个专家的平均概率（可微）─────────────────────────
    # router_probs: [num_tokens, num_experts]，沿 token 维度取平均
    p = router_probs.mean(dim=0)  # [num_experts]

    # ── 计算辅助损失 ──────────────────────────────────────────────────────────
    # f.detach()：f_i 通过 one-hot 计算，不可微，detach 后仅作系数
    # p 保持可微，梯度通过 p 回传给路由器，迫使路由器均衡分配
    aux_loss = num_experts * (f.detach() * p).sum()

    return aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# 4. MoE 层（整合路由 + 专家分发 + 加权合并）
# ══════════════════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """
    完整的 MoE 层，用于替换 Transformer 中的标准 FFN 层。

    使用方式：
        # 标准 Transformer block
        x = x + attention(norm(x))
        x = x + ffn(norm(x))          # ← 把这里的 ffn 换成 MoELayer

        # MoE Transformer block
        x = x + attention(norm(x))
        moe_out, aux_loss = moe_layer(norm(x))
        x = x + moe_out               # ← 结构完全一致，只是 FFN 变成了 MoE
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        aux_loss_alpha: float = 0.01,
    ):
        """
        Args:
            d_model:        输入/输出维度
            d_ff:           每个专家 FFN 的中间层维度
            num_experts:    专家总数 N（如 8、64、256）
            top_k:          每个 token 激活的专家数（如 2）
            aux_loss_alpha: 负载均衡损失的权重系数（通常 0.01 ~ 0.001）
        """
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha

        # 路由器：决定每个 token 去哪些专家
        self.router = TopKRouter(d_model, num_experts, top_k)

        # 专家池：N 个独立的 FFN
        # nn.ModuleList 让 PyTorch 能正确追踪所有专家的参数
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]  — 来自上一层（通常经过 LayerNorm）的输入

        Returns:
            output:   [B, T, d_model]  — MoE 层的输出
            aux_loss: 标量            — 负载均衡辅助损失（训练时加到总损失上）
        """
        B, T, d_model = x.shape

        # ── 展平 batch 和 seq 维度，方便按 token 路由 ─────────────────────────
        # [B, T, d_model] → [B*T, d_model]
        # 路由是逐 token 独立进行的，不需要区分 batch
        x_flat = x.reshape(B * T, d_model)
        num_tokens = x_flat.shape[0]  # = B * T

        # ── Step 1: 路由 → 得到每个 token 的专家分配 ────────────────────────
        # topk_indices:  [num_tokens, top_k]   每个 token 选中的专家编号
        # topk_weights:  [num_tokens, top_k]   对应的路由权重（和为 1）
        # router_probs:  [num_tokens, num_experts]  全局概率（用于辅助损失）
        topk_indices, topk_weights, router_probs = self.router(x_flat)

        # ── Step 2: 初始化输出张量 ────────────────────────────────────────────
        # 先分配一个全零输出，后续累加各专家的加权结果
        output_flat = torch.zeros_like(x_flat)  # [num_tokens, d_model]

        # ── Step 3: 按专家分发 token，分别计算，加权累加 ─────────────────────
        #
        # 核心循环思路：
        #   对每个专家 i，找出所有"被路由到专家 i"的 token，
        #   批量送入专家 i 计算，再按路由权重加回 output_flat。
        #
        # 为什么不能简单地 for token in tokens: forward(expert)?
        #   矩阵乘法是 GPU 最高效的操作，逐 token 串行计算极慢。
        #   按专家分组后，每个专家可以对其所有 token 做一次批量矩阵乘法。
        #
        for expert_id in range(self.num_experts):

            # 找出在 top-k 列表中含有 expert_id 的 token
            # topk_indices: [num_tokens, top_k]
            # expert_mask:  [num_tokens, top_k]，True 表示该位置选中了 expert_id
            expert_mask = (topk_indices == expert_id)

            # token_mask: [num_tokens]，该 token 是否（在任意一个 top-k 槽中）选了 expert_id
            token_mask = expert_mask.any(dim=-1)

            # 如果没有 token 被路由到该专家，跳过（节省计算）
            if not token_mask.any():
                continue

            # 取出被路由到专家 i 的 token
            expert_input = x_flat[token_mask]  # [n_i, d_model]，n_i ≤ num_tokens

            # 专家 i 前向计算
            expert_output = self.experts[expert_id](expert_input)  # [n_i, d_model]

            # 取出这些 token 对专家 i 的路由权重
            # expert_mask[token_mask]: [n_i, top_k]，只看被选中的 token
            # 对每个被选中的 token，找它在 top-k 列表中指向 expert_id 的那个槽的权重
            weights_for_expert = topk_weights[token_mask][expert_mask[token_mask]]
            # weights_for_expert: [n_i]

            # 加权后累加到输出
            # unsqueeze(-1) 让 [n_i] 广播到 [n_i, d_model]
            output_flat[token_mask] += expert_output * weights_for_expert.unsqueeze(-1)

        # ── Step 4: 恢复 [B, T, d_model] 形状 ───────────────────────────────
        output = output_flat.reshape(B, T, d_model)

        # ── Step 5: 计算负载均衡辅助损失 ─────────────────────────────────────
        aux_loss = self.aux_loss_alpha * load_balance_loss(
            router_probs, topk_indices, self.num_experts
        )

        return output, aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# 5. 完整 MoE Transformer Block（展示 MoE 如何嵌入标准 Transformer）
# ══════════════════════════════════════════════════════════════════════════════

class MoETransformerBlock(nn.Module):
    """
    一个完整的 MoE Transformer Block：

        x → LayerNorm → MultiHeadAttention → 残差连接
          → LayerNorm → MoELayer           → 残差连接
          → output, aux_loss

    与标准 Transformer Block 的唯一区别：FFN → MoELayer。
    注意：forward 返回 aux_loss，调用方需要把它加到总训练损失上。
    """

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

        # ── 自注意力子层 ──────────────────────────────────────────────────────
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # ── MoE FFN 子层 ──────────────────────────────────────────────────────
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, aux_loss_alpha)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:         [B, T, d_model]
            attn_mask: [T, T] 因果掩码（可选）

        Returns:
            output:   [B, T, d_model]
            aux_loss: 标量，MoE 负载均衡损失
        """
        # ── 自注意力 + 残差 ───────────────────────────────────────────────────
        residual = x
        x = self.norm1(x)
        # nn.MultiheadAttention 返回 (attn_output, attn_weights)，只取输出
        x, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.dropout(x)
        x = residual + x

        # ── MoE FFN + 残差 ────────────────────────────────────────────────────
        residual = x
        x = self.norm2(x)
        moe_out, aux_loss = self.moe(x)
        x = self.dropout(moe_out)
        x = residual + x

        return x, aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# 6. 测试入口
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 配置参数（Mixtral-8x7B 风格的小尺度版本）────────────────────────────
    d_model = 256        # 隐藏层维度（Mixtral 原版 4096）
    num_heads = 4        # 注意力头数（Mixtral 原版 32）
    d_ff = 512           # 每个专家 FFN 中间层（Mixtral 原版 14336）
    num_experts = 8      # 专家总数（Mixtral 原版 8）
    top_k = 2            # 每个 token 激活的专家数（Mixtral 原版 2）
    batch_size = 2
    seq_len = 16

    print("=" * 60)
    print("MoE 各组件单元测试")
    print("=" * 60)

    # ── 测试 1: 单个专家 ──────────────────────────────────────────────────────
    print("\n[1] Expert FFN")
    expert = Expert(d_model, d_ff).to(device)
    dummy_tokens = torch.randn(10, d_model, device=device)
    expert_out = expert(dummy_tokens)
    print(f"  输入: {dummy_tokens.shape} → 输出: {expert_out.shape}")
    assert expert_out.shape == (10, d_model)

    # ── 测试 2: 路由器 ────────────────────────────────────────────────────────
    print("\n[2] TopKRouter")
    router = TopKRouter(d_model, num_experts, top_k).to(device)
    dummy_flat = torch.randn(batch_size * seq_len, d_model, device=device)
    indices, weights, probs = router(dummy_flat)
    print(f"  输入: {dummy_flat.shape}")
    print(f"  topk_indices:  {indices.shape}  (每个 token 选 {top_k} 个专家)")
    print(f"  topk_weights:  {weights.shape}  (权重之和应为 1)")
    print(f"  router_probs:  {probs.shape}   (全局 softmax 概率)")
    print(f"  权重求和验证:  {weights.sum(dim=-1).mean().item():.4f}  (应为 1.0)")

    # ── 测试 3: 负载均衡损失 ──────────────────────────────────────────────────
    print("\n[3] Load Balance Loss")
    aux = load_balance_loss(probs, indices, num_experts)
    print(f"  aux_loss = {aux.item():.6f}")
    print(f"  均匀分布期望值 ≈ 1.0（N * 1/N * 1/N * N = 1）")

    # ── 测试 4: MoE 层 ────────────────────────────────────────────────────────
    print("\n[4] MoELayer")
    moe_layer = MoELayer(d_model, d_ff, num_experts, top_k).to(device)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    moe_out, moe_aux = moe_layer(x)
    print(f"  输入: {x.shape} → 输出: {moe_out.shape}")
    print(f"  aux_loss = {moe_aux.item():.6f}")
    assert moe_out.shape == x.shape

    # ── 测试 5: 完整 MoE Transformer Block ───────────────────────────────────
    print("\n[5] MoETransformerBlock")
    block = MoETransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_experts=num_experts,
        top_k=top_k,
        dropout=0.0,
    ).to(device)

    # 构造因果掩码（上三角为 -inf，下三角为 0）
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device),
        diagonal=1,
    )

    block_out, block_aux = block(x, attn_mask=causal_mask)
    print(f"  输入: {x.shape} → 输出: {block_out.shape}")
    print(f"  aux_loss = {block_aux.item():.6f}")
    assert block_out.shape == x.shape

    # ── 测试 6: 反向传播（验证梯度流通）─────────────────────────────────────
    print("\n[6] 反向传播验证")
    # 模拟任务损失 + 辅助损失的联合训练
    task_loss = block_out.mean()
    total_loss = task_loss + block_aux
    total_loss.backward()
    print(f"  task_loss = {task_loss.item():.6f}")
    print(f"  total_loss = {total_loss.item():.6f}")
    print(f"  router.gate.weight.grad 不为 None: {block.moe.router.gate.weight.grad is not None}")

    # ── 测试 7: 专家负载分布统计 ──────────────────────────────────────────────
    print("\n[7] 专家负载分布（越均匀越好）")
    with torch.no_grad():
        _, _, probs_block = block.moe.router(x.reshape(-1, d_model))
        _, topk_idx_block = torch.topk(probs_block, top_k, dim=-1)
        counts = torch.bincount(topk_idx_block.reshape(-1), minlength=num_experts)
        total = counts.sum().item()
        print(f"  各专家被选次数（共 {total} 次选择，{num_experts} 个专家）:")
        for i, c in enumerate(counts.tolist()):
            bar = "█" * int(c / total * 40)
            print(f"  专家 {i}: {c:4d}次  {bar}")

    print("\n所有测试通过！")