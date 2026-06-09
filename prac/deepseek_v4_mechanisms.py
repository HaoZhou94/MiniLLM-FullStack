"""
DeepSeek-V4 核心机制教学脚本

涵盖 DeepSeek 系列最关键的 3 大创新（MLA 和 MoE 已在单独文件中覆盖）：

  Part 1 ── Multi-Token Prediction (MTP)
             训练时同时预测未来多个 token，提升推理时的 speculative decoding 效率
  Part 2 ── GRPO (Group Relative Policy Optimization)
             去掉 Critic 网络的强化学习算法，R1 的核心训练方法
  Part 3 ── Auxiliary-Loss-Free Load Balancing
             不引入辅助损失，通过动态偏置实现 MoE 专家负载均衡

每个 Part 的结构：【讲解】 → 【填空练习】 → 【验证测试】

使用方法：
  python prac/deepseek_v4_mechanisms.py
  完成每个 TODO 后运行，看测试是否通过。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"运行设备: {device}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: Multi-Token Prediction (MTP)
# ══════════════════════════════════════════════════════════════════════════════
#
# 【背景】
#   传统语言模型（GPT、LLaMA 等）都是"下一个 token 预测"（Next-Token Prediction）：
#     输入 "今天天气真"，目标 "好"
#     每个位置只预测紧随其后的一个 token。
#
#   DeepSeek-V3 的 MTP 思路很直接：
#     与其只预测下一个 token，不如同时预测后面 D 个 token！
#     输入 "今天天气真好" → 同时预测 "天"、"气"、"真"、"好"
#
# 【核心洞察】
#   MTP 不是一个独立模型，而是在主模型上添加 D 个"浅层"预测头（MTP Module）。
#   每个 MTP Module 共享主模型的 embedding 和输出层，但有自己的 Transformer Layer。
#
#   数学上，第 k 个 MTP Module 的预测：
#     h_k^i = MTP_k( h_{k-1}^i, emb(t_{i+k}) )
#     其中 h_{k-1}^i 是主模型在第 i 个位置的 hidden state
#           emb(t_{i+k}) 是第 i+k 个 token 的 embedding
#
#   训练时每个 MTP Module 独立计算交叉熵损失，总损失 = 主损失 + sum(MTP 损失)
#
# 【为什么 MTP 有效？】
#   1. 训练信号更密集：每个位置提供 D 个训练信号，而不是 1 个
#   2. 学习长远依赖：预测更远的 token 迫使模型学到更高层的语义结构
#   3. 推理加速：训练好的 MTP 头可以作为 speculative drafting model，实现
#      speculative decoding（草稿→验证），大幅提升推理吞吐
#
# 【数据流】
#   ●───●───●───●───●  主模型 trunk
#   │   │   │   │   │
#   ●   ●   ●   ●   ●  MTP Module 1（预测下一个 token）
#   │   │   │   │   │
#   ●   ●   ●   ●   ●  MTP Module 2（预测下两个 token）
#   │   │   │   │   │
#   ●   ●   ●   ●   ●  MTP Module 3（预测下三个 token）
#
#   ⚡ 每个 MTP Module 不是独立的大模型 —— 它们只有 1 层 Transformer + 1 个 Linear
#   ⚡ 但每个 Module 有自己独立的参数，串联起来形成"阶梯式"预测
# =============================================================================


class RMSNorm(nn.Module):
    """RMS Layer Normalization（DeepSeek 系列标配，无偏置、无均值减法）"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MTPModule(nn.Module):
    """
    单一 MTP 预测模块。

    每个模块包含：
      1. 一个 RMSNorm（归一化）
      2. 一个 Attention 层
      3. 一个 FFN 层
      4. 一个输出 LM Head（linear → vocab）

    输入是  (主模型 hidden_state + 目标 token embedding) 的融合。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, vocab_size: int):
        super().__init__()
        # 输入融合：先把 hidden_state 和目标 token embedding 相加
        self.input_norm = RMSNorm(d_model)

        # 单层 Transformer
        self.attn_norm = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        # 输出预测头（与主模型共享 vocab）
        self.output_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self,
                hidden_states: torch.Tensor,
                target_embed: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, T, d_model]  主模型当前层的输出
            target_embed:  [B, T, d_model]  目标 token 的 embedding（要预测的位置的 token）
            attention_mask: 因果注意力掩码

        Returns:
            output:       [B, T, d_model]  处理后的 hidden states
            logits:       [B, T, vocab]    预测 logits
        """
        # Step 1: 融合输入 —— 将 hidden_state 与 target embedding 相加
        # 注意：这不是简单的加法，而是对两者分别处理后相加
        # 这里我们做简化：先 norm hidden_states，再与 target_embed 相加

        # ── TODO 1.1: 实现输入融合 ─────────────────────────────────────
        # hidden_states: [B, T, d_model], target_embed: [B, T, d_model]
        # 对 hidden_states 做 input_norm，然后与 target_embed 相加
        # Hint: self.input_norm(hidden_states) + target_embed
        x = self.input_norm(hidden_states) + target_embed
        # ───────────────────────────────────────────────────────────────

        residual = x

        # Step 2: Self-Attention
        x = self.attn_norm(x)
        x, _ = self.attn(x, x, x, attn_mask=attention_mask, need_weights=False)
        x = x + residual

        residual = x

        # Step 3: FFN
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual

        # Step 4: Output → logits
        x = self.output_norm(x)
        logits = self.lm_head(x)

        return x, logits


class MultiTokenPrediction(nn.Module):
    """
    完整的 MTP 模块集合。

    D 个 MTP Module 级联排列，每个模块预测 D 步后的 token。
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 vocab_size: int, num_mtp_modules: int = 3):
        super().__init__()
        self.num_mtp_modules = num_mtp_modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(d_model, n_heads, d_ff, vocab_size)
            for _ in range(num_mtp_modules)
        ])

    def forward(self,
                main_model_hidden: torch.Tensor,
                input_ids: torch.Tensor,
                embedding_layer: nn.Embedding,
                attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Args:
            main_model_hidden: [B, T, d_model]  主模型最后一层的输出
            input_ids:         [B, T + D]        输入 token 序列（包含 D 个额外 token）
            embedding_layer:   共享的 embedding 层
            attention_mask:    因果注意力掩码

        Returns:
            all_logits: [D] 个 logits 列表，每个 [B, T, vocab_size]
        """
        # MTP 的关键：第 k 个模块用 hidden_states_{k-1} 和 emb(t_{i+k}) 作为输入
        # t_{i+k} 是"往后 k 步"的 token

        all_logits = []
        hidden_k = main_model_hidden  # h_0 = 主模型输出

        for k in range(self.num_mtp_modules):
            # 目标 token 是 input_ids 中偏移 (k+1) 的位置
            # 比如 k=0 时，目标是下一个 token: input_ids[:, 1:]
            #     k=1 时，目标是下两个 token: input_ids[:, 2:]
            # 等等……
            # 我们需要对齐长度：hidden 是 [B, T, d]，偏移后的目标也是 [B, T]

            # ── TODO 1.2: 获取第 k 个 MTP 的目标 token ────────────────
            # 从 input_ids 中切片出偏移 (k+1) 的位置，shape: [B, T]
            # target_ids = input_ids[:, k+1 : k+1 + hidden_k.size(1)]
            target_ids = input_ids[:, k + 1: k + 1 + hidden_k.size(1)]
            # ───────────────────────────────────────────────────────────

            # 通过共享 embedding 层转为向量
            target_embed = embedding_layer(target_ids)  # [B, T, d_model]

            # 送入第 k 个 MTP 模块
            hidden_k, logits = self.mtp_modules[k](
                hidden_k, target_embed, attention_mask
            )
            all_logits.append(logits)

        return all_logits

    def compute_mtp_loss(self,
                         all_logits: List[torch.Tensor],
                         input_ids: torch.Tensor,
                         loss_weights: Optional[List[float]] = None) -> torch.Tensor:
        """
        计算 MTP 的总损失 = sum( w_k * CE_loss_k )

        Args:
            all_logits: [D] 个 logits 的列表
            input_ids:  [B, T + D]  完整输入序列
            loss_weights: 各模块损失的权重，默认 w_k = 1/(k+1)（越远权重越低）

        Returns:
            标量总损失
        """
        B, T = all_logits[0].shape[:2]
        D = len(all_logits)

        if loss_weights is None:
            # 默认权重：越远的预测权重越低
            loss_weights = [1.0 / (k + 1) for k in range(D)]

        total_loss = 0.0
        for k in range(D):
            # ── TODO 1.3: 计算第 k 个 MTP 头的损失 ────────────────────
            # 目标 token = input_ids[:, k+1 : k+1+T]
            # logits = all_logits[k]，shape [B, T, vocab_size]
            # 使用 F.cross_entropy
            # Hint: F.cross_entropy 需要 logits [B*T, vocab], targets [B*T]
            targets = input_ids[:, k + 1: k + 1 + T].reshape(-1)
            logits = all_logits[k].reshape(-1, all_logits[k].size(-1))
            loss = F.cross_entropy(logits, targets)
            # ───────────────────────────────────────────────────────────

            total_loss += loss_weights[k] * loss

        return total_loss


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: GRPO（Group Relative Policy Optimization）
# ══════════════════════════════════════════════════════════════════════════════
#
# 【背景】
#   传统 RLHF 使用 PPO，需要 4 个模型：
#     Policy（Actor）、Reference（冻结的 Actor）、Critic（价值网络）、Reward
#   Critic 的价值估计不准确 → 训练不稳定 → 需要大量调参
#
#   DeepSeek-R1 的 GRPO 彻底去掉了 Critic！
#     用"同一 prompt 的多个采样结果"的组内相对分数来代替价值估计。
#
# 【核心思想】
#   对每个 prompt，从 Policy 采样 G 个回答。
#   计算每个回答的 reward，然后组内归一化（减均值、除标准差）。
#   用这个归一化后的"优势"来更新 Policy。
#
#   这样做的好处：
#     1. 不需要 Critic 网络 → 参数量减半，训练更稳定
#     2. 组内比较天然消除了 prompt 难度差异的影响
#     3. 简单！实现只需要几十行代码
#
# 【GRPO 损失函数】
#   L_GRPO = -1/G * sum_i=1^G [ min( r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i ) ] + β * KL(π_θ || π_ref)
#
#   其中：
#     r_i = π_θ(y_i|x) / π_old(y_i|x)   ← 重要性采样比（importance sampling ratio）
#     A_i = (R_i - mean(R)) / std(R)     ← 组内归一化优势（没有 Critic！）
#     KL 散度用近似公式，不需要 ref model 每次都 forward
#
# 【和 PPO 的关键区别】
#   PPO:   Advantage = R - V(s)          ← 需要 Critic 网络 V(s)
#   GRPO:  Advantage = (R_i - μ_R) / σ_R  ← 只需要组内统计量！
#
# 【近似 KL 散度】
#   KL(π_θ || π_ref) ≈ (π_ref/π_θ - 1) - log(π_ref/π_θ)
#   这是 KL 的一阶泰勒展开近似，不要求 π_ref 前向传播
# =============================================================================


def grpo_loss(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    eps: float = 0.2,
    kl_beta: float = 0.01,
) -> torch.Tensor:
    """
    GRPO 损失函数。

    Args:
        log_probs:     [B, G, T]  当前 policy 的 log 概率
        ref_log_probs: [B, G, T]  参考 policy（冻结）的 log 概率
        rewards:       [B, G]     每个回答的 reward 分数
        eps:           PPO-clip 的裁剪阈值
        kl_beta:       KL 惩罚系数

    Returns:
        标量损失值
    """

    B, G, T = log_probs.shape

    # ── Step 1: 计算重要性采样比 r_i ──────────────────────────────────
    # r_i = exp(log_probs - ref_log_probs)  （沿 T 维求和）
    # log_probs / ref_log_probs 都是 [B, G, T]，我们先沿序列求和

    # ── TODO 2.1: 计算每个回答的总概率（沿序列维度求和） ─────────────
    # sum_log_probs: [B, G], sum_ref_log_probs: [B, G]
    sum_log_probs = log_probs.sum(dim=-1)
    sum_ref_log_probs = ref_log_probs.sum(dim=-1)
    # ───────────────────────────────────────────────────────────────────

    # 重要性采样比
    ratio = torch.exp(sum_log_probs - sum_ref_log_probs)  # [B, G]

    # ── Step 2: 组内归一化（这就是 GRPO 名字的来源！） ────────────────
    # 优势 A_i = (R_i - mean_group(R)) / std_group(R)
    # 对每个 prompt（batch 中的每一行），独立计算组内均值和标准差

    # ── TODO 2.2: 实现组内归一化 ──────────────────────────────────────
    # rewards: [B, G]，对每个 batch 行（dim=1）计算均值和标准差
    # 然后 A = (rewards - mean) / (std + 1e-8)
    # Hint: 保持 shape [B, G] 可广播
    mean_rewards = rewards.mean(dim=-1, keepdim=True)       # [B, 1]
    std_rewards = rewards.std(dim=-1, keepdim=True)         # [B, 1]
    advantages = (rewards - mean_rewards) / (std_rewards + 1e-8)  # [B, G]
    # ─────────────────────────────────────────────────────────────────

    # ── Step 3: Clipped PPO 损失 ──────────────────────────────────────
    # L_policy = -1/G * sum_i min(r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i)
    # 负号是因为我们要"最小化"损失，而目标是"最大化"期望 reward

    # ── TODO 2.3: 计算 clipped 策略损失 ──────────────────────────────
    # ratio: [B, G], advantages: [B, G] (unsqueeze 到 [B, G] 或广播)
    # 1. 计算 ratio * advantages
    # 2. 计算 clipped_ratio * advantages
    #    clipped_ratio = clamp(ratio, 1-eps, 1+eps)
    # 3. 取两者的 min，然后 mean over [B, G]
    surr1 = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1 - eps, 1 + eps)
    surr2 = clipped_ratio * advantages
    pg_loss = -torch.min(surr1, surr2).mean()
    # ──────────────────────────────────────────────────────────────────

    # ── Step 4: 近似 KL 散度惩罚 ──────────────────────────────────────
    # KL(π_θ || π_ref) ≈ (π_ref/π_θ - 1) - log(π_ref/π_θ)
    # 使用近似公式避免额外 forward 计算
    # 注意这里用逐 token 的 log_probs，而不是 sum

    # ── TODO 2.4: 计算近似 KL 散度 ───────────────────────────────────
    # log_probs / ref_log_probs: [B, G, T]
    # 近似 KL = exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    # 然后取 mean
    kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
    kl_loss = kl_div.mean()
    # ──────────────────────────────────────────────────────────────────

    total_loss = pg_loss + kl_beta * kl_loss
    return total_loss


# ══════════════════════════════════════════════════════════════════════════════
# Part 3: Auxiliary-Loss-Free Load Balancing（无辅助损失的负载均衡）
# ══════════════════════════════════════════════════════════════════════════════
#
# 【背景】
#   传统 MoE（Switch Transformer、Mixtral 等）使用辅助损失（auxiliary loss）来
#   平衡专家负载。这有个根本矛盾：
#     辅助损失鼓励负载均衡 ←→ 主损失（语言建模）鼓励专业化
#     两个目标互相拉扯，训练不稳定，且需要精细调节辅助损失系数 α
#
# 【DeepSeek-V3 的创新方案】
#   思路极其简单：
#     路由器输出 logits 后，加上一个"可学习的偏置项"（bias），
#     根据每个专家当前的负载，动态调整这个偏置。
#
#   如果专家 i 过载（收到的 token 过多）→ 降低偏置 b_i → 该专家被选中的概率降低
#   如果专家 i 欠载（收到的 token 过少）→ 提高偏置 b_i → 该专家被选中的概率升高
#
# 【关键优点】
#   1. 不需要辅助损失 → 没有 α 超参数需要调节
#   2. 偏置调整是确定性规则（类似 batch normalization 的 moving average），
#      不参与梯度计算，完全独立于反向传播
#   3. 训练更稳定，效果更好
#
# 【算法】
#   每个 step 结束后：
#     统计每个专家收到的 token 数量
#     如果专家 i 超过平均负载 → b_i -= 衰减率
#     如果专家 i 低于平均负载 → b_i += 衰减率
#   这个过程类似"水位调节"，非常直觉。
#
# 【对比】
#             传统方式                          DeepSeek 方式
#   ┌───────────────────────┐    ┌───────────────────────────┐
#   │ router logits → softmax│    │ router logits + bias      │
#   │ top-k 选择             │    │ → softmax                 │
#   │ 辅助损失 = f * p       │    │ → top-k 选择              │
#   │ 总损失 = 主损失 + α*辅助 │    │ → 动态调节 bias（不参与梯度）│
#   └───────────────────────┘    └─────────────────────────────┘
# =============================================================================


class AuxiliaryFreeRouter(nn.Module):
    """
    无辅助损失的 MoE 路由器。

    在 DeepSeek-V3 中，路由器的 logits 上额外加了一个可学习的偏置向量，
    根据负载动态调节这个偏置来实现负载均衡，无需辅助损失。
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        bias_update_rate: float = 0.001,
        bias_epsilon: float = 0.02,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias_update_rate = bias_update_rate
        self.bias_epsilon = bias_epsilon

        # 路由网络：线性层 + 偏置
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # ── TODO 3.1: 定义可学习的偏置向量 ──────────────────────────
        # 每个专家一个偏置值，shape: [num_experts]
        # 初始化为 0（不偏向任何专家）
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        # ──────────────────────────────────────────────────────────────

        # 用于追踪负载统计（推理时不更新偏置）
        self.training_step = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B*T, d_model]  输入 token

        Returns:
            topk_weights:  [B*T, top_k]  选中的专家权重
            topk_indices:  [B*T, top_k]  选中的专家索引
        """
        # Step 1: 计算 logits 并加偏置（这是核心创新！）
        logits = self.router(x)  # [num_tokens, num_experts]

        # ── TODO 3.2: 添加可学习偏置 ──────────────────────────────────
        # logits 加上 self.expert_bias（广播到 [num_tokens, num_experts]）
        biased_logits = logits + self.expert_bias.unsqueeze(0)
        # ───────────────────────────────────────────────────────────────

        # Step 2: softmax → top-k 选择
        router_probs = F.softmax(biased_logits, dim=-1)  # [num_tokens, num_experts]
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Step 3: 对 top-k 权重重新归一化，使其和为 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return topk_weights, topk_indices, biased_logits

    @torch.no_grad()
    def update_bias(self, topk_indices: torch.Tensor):
        """
        训练完成后动态更新偏置（不参与梯度计算！）。

        这是 DeepSeek-V3 负载均衡的核心：
          每个专家收到 token > 平均 → 降低偏置
          每个专家收到 token < 平均 → 升高偏置

        Args:
            topk_indices: [num_tokens, top_k]  本次 step 每个 token 选中的专家
        """
        num_tokens = topk_indices.shape[0]
        # 1. 统计每个专家被选中的次数
        expert_counts = torch.zeros(self.num_experts, device=topk_indices.device)
        flat_indices = topk_indices.reshape(-1)
        counts = torch.bincount(flat_indices, minlength=self.num_experts).float()

        # 2. 每个 token 选 top_k 个专家，所以总选择次数 = num_tokens * top_k
        # 平均每个专家应被选中的次数 = num_tokens * top_k / num_experts
        # ── TODO 3.3: 完成偏置更新逻辑 ──────────────────────────────
        # 如果 count_i > avg_load → bias_i -= bias_update_rate
        # 如果 count_i < avg_load → bias_i += bias_update_rate
        avg_load = num_tokens * self.top_k / self.num_experts
        # 逐专家判断
        for i in range(self.num_experts):
            if counts[i] > avg_load:
                self.expert_bias.data[i] -= self.bias_update_rate
            elif counts[i] < avg_load:
                self.expert_bias.data[i] += self.bias_update_rate
        # ──────────────────────────────────────────────────────────────
        self.training_step += 1


# ══════════════════════════════════════════════════════════════════════════════
# Part 4: CSA + HCA + V4 MoE（DeepSeek-V4 混合注意力 + 专家架构）
# ══════════════════════════════════════════════════════════════════════════════
#
# 【背景】
#   DeepSeek-V4（2026年4月发布）最大的架构变革：
#     从 V3 的 MLA（单头潜注意力）转向混合注意力架构。
#
#   核心问题：如何让 1M token 上下文的高效推理成为可能？
#     传统注意力：O(n²) 计算 + O(n) KV Cache → 1M 序列直接 OOM
#     MLA 虽压缩了 KV Cache 但仍需 O(n) 计算量
#     V4 方案：双重压缩 + 稀疏检索 + 局部窗口 = O(n) 计算 + O(c) KV Cache
#
# 【三路协同注意力】
#   CSA 和 HCA 不是替代关系，而是互补协同，再配合 SWA：
#
#     ┌─────────────────────────────────────────────────────┐
#     │               Attention Output                      │
#     │         ┌──────┴──────┐                             │
#     │    CSA Path      HCA Path     SWA Path (共128 tok)  │
#     │    (4:1 稀疏)   (128:1 稠密)   (局部精确)            │
#     │    top-1024     全部压缩块    滑动窗口               │
#     └─────────────────────────────────────────────────────┘
#
#   CSA (Compressed Sparse Attention / 压缩稀疏注意力)：
#     4 token → 1 压缩块 → 索引器评分 → top-k 稀疏选择 → 注意力
#     ⚡ 精准定位最相关的上下文片段（像"搜索引擎"）
#
#   HCA (Heavily Compressed Attention / 重度压缩注意力)：
#     128 token → 1 压缩块 → 全量稠密注意力
#     ⚡ 廉价的全景视野，捕捉 CSA 可能遗漏的远端依赖
#
#   SWA (Sliding Window Attention / 滑动窗口注意力)：
#     保留最近 128 token 不做压缩，保证局部精度
#
# 【亮点】V4-Pro 在 1M token 下推理 FLOPs 仅为 V3.2 的 27%，KV Cache 为 10%
# =============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# 4.1 KV 压缩模块（CSA 和 HCA 的共享基础）
# ──────────────────────────────────────────────────────────────────────────────
#
# 压缩是 V4 的根基。将连续 n 个 token 的 KV 向量压缩为一个"语义块"。
# 压缩方式：加权平均，权重由一个轻量级 MLP 学习得到。
#
# CSA 压缩比 = 4:1（保留细节，用于精准检索）
# HCA 压缩比 = 128:1（极度压缩，用于全局概览）
# =============================================================================


class KVCompressor(nn.Module):
    """
    KV 压缩模块：将连续的 N 个 KV 向量压缩为 1 个语义块。

    压缩方式：学习每个位置在块内的加权系数，加权求和。
    """

    def __init__(self, d_model: int, compress_ratio: int, head_dim: Optional[int] = None):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.comp_weights = nn.Parameter(torch.ones(compress_ratio) / compress_ratio)

    @staticmethod
    def compress_per_head(k: torch.Tensor, v: torch.Tensor, comp_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对已分头的 KV 做压缩"""
        B, H, T, D = k.shape
        R = comp_w.shape[0]
        T_comp = T // R
        comp_w_b = comp_w.reshape(1, 1, 1, R, 1)
        k_g = k.reshape(B, H, T_comp, R, D)
        v_g = v.reshape(B, H, T_comp, R, D)
        return (k_g * comp_w_b).sum(dim=3), (v_g * comp_w_b).sum(dim=3)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            k: [B, n_heads, T, head_dim]  key 张量
            v: [B, n_heads, T, head_dim]  value 张量

        Returns:
            k_comp: [B, n_heads, T//R, head_dim]  压缩后的 key
            v_comp: [B, n_heads, T//R, head_dim]  压缩后的 value
        """
        B, H, T, D = k.shape
        R = self.compress_ratio
        T_comp = T // R

        # softmax 归一化权重
        comp_w = F.softmax(self.comp_weights, dim=0)

        # 将序列重塑为 [B, H, T_comp, R, D] 并加权求和
        comp_w_b = comp_w.reshape(1, 1, 1, R, 1)
        k_comp = (k.reshape(B, H, T_comp, R, D) * comp_w_b).sum(dim=3)
        v_comp = (v.reshape(B, H, T_comp, R, D) * comp_w_b).sum(dim=3)

        return k_comp, v_comp


# ──────────────────────────────────────────────────────────────────────────────
# 4.2 Lightning Indexer（闪电索引器）— CSA 的"搜索引擎"
# ──────────────────────────────────────────────────────────────────────────────
#
# CSA 的核心：不是盲目地做注意力，而是先"搜索"出相关的压缩块再做注意力。
# 这个搜索任务由一个轻量级网络 —— Lightning Indexer 完成。
#
# Lightning Indexer 的结构：
#   1. 把 query 和 压缩后的 key 做点积打分
#   2. 对每个 query，留下分数最高的 top-k 个块
#   3. 只有选中的块参与后续注意力计算
#
#   为什么用 ReLU？ReLU 天然稀疏，过滤掉不相关的打分。
#
# 在 V4 中，Indexer 是多头的（n_heads_I < n_heads），
# 多个头综合投票决定哪些块最重要。
#
# StreamIndex 优化（工程实现）：
#   不物化完整的 [B, S, n_I, T] 分数矩阵（那需要 256GB！），
#   而是用 partition-merge top-k 流式处理。
#
# 本教学脚本使用简化版（直接打分 + top-k）。
# =============================================================================


class LightningIndexer(nn.Module):
    """
    闪电索引器：对压缩后的 KV 块做稀疏选择。

    对每个 query，从所有压缩块中选出分数最高的 top-k 个。
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, top_k: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.top_k = top_k

        # Q 和 K 的投影（比主注意力头数少）
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)

    def forward(self,
                query: torch.Tensor,
                k_compressed: torch.Tensor,
                causal: bool = True) -> torch.Tensor:
        """
        Args:
            query:       [B, T, d_model]  主模型的 query（未压缩）
            k_compressed:[B, T_comp, d_model]  压缩后的 key
            causal:      是否因果掩码

        Returns:
            topk_indices: [B, n_heads, T, top_k]  每个 query 选中的压缩块索引
        """
        B, T, _ = query.shape
        T_comp = k_compressed.shape[1]

        # 投影到索引器空间（比主注意力维度小）
        q_idx = self.q_proj(query)      # [B, T, n_heads * d_head]
        k_idx = self.k_proj(k_compressed)  # [B, T_comp, n_heads * d_head]

        # 分头
        q_idx = q_idx.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k_idx = k_idx.reshape(B, T_comp, self.n_heads, self.d_head).transpose(1, 2)

        # ── TODO 4.2: 计算索引器分数 ────────────────────────────────
        # scores = ReLU(q @ k^T / sqrt(d_head))
        # 对 query 和压缩 key 做点积（注意力分数）
        # q_idx: [B, n_heads, T, d_head], k_idx: [B, n_heads, T_comp, d_head]
        # 结果 scores: [B, n_heads, T, T_comp]
        scores = torch.matmul(q_idx, k_idx.transpose(-2, -1))  # [B, n, T, T_comp]
        scores = scores / math.sqrt(self.d_head)
        scores = F.relu(scores)  # 稀疏化（DeepSeek-V4 使用 ReLU 而非 softmax！）
        # ───────────────────────────────────────────────────────────────

        # 因果掩码：query i 只能看到前面的压缩块
        if causal:
            # 压缩后序列长度为 T_comp，每个压缩块覆盖 R 个 token
            # query i 能看到的最后一个完整块是 i // R
            causal_mask = torch.arange(T, device=scores.device)[:, None] >= \
                          torch.arange(T_comp, device=scores.device)[None, :] * 1
            # 简化版：每个 query 能看到所有索引 i < T_comp 的块（保守）
            mask = torch.triu(torch.ones(T, T_comp, dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # top-k 选择
        # ── TODO 4.3: top-k 稀疏选择 ────────────────────────────────
        # 对每个 query 的分数做 top-k，返回索引
        # scores: [B, n_heads, T, T_comp]，选取最大的 top_k 个
        topk_scores, topk_indices = torch.topk(
            scores, min(self.top_k, T_comp), dim=-1
        )
        # ──────────────────────────────────────────────────────────────

        return topk_indices, topk_scores


# ──────────────────────────────────────────────────────────────────────────────
# 4.3 CSA（Compressed Sparse Attention）
# ──────────────────────────────────────────────────────────────────────────────
#
# CSA = KV 压缩 (4:1) + Lightning Indexer top-k 稀疏选择 + 注意力计算
#
# 数据流：
#   Q (未压缩) ──┐
#                ├──→ Lightning Indexer ──→ top-k 索引
#   K, V ──→ 4:1 压缩 ──→ 根据 top-k 索引选出相关块 ──→ 注意力 ──→ 输出
#                                                           ↑
#   SWA (滑动窗口) ──────────────────────────────────────────┘
#
#   最终输出 = CSA 输出 + SWA 输出
# =============================================================================


class CompressedSparseAttention(nn.Module):
    """
    CSA: 4:1 压缩 + Lightning Indexer top-k 稀疏注意力。
    """

    def __init__(self, d_model: int, n_heads: int, n_indexer_heads: int,
                 compress_ratio: int = 4, indexer_top_k: int = 512,
                 swa_window: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.compress_ratio = compress_ratio
        self.swa_window = swa_window

        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        # KV 压缩器（4:1）
        self.compressor = KVCompressor(d_model, compress_ratio)

        # Lightning Indexer
        self.indexer = LightningIndexer(
            d_model, n_indexer_heads, self.d_head, indexer_top_k
        )

        # FP4 量化模拟（为了教学，实际 V4 用真正的 FP4 存储）
        self.fp4_enabled = False

    def enable_fp4(self):
        """模拟 FP4 量化：KV 以 4-bit 存储（实际使用量化感知训练）"""
        self.fp4_enabled = True

    @staticmethod
    def simulate_fp4_quant(tensor: torch.Tensor) -> torch.Tensor:
        """模拟 FP4 量化效果：保留正负符号 + 4 个有效 bit 的信息"""
        scale = tensor.abs().max(dim=-1, keepdim=True).values / 7.0
        tensor_q = (tensor / (scale + 1e-10)).round().clamp(-7, 7)
        return tensor_q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]  输入

        Returns:
            output: [B, T, d_model]  注意力输出
        """
        B, T, D = x.shape
        T_comp = T // self.compress_ratio

        # ── 分支 A: CSA 压缩稀疏注意力 ──────────────────────────────
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # KV 压缩（4:1）
        k_comp, v_comp = self.compressor(k, v)

        # Lightning Indexer 获取 top-k 索引
        topk_indices, _ = self.indexer(x, k_comp.transpose(1, 2).reshape(B, -1, D))
        # topk_indices: [B, n_idx_heads, T, top_k]

        # 根据索引收集压缩块并做注意力
        csa_out = torch.zeros_like(q)  # [B, n_heads, T, D]

        for b in range(B):
            for h in range(self.n_heads):
                idx_h = h % self.indexer.n_heads
                idx = topk_indices[b, idx_h]   # [T, top_k] — values in [0, T_comp)

                # q_bh: [T, D], k_comp_bh: [T_comp, D], v_comp_bh: [T_comp, D]
                q_bh = q[b, h]
                k_bh = k_comp[b, h]
                v_bh = v_comp[b, h]

                # 收集被选中的压缩块
                # idx: [T, top_k] → k_sel: [T, top_k, D]
                k_sel = k_bh[idx]
                v_sel = v_bh[idx]

                # q vs k_sel 注意力: scores [T, top_k]
                scores = (q_bh.unsqueeze(1) * k_sel).sum(-1) / math.sqrt(self.d_head)
                weights = F.softmax(scores, dim=-1)
                csa_out[b, h] = (weights.unsqueeze(-1) * v_sel).sum(dim=1)

        # ── 分支 B: SWA ──────────────────────────────────────────────
        swa_out = torch.zeros_like(q)
        swa_len = min(T, self.swa_window)
        swa_q = q[:, :, -swa_len:]
        swa_k = k[:, :, -swa_len:]
        swa_v = v[:, :, -swa_len:]

        swa_scores = torch.matmul(swa_q, swa_k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # 窗口内因果掩码
        W = swa_scores.shape[-1]
        swa_mask = torch.triu(torch.ones(W, W, dtype=torch.bool, device=x.device), diagonal=1)
        swa_scores = swa_scores.masked_fill(swa_mask[-swa_scores.shape[2]:, :W], float('-inf'))
        swa_weights = F.softmax(swa_scores, dim=-1)
        swa_out[:, :, -swa_len:] = torch.matmul(swa_weights, swa_v)

        # ── 合并 CSA + SWA ──────────────────────────────────────────
        attn_out = (csa_out + swa_out).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_out)


# ──────────────────────────────────────────────────────────────────────────────
# 4.4 HCA（Heavily Compressed Attention）
# ──────────────────────────────────────────────────────────────────────────────
#
# HCA = KV 压缩 (128:1) + 压缩块上的全量稠密注意力
#
# 和 CSA 的关键区别：
#   - 压缩比更高（128:1 vs 4:1），块数极少，可以全量计算
#   - 不需要 Lightning Indexer（太少块了，不需要稀疏化）
#   - 提供"全局概览"能力
#
# 数据流：
#   Q, K, V ──→ 128:1 压缩 ──→ 压缩块上的全量注意力 ──→ 输出
#   SWA ──────────────────────────────────────────→ 合并
# =============================================================================


class HeavilyCompressedAttention(nn.Module):
    """
    HCA: 128:1 重度压缩 + 稠密注意力。
    无需索引器，直接对压缩后的所有块做注意力。
    """

    def __init__(self, d_model: int, n_heads: int,
                 compress_ratio: int = 128, swa_window: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.compress_ratio = compress_ratio
        self.swa_window = swa_window

        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * self.d_head, d_model, bias=False)

        # KV 压缩器（128:1）
        self.compressor = KVCompressor(d_model, compress_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        # QKV 投影
        q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ── 压缩 ─────────────────────────────────────────────────────
        # 对 KV 做 128:1 压缩
        k_comp, v_comp = self.compressor(k, v)         # [B, n_heads, T//128, d_head]

        # ── 压缩块上的全量稠密注意力 ────────────────────────────────
        # ── TODO 4.4: 对 q 做压缩 ─────────────────────────────────────
        R = self.compress_ratio
        T_comp = T // R
        q_grouped = q[:, :, :T_comp * R].reshape(B, self.n_heads, T_comp, R, self.d_head)
        q_pooled = q_grouped.mean(dim=3)  # [B, n_heads, T_comp, d_head]
        # ──────────────────────────────────────────────────────────────

        # 压缩块上的注意力
        hca_scores = torch.matmul(q_pooled, k_comp.transpose(-2, -1)) / math.sqrt(self.d_head)
        hca_weights = F.softmax(hca_scores, dim=-1)    # [B, n_heads, T_comp, T_comp]
        hca_out = torch.matmul(hca_weights, v_comp)    # [B, n_heads, T_comp, d_head]

        # 上采样回原始序列长度（每个压缩块内的所有 token 共享 HCA 输出）
        # ── TODO 4.5: HCA 输出上采样 ────────────────────────────────
        hca_up = hca_out.repeat_interleave(R, dim=2)   # [B, n_heads, T_comp*R, d_head]
        hca_full = torch.zeros(B, self.n_heads, T, self.d_head, device=x.device)
        hca_full[:, :, :hca_up.shape[2]] = hca_up[:, :, :T]
        # ──────────────────────────────────────────────────────────────

        # ── SWA ──────────────────────────────────────────────────────
        swa_out = torch.zeros(B, self.n_heads, T, self.d_head, device=x.device)
        swa_k = k[:, :, -self.swa_window:]
        swa_v = v[:, :, -self.swa_window:]
        swa_q = q[:, :, -min(T, self.swa_window):]

        swa_scores = torch.matmul(swa_q, swa_k.transpose(-2, -1)) / math.sqrt(self.d_head)
        swa_weights = F.softmax(swa_scores, dim=-1)
        swa_out[:, :, -min(T, self.swa_window):] = torch.matmul(swa_weights, swa_v)

        # ── 合并 HCA + SWA ──────────────────────────────────────────
        attn_out = hca_full + swa_out
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head)
        return self.out_proj(attn_out)


# ──────────────────────────────────────────────────────────────────────────────
# 4.5 V4 MoE（DeepSeek-V4 的专家架构改进）
# ──────────────────────────────────────────────────────────────────────────────
#
# DeepSeek-V4 在 MoE 上有几项关键改进：
#
# 1. Sqrt(Softplus) 亲和度分数
#    传统路由器: sigmoid(score) → 归一化后得到路由权重
#    V4 路由器:   softplus(score) → sqrt() → 归一化
#    为什么？Softplus 更平滑，sqrt 放大差异，路由决策更清晰
#
# 2. Hash Routing（哈希路由）
#    前几层不用学习的路由器，而是根据 token id 的哈希值分配专家。
#    好处：不需要训练路由参数，训练更稳定。
#
# 3. Per-Sequence Balance Loss（序列级平衡损失）
#    在 V3 auxiliary-loss-free bias 调节的基础上，
#    对每个序列内部也做负载均衡，防止单序列内部专家坍缩。
#
# 4. FP4 量化专家权重
#    MoE 专家权重用 FP4 精度存储，其余部分用 FP8
#    量化感知训练，不影响模型质量
#
# 5. MegaMoE 波次调度
#    把专家分成多个微批次，通信和计算高度流水线化
#    （工程优化，本教学不实现）
# =============================================================================


class SqrtSoftplusRouter(nn.Module):
    """
    DeepSeek-V4 的路由器：使用 Sqrt(Softplus(.)) 替代 Sigmoid。

    Why Sqrt(Softplus)?
      sigmoid:  输出在 [0, 1]，梯度最大 0.25，容易饱和
      softplus: ln(1 + e^x)，处处光滑，无饱和区
      sqrt(softplus): 进一步放大分数差异，路由更"尖锐"
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B*T, d_model]  token 输入

        Returns:
            weights: [B*T, top_k]  路由权重（已归一化）
            indices: [B*T, top_k]  选中的专家索引
        """
        # ── TODO 4.6: 实现 Sqrt(Softplus) 路由 ──────────────────────
        # 1. 线性层得到 logits
        # 2. softplus 激活（F.softplus）
        # 3. sqrt 缩放
        # 4. top-k 选择
        # 5. 对选中的 k 个权重归一化
        logits = self.router(x)                                  # [N, E]
        affinity = torch.sqrt(F.softplus(logits) + 1e-10)        # [N, E]
        # ──────────────────────────────────────────────────────────────

        weights, indices = torch.topk(affinity, self.top_k, dim=-1)
        # 归一化
        weights = weights / weights.sum(dim=-1, keepdim=True)

        return weights, indices


class HashRouter(nn.Module):
    """
    哈希路由器：不学习路由参数，根据 token 的 hash 值分配专家。

    用于前几层（通常是前 3 层），V4 发现这比学习路由更稳定。
    """

    def __init__(self, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用 token 的 L2 norm 的量化值做 hash，决定分配到哪个专家。
        注意：这个"hash"不能太随机——同一个 token 必须总是映射到同一批专家。
        """
        num_tokens = x.shape[0]

        # 用 norm 值的二进制表示生成 hash（确定性）
        token_norms = torch.norm(x, dim=-1)  # [num_tokens]
        # 量化 norm 到 [0, num_experts) 空间
        norm_scaled = (token_norms - token_norms.min()) / \
                      (token_norms.max() - token_norms.min() + 1e-10)
        hash_base = (norm_scaled * self.num_experts).long()

        # 为每个 token 生成 top_k 个不同的专家
        indices = torch.stack([
            (hash_base + i) % self.num_experts for i in range(self.top_k)
        ], dim=-1)

        # 均匀权重
        weights = torch.ones(num_tokens, self.top_k, device=x.device) / self.top_k
        return weights, indices


class V4MoELayer(nn.Module):
    """
    DeepSeek-V4 的 MoE 层。

    结合了：
    - Shared Experts（共享专家，所有 token 都经过）
    - Routed Experts（路由专家，token 按需选择）
    - Sqrt(Softplus) 亲和度
    - 可选 Hash Routing
    - Per-Sequence Balance Loss
    """

    def __init__(self, d_model: int, d_ff: int,
                 num_shared_experts: int = 2,
                 num_routed_experts: int = 64,
                 top_k: int = 6,
                 use_hash_router: bool = False,
                 seq_balance_coeff: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        self.num_routed_experts = num_routed_experts
        self.seq_balance_coeff = seq_balance_coeff

        # Shared experts（所有 token 都计算）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            ) for _ in range(num_shared_experts)
        ])

        # Routed experts
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            ) for _ in range(num_routed_experts)
        ])

        # 路由器
        if use_hash_router:
            self.router = HashRouter(num_routed_experts, top_k)
        else:
            self.router = SqrtSoftplusRouter(d_model, num_routed_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model]

        Returns:
            output:   [B, T, d_model]
            aux_loss: 标量辅助损失（序列级平衡）
        """
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)  # [B*T, D]
        num_tokens = flat_x.shape[0]

        # ── Step 1: Shared Experts（所有 token 都计算） ──────────────
        shared_out = sum(expert(flat_x) for expert in self.shared_experts)
        shared_out = shared_out / len(self.shared_experts)  # 平均

        # ── Step 2: Routing ─────────────────────────────────────────
        routing_weights, routing_indices = self.router(flat_x)
        # routing_weights: [B*T, top_k], routing_indices: [B*T, top_k]

        # ── Step 3: Expert Dispatch & Compute ───────────────────────
        # 简化版：对每个 token，用 router 选中的 expert 计算
        routed_out = torch.zeros_like(flat_x)  # [B*T, D]

        for i in range(num_tokens):
            for k in range(self.top_k):
                expert_idx = routing_indices[i, k]
                weight = routing_weights[i, k]
                expert_out = self.routed_experts[expert_idx](flat_x[i:i+1])
                routed_out[i:i+1] += weight * expert_out

        # ── Step 4: 序列级辅助平衡损失 ──────────────────────────────
        # 确保每个序列内部各专家负载也均衡
        # ── TODO 4.7: 计算序列级平衡损失 ────────────────────────────
        # 对每个序列（batch 中的每一行），统计 expert 分布
        # 每个序列应尽量均匀地使用所有专家
        # 将 routing_indices 也 reshape 到 [B, T, top_k]
        seq_indices = routing_indices.reshape(B, T, self.top_k)
        seq_balance_loss = torch.tensor(0.0, device=x.device)

        for b in range(B):
            # 统计第 b 个序列中各专家被选中的频次
            seq_counts = torch.zeros(self.num_routed_experts, device=x.device)
            for t in range(T):
                for k in range(self.top_k):
                    idx = seq_indices[b, t, k]
                    seq_counts[idx] += 1
            # 计算该序列的均匀度损失（方差越小越均匀）
            expected = T * self.top_k / self.num_routed_experts
            seq_var = ((seq_counts - expected) ** 2).mean()
            seq_balance_loss = seq_balance_loss + seq_var

        seq_balance_loss = seq_balance_loss / B
        aux_loss = self.seq_balance_coeff * seq_balance_loss
        # ──────────────────────────────────────────────────────────────

        # ── 合并 Shared + Routed ─────────────────────────────────────
        output = shared_out + routed_out
        return output.reshape(B, T, D), aux_loss


# ══════════════════════════════════════════════════════════════════════════════
# 测试验证
# ══════════════════════════════════════════════════════════════════════════════

def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_mtp():
    """测试 Multi-Token Prediction 的前向传播和损失计算"""
    print_separator("Part 1: Multi-Token Prediction 测试")

    B, T, D, V = 2, 8, 3, 100
    d_model, n_heads, d_ff = 64, 4, 128

    # 构造输入
    input_ids = torch.randint(0, V, (B, T + D))
    main_hidden = torch.randn(B, T, d_model)

    # 共享 embedding
    embed = nn.Embedding(V, d_model)

    mtp = MultiTokenPrediction(d_model, n_heads, d_ff, V, num_mtp_modules=D)

    # 测试前向传播
    all_logits = mtp(main_hidden, input_ids, embed)

    assert len(all_logits) == D, f"MTP 应输出 {D} 个 logits，得到 {len(all_logits)}"
    for k, logits in enumerate(all_logits):
        assert logits.shape == (B, T, V), \
            f"MTP-{k} logits shape 应为 ({B}, {T}, {V})，得到 {logits.shape}"

    print(f"  ✓ MTP 前向传播正确：{D} 个模块，每个输出 [{B}, {T}, {V}]")

    # 测试损失计算
    loss = mtp.compute_mtp_loss(all_logits, input_ids)
    assert loss.ndim == 0, f"MTP loss 应为标量，得到 shape {loss.shape}"
    assert loss > 0, f"MTP loss 应为正数，得到 {loss}"

    print(f"  ✓ MTP 损失计算正确：loss = {loss:.4f}")
    print(f"  ✓ Part 1 全部通过！")


def test_grpo():
    """测试 GRPO 损失计算"""
    print_separator("Part 2: GRPO 测试")

    B, G, T = 4, 8, 16  # 4 个 prompt，每组 8 个回答，每个 16 token

    # 模拟 log_probs 和 rewards
    log_probs = torch.randn(B, G, T) * 0.5 - 1.0
    ref_log_probs = torch.randn(B, G, T) * 0.5 - 1.0
    rewards = torch.randn(B, G) * 2.0 + 1.0  # 随机 reward

    loss = grpo_loss(log_probs, ref_log_probs, rewards)

    assert loss.ndim == 0, f"GRPO loss 应为标量，得到 shape {loss.shape}"
    assert loss > 0, f"GRPO loss 应为正数，得到 {loss:.4f}"

    print(f"  ✓ GRPO 损失计算正确：loss = {loss:.4f}")

    # 测试当所有回答 reward 相等时的行为（优势应为 0）
    uniform_rewards = torch.ones(B, G) * 2.0
    loss_uniform = grpo_loss(log_probs, ref_log_probs, uniform_rewards)
    print(f"  ✓ 均匀 reward 时 loss = {loss_uniform:.4f}（只有 KL 惩罚贡献）")

    # 验证 KL 惩罚方向：与 ref 完全一致的 policy 应比偏离的 policy loss 更小
    identical_log_probs = ref_log_probs.clone()  # policy = reference, KL=0
    loss_identical = grpo_loss(identical_log_probs, ref_log_probs, rewards)
    assert loss_identical < loss, \
        f"与 ref 一致的 policy 应有更小的 loss: {loss_identical} >= {loss}"

    print(f"  ✓ KL 惩罚方向正确：policy 越接近 reference，loss 越小")
    print(f"      loss(ref)  = {loss_identical:.4f}  <  loss(random) = {loss:.4f}")
    print(f"  ✓ Part 2 全部通过！")


def test_auxiliary_free_load_balance():
    """测试无辅助损失负载均衡"""
    print_separator("Part 3: Auxiliary-Free Load Balance 测试")

    d_model = 32
    num_experts = 4
    top_k = 2
    num_tokens = 64

    router = AuxiliaryFreeRouter(d_model, num_experts, top_k)

    # 测试前向传播
    x = torch.randn(num_tokens, d_model)
    topk_weights, topk_indices, biased_logits = router(x)

    assert topk_weights.shape == (num_tokens, top_k), \
        f"topk_weights shape 应为 ({num_tokens}, {top_k})，得到 {topk_weights.shape}"
    assert topk_indices.shape == (num_tokens, top_k), \
        f"topk_indices shape 应为 ({num_tokens}, {top_k})，得到 {topk_indices.shape}"

    print(f"  ✓ 路由器前向传播正确")

    # 验证每个 token 的权重和为 1
    assert torch.allclose(topk_weights.sum(dim=-1),
                          torch.ones(num_tokens), atol=1e-5), \
        "每个 token 的 top-k 权重和应为 1"

    print(f"  ✓ 路由权重归一化正确，每个 token 权重和 = 1")

    # 测试偏置更新
    initial_bias = router.expert_bias.data.clone()

    # 模拟多个步骤的负载更新
    for step in range(100):
        x = torch.randn(num_tokens, d_model)
        _, indices, _ = router(x)
        router.update_bias(indices)

    final_bias = router.expert_bias.data

    # 偏置应该发生了变化
    assert not torch.allclose(initial_bias, final_bias, atol=1e-6), \
        "偏置在 100 步更新后应发生变化"

    bias_change = (final_bias - initial_bias).abs().sum().item()
    print(f"  ✓ 偏置动态更新正确：累计偏置变化量 = {bias_change:.4f}")

    # 验证偏置不参与梯度（检查 requires_grad）
    assert router.expert_bias.requires_grad, "偏置参数应需要梯度（虽然 update 用 no_grad）"

    print(f"  ✓ 偏置作为 Parameter（可被优化器更新，但 update_bias 用 no_grad 控制）")
    print(f"  ✓ Part 3 全部通过！")


def test_csa_hca():
    """测试 CSA + HCA + V4 MoE"""
    print_separator("Part 4: CSA + HCA + V4 MoE 测试")

    d_model, n_heads, d_ff = 64, 4, 128

    # ── 4.1 KV Compressor ────────────────────────────────────────────
    print("\n  4.1 KVCompressor...")
    compressor = KVCompressor(d_model, compress_ratio=4)
    k = torch.randn(2, n_heads, 16, d_model // n_heads)
    v = torch.randn_like(k)
    k_comp, v_comp = compressor(k, v)
    assert k_comp.shape == (2, n_heads, 4, d_model // n_heads), \
        f"压缩 4:1 后 shape 应为 (2, 4, 4, 16)，得到 {k_comp.shape}"
    print(f"  ✓ KV 4:1 压缩正确：{k.shape} → {k_comp.shape}")

    # ── 4.2 Lightning Indexer ────────────────────────────────────────
    print("\n  4.2 LightningIndexer...")
    indexer = LightningIndexer(d_model, n_heads=2, d_head=16, top_k=2)
    q = torch.randn(2, 16, d_model)
    k_comp_reshape = k_comp.transpose(1, 2).reshape(2, 4, d_model)
    indices, scores = indexer(q, k_comp_reshape)
    assert indices.shape[-1] == 2, f"top_k=2 应返回每个 query 2 个索引"
    print(f"  ✓ Lightning Indexer top-k 选择正确：每个 query 选 {indices.shape[-1]} 个块")

    # ── 4.3 CSA ──────────────────────────────────────────────────────
    print("\n  4.3 CompressedSparseAttention (简化测试)...")
    csa = CompressedSparseAttention(d_model, n_heads, n_indexer_heads=2, compress_ratio=4, indexer_top_k=2, swa_window=4)
    x = torch.randn(2, 16, d_model)
    csa_out = csa(x)
    assert csa_out.shape == x.shape, f"CSA 输出 shape 应为 {x.shape}，得到 {csa_out.shape}"
    print(f"  ✓ CSA 前向传播正确：输出 {csa_out.shape}")

    # ── 4.4 HCA ──────────────────────────────────────────────────────
    print("\n  4.4 HeavilyCompressedAttention (简化测试)...")
    hca = HeavilyCompressedAttention(d_model, n_heads, compress_ratio=4, swa_window=4)
    x = torch.randn(2, 16, d_model)
    hca_out = hca(x)
    assert hca_out.shape == x.shape, f"HCA 输出 shape 应为 {x.shape}，得到 {hca_out.shape}"
    print(f"  ✓ HCA 前向传播正确：输出 {hca_out.shape}")

    # ── 4.5 SqrtSoftplusRouter ───────────────────────────────────────
    print("\n  4.5 SqrtSoftplusRouter...")
    router = SqrtSoftplusRouter(d_model, num_experts=8, top_k=2)
    x_flat = torch.randn(32, d_model)
    weights, indices = router(x_flat)
    assert weights.shape == (32, 2), f"weights shape 应为 (32, 2)，得到 {weights.shape}"
    assert torch.allclose(weights.sum(dim=-1), torch.ones(32)), "权重和应为 1"
    print(f"  ✓ SqrtSoftplus 路由正确：权重和 = 1，top-2 选择")
    # 验证 sqrt(softplus) 输出为正
    logits = router.router(x_flat)
    assert (F.softplus(logits) >= 0).all(), "Softplus 输出应 ≥ 0"
    print(f"  ✓ Sqrt(Softplus) 始终为非负，梯度光滑")

    # ── 4.6 V4 MoE Layer ─────────────────────────────────────────────
    print("\n  4.6 V4MoELayer...")
    moe = V4MoELayer(d_model, d_ff, num_shared_experts=1, num_routed_experts=4, top_k=2)
    x = torch.randn(2, 8, d_model)
    moe_out, aux_loss = moe(x)
    assert moe_out.shape == x.shape, f"MoE 输出 shape 应为 {x.shape}，得到 {moe_out.shape}"
    assert aux_loss.ndim == 0, f"aux_loss 应为标量，得到 {aux_loss.shape}"
    print(f"  ✓ V4 MoE 前向传播正确：输出 {moe_out.shape}，aux_loss={aux_loss:.6f}")

    # ── 4.7 Hash Router ──────────────────────────────────────────────
    print("\n  4.7 HashRouter...")
    hash_router = HashRouter(num_experts=8, top_k=2)
    h_weights, h_indices = hash_router(x_flat)
    assert h_weights.shape == (32, 2), f"Hash weights shape 应为 (32, 2)"
    # 相同输入应得到相同路由结果（确定性）
    h_weights2, h_indices2 = hash_router(x_flat)
    assert (h_indices == h_indices2).all(), "Hash 路由必须是确定性的"
    print(f"  ✓ HashRouter 确定性分配专家（无需训练！）")

    print(f"\n  ✓ Part 4 全部通过！")


if __name__ == "__main__":
    print("=" * 70)
    print("  DeepSeek-V4 核心机制教学脚本")
    print("  包含：MTP + GRPO + Auxiliary-Free Load Balance + CSA/HCA + V4 MoE")
    print("=" * 70)

    tests = [
        ("Part 1", test_mtp),
        ("Part 2", test_grpo),
        ("Part 3", test_auxiliary_free_load_balance),
        ("Part 4", test_csa_hca),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n  ✗ {name} 失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  全部测试完成！")
    print("=" * 70)
