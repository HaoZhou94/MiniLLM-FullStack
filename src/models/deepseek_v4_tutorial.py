"""
DeepSeek-V4 核心算法原理教程
=============================

V3 → V4 最大架构变革：MLA 被 CSA+HCA 混合注意力替换。

核心概念速览：
--------------------------------------------------------------------
  CSA (压缩稀疏注意力)
  +-- KV 压缩：每 m=4 个 token 压缩成 1 个 entry (约3% 存储)
  +-- Lightning Indexer：轻量打分器，选 top-k 相关 entry
  +-- 滑动窗口：最近 128 token 的未Compressed KV (保留局部精度)
  +-- 稀疏注意力：只在 (top-k + 窗口) 的 KV 上计算

  HCA (重度压缩注意力)
  +-- 超高Compression ratio 128:1 -> 1M token -> ~7800 entry
  +-- Dense attention (不做稀疏选择，保留全局视野)
  +-- 充当"章节摘要"通道，弥补 CSA 可能遗漏的全局信息

  混合排布：前 2 层 HCA -> 后续 CSA/HCA 交替 -> 每层都加滑动窗口
--------------------------------------------------------------------

教程结构：
  第1部分: KV Compression — 压缩原理与实现
  第2部分: Lightning Indexer — 稀疏选择器
  第3部分: CSA (Compressed Sparse Attention) — 完整实现
  第4部分: HCA (Heavily Compressed Attention) — 完整实现
  第5部分: Hybrid Block — 混合注意力 Transformer Block
  第6部分: 测试与效果对比
"""

import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Fix Windows console encoding so Chinese prints without PYTHONIOENCODING
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# ==============================================================================
# 第1部分: KV Compression — 压缩原理与实现
# ==============================================================================
#
# 为什么需要Compressed KV？
#   标准 attention 的 KV Cache = O(L * d)，长序列时显存爆炸
#   MLA (V3): 低秩分解Compressed KV → c_kv 维
#   CSA (V4): 时序压缩 + 稀疏选择 → T/m 个 entry + top-k
#
# 时序压缩 vs 低秩压缩：
#   MLA 低秩压缩：d_model → c_kv（维度压缩，每 token 独立）
#   CSA 时序压缩：T → T/m（时间维度压缩，相邻 token 融合）
#
# CSA 压缩策略：
#   每 m 个相邻 token 的 KV 用学习到的权重加权平均，合并成 1 个 entry
#   entry_k = Σ(w_i * k_i) for i in block
#   entry_v = Σ(w_i * v_i) for i in block
#


class KVCompressor(nn.Module):
    """
    将每 block_size 个相邻 token 的 K/V 压缩成 1 个 entry。

    两种压缩Mode：
    1. learned: 可学习的加权平均（V4 使用的方式，更灵活）
    2. mean: 简单平均池化（基线对比用）

    形状变换：
      输入：K, V 各 [B, num_heads, T, head_dim]
      输出：compressed_K, compressed_V 各 [B, num_heads, T/block_size, head_dim]
    """

    def __init__(
        self,
        head_dim: int,
        block_size: int = 4,
        mode: str = "learned",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size
        self.mode = mode

        if mode == "learned":
            # 每个 block 位置学习一组权重，不同头共享
            # 权重经过 softmax 归一化，保证加权和为 1
            self.compress_weight = nn.Parameter(
                torch.randn(block_size) * 0.02
            )


    def forward(
        self,
        k: torch.Tensor,  # [B, num_heads, T, head_dim]
        v: torch.Tensor,  # [B, num_heads, T, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, nh, T, hd = k.shape
        block_size = self.block_size

        # 截断到 block_size 的整数倍（末尾不足一个 block 的 token 丢弃）
        valid_len = (T // block_size) * block_size

        k = k[:, :, :valid_len, :]
        v = v[:, :, :valid_len, :]

        # reshape 成 blocks: [B, nh, T//block_size, block_size, head_dim]
        k_blocks = k.view(B, nh, valid_len // block_size, block_size, hd)  # [2, 4, 4, 4, 64]
        v_blocks = v.view(B, nh, valid_len // block_size, block_size, hd)  # [2, 4, 4, 4, 64]

        if self.mode == "learned":
            # 学习到的加权平均
            w = F.softmax(self.compress_weight, dim=0)  # [block_size]
            w = w.view(1, 1, 1, block_size, 1)  # 广播维度

            """
            图书馆 A (batch[0]):              图书馆 B (batch[1]):                                                                                                                           
            ┌─────────────────────────┐      ┌─────────────────────────┐
            │ 4层书架 (4 heads)         │      │ 4层书架 (4 heads)       │                                                                                                                   
            │ 每层4格 (4 blocks)        │      │ 每层4格 (4 blocks)      │                                                                                                                   
            │ 每格4册 (4 tokens)        │      │ 每格4册 (4 tokens)      │                                                                                                                   
            │ 每册64页 (64 head_dim)    │      │ 每册64页 (64 head_dim)  │                                                                                                                   
            └─────────────────────────┘      └─────────────────────────┘ 
            sum(dim=3) 是"按页号对齐相加"，不是"拼接"：
             k_blocks: [2, 4, 4, 4, 64]
                         │  │  │  │  └── 每册 64 页
                         │  │  │  └── 每格 4 册薄册子
                         │  │  └── 每层 4 格
                         │  └── 每馆 4 层书架
                         └── 2 个图书馆
            
              两个图书馆结构完全相同，操作各自独立进行——A 馆的 token 不会和 B 馆的混合。sum(dim=3) 后，每个图书馆的每层书架每格都从"4 本薄册"变成"1 本合集"，所以 shape 变成 [2, 4, 4,
              64]，batch 维始终不受影响。

                以一个图书馆、一层书架为例，其余广播逻辑一致。                                                                                                                                   
                                                                                                                                                                                                 
                  ---                                                                                                                                                                              
                  压缩前：一层书架有 4 格（4 个 block），每格 4 本薄册子（4 token），每本 64 页
                                                                                                                                                                                                   
                  只取一层书架的 k_blocks：                                                                                                                                                        
                                                                                                                                                                                                   
                  k_blocks[0, 0] 形状: [4, 4, 64]                                                                                                                                                  
                                        │  │  └── 每本 64 页                                                                                                                                       
                                        │  └── 每格 4 本薄册
                                        └── 这层有 4 格
                
                  w = [0.10, 0.20, 0.30, 0.40]      ← 4 个权重，和=1
                  w.view(1, 4, 1) = [[[0.10],       ← 广播形状
                                      [0.20],
                                      [0.30],
                                      [0.40]]]
                
                  ---
                  Step 1: 相乘 k_blocks * w — 给每本册子标"重要性"
                
                  w 的 3 个维度中 2 个是 1（格=1，页=1），只有"册子"维是 4，所以广播规则是：
                  - 册子维：匹配，按位置一一对应（第 i 本 × w_i）
                  - 格维：1 → 4，4 格重复使用同一套权重
                  - 页维：1 → 64，64 页重复使用同一个权重
                
                  格子 1 的 4 本册子：
                
                  相乘前:                          相乘后:
                    册0: [v0, v1, ..., v63]  ×0.10  册0: [v0×0.10, v1×0.10, ..., v63×0.10]
                    册1: [u0, u1, ..., u63]  ×0.20  册1: [u0×0.20, u1×0.20, ..., u63×0.20]
                    册2: [w0, w1, ..., w63]  ×0.30  册2: [w0×0.30, w1×0.30, ..., w63×0.30]
                    册3: [x0, x1, ..., x63]  ×0.40  册3: [x0×0.40, x1×0.40, ..., x63×0.40]
                
                  格子 2 的 4 本册子：
                
                    册0: [a0, a1, ..., a63]  ×0.10  册0: [a0×0.10, a1×0.10, ..., a63×0.10]
                    册1: [b0, b1, ..., b63]  ×0.20  册1: [b0×0.20, b1×0.20, ..., b63×0.20]
                    册2: [c0, c1, ..., c63]  ×0.30  册2: [c0×0.30, c1×0.30, ..., c63×0.30]
                    册3: [d0, d1, ..., d63]  ×0.40  册3: [d0×0.40, d1×0.40, ..., d63×0.40]
                
                  关键： 格子 1 和格子 2 用的是同一套权重 [0.10, 0.20, 0.30, 0.40]—这就是广播的作用：4 个格各自独立操作，但权重模式相同。

                 ---
                  Step 2: sum(dim=3) — 每格 4 本薄册合成 1 本合集
                
                  格子 1：
                
                  第 0 页: v0×0.10 + u0×0.20 + w0×0.30 + x0×0.40
                  第 1 页: v1×0.10 + u1×0.20 + w1×0.30 + x1×0.40
                  ...
                  第63 页: v63×0.10 + u63×0.20 + w63×0.30 + x63×0.40
                
                  格子 2 同理，4 本合成 1 本。最终这层书架 [4格, 4册, 64页] → [4格, 1合集, 64页]。
                
                  ---
                  一句话总结： 相乘是给每本册标"重要性分数"，sum 是按页号把 4 本册的内容加权融成 1 本。64 页不变是因为只做融合不做拼接。


            """
            k_compressed = (k_blocks * w).sum(dim=3)  # [B, nh, T//bs, hd]
            v_compressed = (v_blocks * w).sum(dim=3)
        else:
            # 简单平均池化
            k_compressed = k_blocks.mean(dim=3)
            v_compressed = v_blocks.mean(dim=3)
        return k_compressed, v_compressed


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_kv_compressor():
    print("=" * 60)
    print("Test: KVCompressor")
    print("=" * 60)

    B, nh, T, hd = 2, 4, 16, 64
    block_size = 4

    k = torch.randn(B, nh, T, hd)
    v = torch.randn(B, nh, T, hd)

    # 学习Mode
    comp = KVCompressor(hd, block_size, mode="learned")


    ck, cv = comp(k, v)
    expected_len = T // block_size  # 4
    assert ck.shape == (B, nh, expected_len, hd), f"Shape error: {ck.shape}"
    assert cv.shape == (B, nh, expected_len, hd), f"Shape error: {cv.shape}"
    print(f"  Input K: {k.shape} → Compressed K: {ck.shape}")
    print(f"  Compression ratio: {T}:{expected_len} = {T/expected_len:.0f}:1")

    # 均值Mode
    comp2 = KVCompressor(hd, block_size, mode="mean")
    ck2, cv2 = comp2(k, v)
    assert ck2.shape == ck.shape
    print(f"  learned Mode ✓ | mean Mode ✓")


# ==============================================================================
# 第2部分: Lightning Indexer — 轻量级稀疏选择器
# ==============================================================================
#
# 核心问题：压缩后还有 T/m 个 entry，1M 上下文下 = 250K entry，
#          全量 attention 仍然 O((T/m)²) 太重。
#
# Lightning Indexer 的解决方案：
#   对每个 query，用一个极轻量级的打分网络，快速选出 top-k 个最相关的 entry，
#   只在这些 entry 上计算真实 attention。
#
# 计算流程：
#   1. 用轻量投影把 Q 和Compressed K 映射到低维空间 d_index（如 64 维）
#   2. 计算 score = Q_index @ K_index^T → [B, nh, Tq, T/m]
#   3. top-k 选出最相关的 k 个 entry 索引
#   4. 只在这些 entry 上做真实 attention
#
# 为什么叫 "Lightning"？
#   - Indexer 的维度 d_index 远小于 head_dim（如 64 vs 128）
#   - V4 中 Indexer 的 QK 路径跑在 FP4 精度上（我们模拟为低精度）
#   - 计算量 = O(Tq * T/m * d_index)，远小于 O(Tq * T/m * head_dim)
#
# 关键实现细节：
#   topk_indices 返回的是压缩 entry 的索引（0 ~ T/m-1），
#   后续真实 attention 时，需要从Compressed KV 中 gather 对应 entry。


class LightningIndexer(nn.Module):
    """
    轻量级索引器：为每个 query 选出 top_k_index 个最相关的Compressed KV entry。

    V4 中 Indexer 的 QK 路径全程跑在 FP4（Float4）精度上，
    这里用低维投影模拟轻量级设计的思路。
    """

    def __init__(
        self,
        head_dim: int = 128,
        index_dim: int = 64,  # 索引空间的维度（远小于 head_dim）
        top_k_index: int = 1024,  # 选出的压缩 entry 数
    ):
        super().__init__()
        self.head_dim = head_dim
        self.index_dim = index_dim
        self.top_k_index = top_k_index

        # 轻量投影：Q/K 都映射到低维索引空间
        self.q_index_proj = nn.Linear(head_dim, index_dim, bias=False)
        self.k_index_proj = nn.Linear(head_dim, index_dim, bias=False)

        # 索引空间的缩放因子
        self.scale = 1.0 / math.sqrt(index_dim)

    def forward(
        self,
        q: torch.Tensor,              # [B, nh, Tq, head_dim]
        k_compressed: torch.Tensor,   # [B, nh, Tck, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            topk_indices:  [B, nh, Tq, top_k_index] — 压缩 entry 的索引
            topk_scores:   [B, nh, Tq, top_k_index] — 对应的索引得分（用于后续权重调整）
        """
        B, nh, Tq, _ = q.shape
        Tck = k_compressed.shape[2]

        # Step 1: 投影到低维索引空间
        q_index = self.q_index_proj(q)  # [B, nh, Tq, index_dim]
        k_index = self.k_index_proj(k_compressed)  # [B, nh, Tck, index_dim]

        # Step 2: 计算轻量级相关性得分
        # [B, nh, Tq, index_dim] @ [B, nh, index_dim, Tck] → [B, nh, Tq, Tck]
        index_scores = torch.matmul(q_index, k_index.transpose(-1, -2))
        index_scores = index_scores * self.scale

        # Step 3: top-k 选择
        # 取 top_k_index 个最相关的 entry
        actual_k = min(self.top_k_index, Tck)
        topk_scores, topk_indices = torch.topk(index_scores, actual_k, dim=-1)

        return topk_indices, topk_scores


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_lightning_indexer():
    print("\n" + "=" * 60)
    print("Test: LightningIndexer")
    print("=" * 60)

    B, nh, Tq, Tck, hd = 2, 4, 8, 32, 128
    index_dim, top_k = 64, 16

    q = torch.randn(B, nh, Tq, hd)
    k_compressed = torch.randn(B, nh, Tck, hd)

    indexer = LightningIndexer(hd, index_dim, top_k)
    indices, scores = indexer(q, k_compressed)

    assert indices.shape == (B, nh, Tq, min(top_k, Tck)), f"形状: {indices.shape}"
    print(f"  Q: {q.shape}, 压缩K: {k_compressed.shape}")
    print(f"  Selected top-k indices: {indices.shape}")
    print(f"  Index score range: [{scores.min().item():.3f}, {scores.max().item():.3f}]")

    # 验证索引在有效范围内
    assert indices.max() < Tck, "Index out of bounds!!"
    print(f"  Index range [0, {indices.max().item()}] ✓ (Total entries: {Tck})")


# ==============================================================================
# 第3部分: CSA (Compressed Sparse Attention)
# ==============================================================================
#
# CSA = 压缩(Compress) + 稀疏选择(Sparse) + 滑动窗口(Sliding Window)
#
# 完整数据流：
#   x [B, T, d_model]
#   → Q/K/V 投影
#   → K/V 压缩: T → T/4（KVCompressor）
#   → Lightning Indexer: Q vs 压缩K → top-k 压缩 entry 索引
#   → 从Compressed K/V 中 gather 选中的 entry
#   → 拼接滑动窗口（最近 window_size 个 token 的原始 KV）
#   → 在 [selected compressed KV + sliding window KV] 上计算 attention
#   → 加权聚合 → 输出投影
#
# 复杂度分析（T=1M, m=4, k=1024, w=128）:
#   Compressed entries 数: 250K
#   稀疏 attention 数: 1024 + 128 = 1152 / query
#   vs 全量 attention: 1M / query
#   Sparsity: 1152 / 1M ≈ 0.1%


class CompressedSparseAttention(nn.Module):
    """
    CSA 层：压缩稀疏注意力，V4 混合注意力的核心组件。
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        compression_block: int = 4,      # m: 每 m 个 token 压缩
        index_dim: int = 64,             # Indexer 低维空间
        top_k_index: int = 1024,         # 选出的压缩 entry 数
        window_size: int = 128,          # 滑动窗口大小（未压缩 token）
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.compression_block = compression_block
        self.window_size = window_size
        self.top_k_index = top_k_index
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # ── Q/K/V 投影 ──────────────────────────────────────────────────
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # ── KV 压缩器 ────────────────────────────────────────────────────
        self.kv_compressor = KVCompressor(
            self.head_dim, block_size=compression_block, mode="learned"
        )

        # ── Lightning Indexer ─────────────────────────────────────────────
        self.indexer = LightningIndexer(
            head_dim=self.head_dim,
            index_dim=index_dim,
            top_k_index=top_k_index,
        )

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, T, d_model]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d_model = hidden_states.shape
        device = hidden_states.device
        nh, hd = self.num_heads, self.head_dim

        # ── Step 1: Q/K/V 投影 ─────────────────────────────────────────
        q = self._split_heads(self.q_proj(hidden_states), B)     # [B, nh, T, hd]
        k = self._split_heads(self.k_proj(hidden_states), B)     # [B, nh, T, hd]
        v = self._split_heads(self.v_proj(hidden_states), B)     # [B, nh, T, hd]

        # ── Step 2: KV 压缩 ────────────────────────────────────────────
        # Compression ratio: T → T/m
        k_compressed, v_compressed = self.kv_compressor(k, v)
        # [B, nh, T_c, hd], T_c = T // m
        T_c = k_compressed.shape[2]

        # ── Step 3: Lightning Indexer 稀疏选择 ─────────────────────────
        # 对每个 query，选 top-k 个最相关的压缩 entry
        topk_indices, topk_scores = self.indexer(q, k_compressed)
        # topk_indices: [B, nh, T, top_k]
        actual_k = topk_indices.shape[-1]

        # ── Step 4: Gather selected compressed KV ──────────────────────
        # k_compressed: [B, nh, T_c, hd], topk_indices: [B, nh, T, actual_k]
        k_flat = k_compressed.reshape(B * nh, T_c, hd)       # [B*nh, T_c, hd]
        v_flat = v_compressed.reshape(B * nh, T_c, hd)
        idx_flat = topk_indices.reshape(B * nh, T, actual_k)  # [B*nh, T, actual_k]

        # advanced indexing: k_flat[batch_idx, idx_flat] -> [B*nh, T, actual_k, hd]
        batch_idx = torch.arange(B * nh, device=device).view(B * nh, 1, 1).expand(-1, T, actual_k)
        k_selected = k_flat[batch_idx, idx_flat]
        v_selected = v_flat[batch_idx, idx_flat]

        k_selected = k_selected.view(B, nh, T, actual_k, hd)
        v_selected = v_selected.view(B, nh, T, actual_k, hd)

        # ── Step 5: 滑动窗口 KV ────────────────────────────────────────
        # 保留最近 window_size 个 token 的原始 KV（未压缩）
        # 用于捕捉局部细节
        win_size = min(self.window_size, T)
        if T > win_size:
            k_window = k[:, :, -win_size:, :]  # [B, nh, win_size, hd]
            v_window = v[:, :, -win_size:, :]
        else:
            k_window = k
            v_window = v

        # ── Step 6: 拼接稀疏 KV + 窗口 KV 作为最终 Key/Value ──────────
        # 每个 query 看到的 KV = [selected compressed entries, sliding window]
        # k_selected: [B, nh, T, actual_k, hd]
        # k_window:   [B, nh, win_size, hd]
        k_window_exp = k_window.unsqueeze(2).expand(B, nh, T, win_size, hd)
        v_window_exp = v_window.unsqueeze(2).expand(B, nh, T, win_size, hd)

        k_final = torch.cat([k_selected, k_window_exp], dim=3)  # [B, nh, T, K_total, hd]
        v_final = torch.cat([v_selected, v_window_exp], dim=3)
        K_total = k_final.shape[3]  # actual_k + win_size

        # ── Step 7: 缩放点积注意力（稀疏版本）─────────────────────────
        # q: [B, nh, T, hd], k_final: [B, nh, T, K_total, hd]
        # scores: [B, nh, T, K_total]
        attn_scores = torch.matmul(
            q.unsqueeze(3), k_final.transpose(-1, -2)
        ).squeeze(3)  # [B, nh, T, K_total]
        attn_scores = attn_scores * self.scale

        # 因果掩码只对滑动窗口部分有效
        # 简化处理：整体 softmax（窗口部分自然满足因果性）
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # ── Step 8: 加权聚合 ───────────────────────────────────────────
        context = torch.matmul(attn_probs.unsqueeze(3), v_final).squeeze(3)
        # context: [B, nh, T, hd]

        # ── Step 9: 合并多头 + 输出投影 ────────────────────────────────
        output = self._merge_heads(context, B)
        output = self.o_proj(output)

        return output


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_csa():
    print("\n" + "=" * 60)
    print("Test: CompressedSparseAttention (CSA)")
    print("=" * 60)

    d_model, num_heads = 512, 8
    B, T = 2, 64

    csa = CompressedSparseAttention(
        d_model=d_model,
        num_heads=num_heads,
        compression_block=4,
        index_dim=32,
        top_k_index=8,    # 小值方便测试
        window_size=8,
    )
    csa.eval()

    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out = csa(x)

    assert out.shape == (B, T, d_model), f"输出形状: {out.shape}"
    print(f"  Input: {x.shape} → Output: {out.shape}")

    # 统计信息
    head_dim = d_model // num_heads
    T_c = T // 4  # 压缩后
    print(f"  Compression ratio: {T}:{T_c} = {T/T_c:.0f}:1")
    print(f"  Each query attends to: top-k={8} + 窗口={8} = 16 entry")
    print(f"  Full attention needs: {T} 个 key-value")
    print(f"  Sparsity: 16/{T} = {16/T*100:.1f}%")


# ==============================================================================
# 第4部分: HCA (Heavily Compressed Attention)
# ==============================================================================
#
# HCA 与 CSA 的区别：
#
#             CSA                              HCA
#  Compression ratio     4:1（温和）                      128:1（激进）
#  注意力     稀疏 top-k（只看部分）           密集 dense（看全部压缩 entry）
#  滑动窗口   有（128 token 未压缩）            有（128 token 未压缩）
#  目的       长序列的细粒度检索                长序列的全局摘要
#
# HCA 的设计哲学：
#   CSA 的稀疏选择可能遗漏某些全局信息（如"第500页的主角叫小明"，
#   这在 top-1024 个片段中排不到前面），HCA 用超高Compression ratio把全文压成
#   少量 entry，dense attention 遍历全部摘要，保证全局信息不丢失。
#
#   1M token → 128:1 压缩 → ~7800 entry → 全量 attention O(7800²) ≈ 60M
#   对比原始 attention O(1M²) = 1T，计算量降低 16000 倍。


class HeavilyCompressedAttention(nn.Module):
    """
    HCA 层：重度压缩注意力，V4 混合注意力的全局摘要组件。
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        compression_block: int = 128,   # 超高Compression ratio
        window_size: int = 128,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.compression_block = compression_block
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # ── Q/K/V 投影 ──────────────────────────────────────────────────
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        # ── 重度 KV 压缩器 ───────────────────────────────────────────────
        self.kv_compressor = KVCompressor(
            self.head_dim, block_size=compression_block, mode="learned"
        )

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, -1, self.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, d_model = hidden_states.shape
        nh, hd = self.num_heads, self.head_dim

        # ── Q/K/V 投影 ─────────────────────────────────────────────────
        q = self._split_heads(self.q_proj(hidden_states), B)
        k = self._split_heads(self.k_proj(hidden_states), B)
        v = self._split_heads(self.v_proj(hidden_states), B)

        # ── 重度Compressed K/V ────────────────────────────────────────────────
        k_comp, v_comp = self.kv_compressor(k, v)
        # 1M token → ~7800 entry（128:1）

        # ── 滑动窗口 ─────────────────────────────────────────────────────
        win_size = min(self.window_size, T)
        if T > win_size:
            k_window = k[:, :, -win_size:, :]
            v_window = v[:, :, -win_size:, :]
        else:
            # 短序列不需要窗口
            k_window = v_window = None

        # ── 拼接Compressed KV + 窗口 KV ───────────────────────────────────────
        if k_window is not None:
            k_window_exp = k_window.unsqueeze(2).expand(B, nh, T, win_size, hd)
            v_window_exp = v_window.unsqueeze(2).expand(B, nh, T, win_size, hd)
            k_comp_exp = k_comp.unsqueeze(2).expand(B, nh, T, k_comp.shape[2], hd)
            v_comp_exp = v_comp.unsqueeze(2).expand(B, nh, T, v_comp.shape[2], hd)
            k_final = torch.cat([k_comp_exp, k_window_exp], dim=3)
            v_final = torch.cat([v_comp_exp, v_window_exp], dim=3)
        else:
            k_final = k_comp.unsqueeze(2).expand(B, nh, T, k_comp.shape[2], hd)
            v_final = v_comp.unsqueeze(2).expand(B, nh, T, v_comp.shape[2], hd)

        # ── Dense Attention（不稀疏选择，遍历全部压缩 entry）────────────
        attn_scores = torch.matmul(
            q.unsqueeze(3), k_final.transpose(-1, -2)
        ).squeeze(3)
        attn_scores = attn_scores * self.scale

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs.unsqueeze(3), v_final).squeeze(3)

        output = self._merge_heads(context, B)
        output = self.o_proj(output)

        return output


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_hca():
    print("\n" + "=" * 60)
    print("Test: HeavilyCompressedAttention (HCA)")
    print("=" * 60)

    d_model, num_heads = 512, 8
    B, T = 2, 256  # simulate longer sequence

    hca = HeavilyCompressedAttention(
        d_model=d_model,
        num_heads=num_heads,
        compression_block=128,
        window_size=32,
    )
    hca.eval()

    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out = hca(x)

    assert out.shape == (B, T, d_model), f"输出形状: {out.shape}"
    T_c = T // 128  # Compressed entries 数
    print(f"  Input: {x.shape} → Output: {out.shape}")
    print(f"  Compression ratio: {T}:{T_c} = 128:1")
    print(f"  Compressed entries: {T_c} + 窗口 32 = {T_c + 32} key-value")
    print(f"  Dense attention compute: O({(T_c + 32)**2}) ≈ {(T_c + 32)**2}")
    print(f"  Original attention compute: O({T**2}) ≈ {T**2}")


# ==============================================================================
# 第5部分: Hybrid Block — V4 混合注意力 Transformer Block
# ==============================================================================
#
# V4-Pro 的层排布策略：
#   第 1-2 层：HCA（建立全局上下文摘要）
#   第 3-61 层：CSA / HCA 交替（CSA=3:1 HCA，即每 4 层中 3 层 CSA + 1 层 HCA）
#   每层都保留滑动窗口（128 token 未压缩）
#
# 直觉：前 2 层先 HCA 粗读全文建立摘要，后续主要用 CSA 精读相关片段，
#       每隔几层再用 HCA 校准全局理解。


class HybridAttentionBlock(nn.Module):
    """
    V4 混合注意力 Block，可选 CSA 或 HCA Mode。
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        attn_type: str = "csa",  # "csa" or "hca"
        compression_block: int = 4,
        index_dim: int = 64,
        top_k_index: int = 1024,
        window_size: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.attn_type = attn_type

        # ── Pre-Norm ─────────────────────────────────────────────────────
        self.norm_attn = nn.LayerNorm(d_model)

        # ── 注意力层 ─────────────────────────────────────────────────────
        if attn_type == "csa":
            self.attn = CompressedSparseAttention(
                d_model=d_model,
                num_heads=num_heads,
                compression_block=compression_block,
                index_dim=index_dim,
                top_k_index=top_k_index,
                window_size=window_size,
                dropout=dropout,
            )
        elif attn_type == "hca":
            self.attn = HeavilyCompressedAttention(
                d_model=d_model,
                num_heads=num_heads,
                compression_block=compression_block,
                window_size=window_size,
                dropout=dropout,
            )
        else:
            raise ValueError(f"未知 attn_type: {attn_type}")

        # ── 简单的 FFN（可用 MoE 替换）──────────────────────────────────
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 混合注意力 + 残差 ────────────────────────────────────────────
        residual = x
        x_norm = self.norm_attn(x)
        x = self.dropout(self.attn(x_norm))
        x = residual + x

        # ── FFN + 残差 ───────────────────────────────────────────────────
        residual = x
        x_norm = self.norm_ffn(x)
        x = self.dropout(self.ffn(x_norm))
        x = residual + x

        return x


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_hybrid_block():
    print("\n" + "=" * 60)
    print("Test: HybridAttentionBlock (CSA + HCA)")
    print("=" * 60)

    d_model, num_heads = 256, 4
    B, T = 2, 128

    for attn_type in ["csa", "hca"]:
        block = HybridAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            attn_type=attn_type,
            compression_block=4 if attn_type == "csa" else 64,
            index_dim=32,
            top_k_index=16,
            window_size=16,
        )
        block.eval()

        x = torch.randn(B, T, d_model)
        with torch.no_grad():
            out = block(x)

        assert out.shape == (B, T, d_model)
        params = sum(p.numel() for p in block.parameters())
        print(f"  [{attn_type.upper()}] Input: {x.shape} → Output: {out.shape}, "
              f"Params: {params/1e3:.1f}K")


# ==============================================================================
# 第6部分: 综合测试 — 模拟 V4 的混合注意力层排布
# ==============================================================================

def build_v4_hybrid_layers(
    num_layers: int = 12,
    d_model: int = 512,
    num_heads: int = 8,
    csa_ratio: int = 3,  # CSA:HCA = 3:1
) -> nn.ModuleList:
    """
    按 V4 的层排布策略构建混合注意力层：
    前 2 层 → HCA，后续 → CSA/HCA 交替（CSA:HCA = csa_ratio:1）
    """
    layers = nn.ModuleList()

    for layer_idx in range(num_layers):
        if layer_idx < 2:
            attn_type = "hca"
            comp_block = 128
        elif (layer_idx - 2) % (csa_ratio + 1) == 0:
            attn_type = "hca"
            comp_block = 128
        else:
            attn_type = "csa"
            comp_block = 4

        layers.append(HybridAttentionBlock(
            d_model=d_model,
            num_heads=num_heads,
            attn_type=attn_type,
            compression_block=comp_block,
            index_dim=32,
            top_k_index=32,
            window_size=32,
        ))

    return layers


def test_v4_hybrid_stack():
    print("\n" + "=" * 60)
    print("Test: V4 Hybrid Attention Layer Stack")
    print("=" * 60)

    num_layers = 8
    d_model, num_heads = 256, 4
    B, T = 2, 256
    layers = build_v4_hybrid_layers(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
    )

    type_counts = {"csa": 0, "hca": 0}
    for i, layer in enumerate(layers):
        type_counts[layer.attn_type] += 1
        print(f"  Layer {i}: {layer.attn_type.upper()} "
              f"(Compression ratio {layer.attn.compression_block}:1)")

    print(f"\n  Total: CSA={type_counts['csa']}, HCA={type_counts['hca']}")
    print(f"  CSA:HCA ≈ {type_counts['csa']}:{type_counts['hca']}")

    x = torch.randn(B, T, d_model)
    total_params = 0
    with torch.no_grad():
        for layer in layers:
            x = layer(x)
            total_params += sum(p.numel() for p in layer.parameters())

    assert x.shape == (B, T, d_model)
    print(f"\n  Input: ({B}, {T}, {d_model}) → Output: {x.shape}")
    print(f"  Total params: {total_params/1e6:.2f}M")

    # ── 模拟长序列的显存Saved ──────────────────────────────────────────
    print(f"\n  === Long Sequence Inference Estimate (T=100K tokens) ===")
    head_dim = d_model // num_heads
    # 标准 MHA KV Cache
    mha_kv = 2 * num_heads * head_dim * 100000 * 2  # bytes (FP16)
    # CSA 压缩后 KV（4:1 Compression ratio）
    csa_compressed_kv = 2 * num_heads * head_dim * (100000 // 4) * 2
    print(f"  标准 MHA KV Cache: {mha_kv / 1e9:.2f} GB")
    print(f"  CSA Compressed KV Cache: {csa_compressed_kv / 1e9:.2f} GB")
    print(f"  Saved: {(1 - csa_compressed_kv/mha_kv)*100:.0f}%")


# ==============================================================================
# 主入口
# ==============================================================================

# ==============================================================================
# 第7部分: mHC (Manifold-Constrained Hyper-Connections)
#           流形约束超连接，替代标准残差连接
# ==============================================================================
#
# V3 -> V4 的第二大架构变革：用 mHC 替代标准残差连接。
#
# 标准残差的问题：
#   x_{l+1} = x_l + F(x_l)
#   堆叠 60+ 层后，信号可能不断膨胀（因为每层都"加"信息）
#   -> 前向输出爆炸 / 反向梯度爆炸 / 训练不稳定
#
# Hyper-Connections (DeepSeek-V3) 的解决方案：
#   [x_{l+1}]   [B_11  B_12]   [x_l    ]
#   [h_{l+1}] = [B_21  B_22] @ [F(x_l)]
#   引入一个 2x2 的混合矩阵 B，让 x 和 F(x) 以更灵活的方式混合
#
# mHC (V4) 的关键约束：
#   把混合矩阵 B 约束在 doubly stochastic 矩阵流形（Birkhoff polytope）上：
#   - 每行每列和为 1
#   - 所有元素非负
#   -> 谱范数 ≤ 1，保证深层堆叠时信号非膨胀
#
# Sinkhorn-Knopp 迭代：
#   交替归一化行和列，把任意方阵投影到 doubly stochastic 流形上。
#   算法：重复执行 row_norm -> col_norm，~20 步即可收敛。
#


def sinkhorn_knopp(matrix: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """
    Sinkhorn-Knopp 迭代：把方阵投影到 doubly stochastic 流形上。

    算法：
      for i in range(num_iters):
          matrix = matrix / matrix.sum(dim=-1, keepdim=True)  # 行归一化
          matrix = matrix / matrix.sum(dim=-2, keepdim=True)  # 列归一化

    Args:
        matrix:    [..., N, N] 任意方阵
        num_iters: 迭代次数（通常 20 步足够）

    Returns:
        [..., N, N] doubly stochastic 矩阵（每行每列和为 1，元素非负）
    """
    # 先通过 exp 保证非负
    matrix = torch.exp(matrix)
    for _ in range(num_iters):
        # 行归一化
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-8)
        # 列归一化
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-8)
    return matrix


class ManifoldHyperConnection(nn.Module):
    """
    流形约束超连接（mHC）：V4 的残差连接替代方案。

    数学形式：
      [x_new]   [B_11  B_12]   [x_old]
      [h_new] = [B_21  B_22] @ [F(x_old)]

    其中 B 是经过 Sinkhorn-Knopp 约束的 2x2 doubly stochastic 矩阵。

    核心性质：
      1. 谱范数 ≤ 1 -> 信号不膨胀
      2. Perron-Frobenius -> 保证至少一个全正的特征向量
      3. 深层堆叠时前向不爆、梯度不炸
    """

    def __init__(self, num_iters: int = 20):
        super().__init__()
        self.num_iters = num_iters
        # 学习原始参数 B_raw，通过 Sinkhorn-Knopp 投影到 doubly stochastic
        self.B_raw = nn.Parameter(torch.randn(2, 2) * 0.02)

    def forward(
        self,
        x: torch.Tensor,            # [B, T, d_model] 残差流
        f_x: torch.Tensor,          # [B, T, d_model] 子层输出（attention 或 FFN 的结果）
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_new: [B, T, d_model]  更新后的残差流
            h_new: [B, T, d_model]  辅助隐藏状态（可选，用于下一层的额外信息）
        """
        # Step 1: Sinkhorn-Knopp 投影 -> doubly stochastic 矩阵
        B = sinkhorn_knopp(self.B_raw, self.num_iters)  # [2, 2]

        # Step 2: 超连接混合
        # [x_new]   [B_11  B_12]   [x]
        # [h_new] = [B_21  B_22] @ [f_x]
        B = B.to(x.dtype)
        x_new = B[0, 0] * x + B[0, 1] * f_x
        h_new = B[1, 0] * x + B[1, 1] * f_x

        return x_new, h_new


class MHCBlock(nn.Module):
    """
    使用 mHC 的 Transformer Block，对比标准 Pre-Norm Residual Block：

    标准残差：
      x = x + attn(norm(x))
      x = x + ffn(norm(x))

    mHC：
      x_norm = norm(x)
      attn_out = attn(x_norm)
      x, h = mhc_attn(x, attn_out)   # 用 mHC 替代 x + attn_out

      x_norm = norm(x)
      ffn_out = ffn(x_norm)
      x, h = mhc_ffn(x, ffn_out)     # 用 mHC 替代 x + ffn_out
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── 注意力子层 ──────────────────────────────────────────────────
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.mhc_attn = ManifoldHyperConnection()

        # ── FFN 子层 ────────────────────────────────────────────────────
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.SiLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.mhc_ffn = ManifoldHyperConnection()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ── 注意力子层 (mHC 替代残差) ───────────────────────────────────
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        attn_out = self.dropout(attn_out)
        x, h_attn = self.mhc_attn(x, attn_out)

        # ── FFN 子层 (mHC 替代残差) ─────────────────────────────────────
        x_norm = self.norm_ffn(x)
        ffn_out = self.ffn(x_norm)
        ffn_out = self.dropout(ffn_out)
        x, h_ffn = self.mhc_ffn(x, ffn_out)

        return x, h_ffn


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_mhc():
    print("\n" + "=" * 60)
    print("Test: mHC (Manifold-Constrained Hyper-Connections)")
    print("=" * 60)

    # ── 测试 Sinkhorn-Knopp ───────────────────────────────────────────
    raw = torch.randn(2, 2) * 0.5
    B = sinkhorn_knopp(raw, num_iters=20)
    row_sum = B.sum(dim=-1)
    col_sum = B.sum(dim=-2)
    print(f"  Raw matrix:\n{raw}")
    print(f"  After Sinkhorn-Knopp (doubly stochastic):\n{B}")
    print(f"  Row sums: {row_sum.tolist()}  (should be ~1.0)")
    print(f"  Col sums: {col_sum.tolist()}  (should be ~1.0)")
    assert torch.allclose(row_sum, torch.ones(2), atol=1e-3), f"Row sums != 1: {row_sum}"
    assert torch.allclose(col_sum, torch.ones(2), atol=1e-3), f"Col sums != 1: {col_sum}"

    # ── 测试 mHC Block ────────────────────────────────────────────────
    d_model, num_heads, d_ff = 256, 4, 512
    B, T = 2, 32

    block = MHCBlock(d_model, num_heads, d_ff)
    block.eval()

    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out, h = block(x)

    assert out.shape == (B, T, d_model), f"Output shape: {out.shape}"
    assert h.shape == (B, T, d_model), f"Hidden state shape: {h.shape}"
    print(f"  Input: {x.shape} -> Output: {out.shape}, Hidden: {h.shape}")

    # ── 验证 mHC 的稳定性质 ────────────────────────────────────────────
    # Deep stack: 50 layers, check output doesn't explode
    n_layers = 50
    layers = nn.ModuleList([
        MHCBlock(d_model, num_heads, d_ff) for _ in range(n_layers)
    ])
    layers.eval()

    x_deep = torch.randn(1, 16, d_model)
    with torch.no_grad():
        for layer in layers:
            x_deep, _ = layer(x_deep)

    out_norm = x_deep.norm().item()
    print(f"  50-layer deep stack output norm: {out_norm:.3f}")
    print(f"  (With standard residuals, norm would typically explode)")

    # ── 对比标准残差 vs mHC 的深层稳定性 ───────────────────────────────
    class StandardResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_attn = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
            self.norm_ffn = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False),
            )

        def forward(self, x):
            x_norm = self.norm_attn(x)
            a_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            x = x + a_out
            x_norm = self.norm_ffn(x)
            x = x + self.ffn(x_norm)
            return x

    std_layers = nn.ModuleList([StandardResBlock() for _ in range(n_layers)])
    std_layers.eval()
    x_std = torch.randn(1, 16, d_model)
    with torch.no_grad():
        for layer in std_layers:
            x_std = layer(x_std)

    std_norm = x_std.norm().item()
    print(f"\n  Comparison (same random init, 50 layers):")
    print(f"    mHC stack output norm:   {out_norm:.3f}")
    print(f"    Standard residual norm:  {std_norm:.3f}")
    print(f"    mHC is more stable:      {out_norm < std_norm}")


# ==============================================================================
# 第8部分: DeepSeek-V4 高级 MoE
#          Sqrt(Softplus) 路由 + Hash 路由 + Anticipatory Routing
# ==============================================================================
#
# V4 对 MoE 的三个关键升级：
#
# 1. Sqrt(Softplus) 亲和度函数（替代 V3 的 Sigmoid）
#    - sigmoid 输出范围 [0,1]，饱和后梯度消失
#    - softplus(x) = log(1+exp(x))，范围 [0, +inf)，不饱和
#    - sqrt 压平大值，防止少数专家主导路由
#    - 效果：路由分布更均匀，专家利用率更高
#
# 2. Hash 路由（早期层替代 dense FFN）
#    - 静态映射：token_id -> expert_id（不需要学习）
#    - 动机：早期层的 token 语义很简单，静态分桶比动态路由更高效
#    - 实现：hash(token_id) % num_experts
#
# 3. Anticipatory Routing（预期路由）
#    - 问题：路由器和专家参数同时更新，导致"追尾"效应
#      （router 刚选好专家，专家参数又变了）
#    - 解决：用历史参数算路由分数、当前参数算专家特征
#    - 实现：维护 router weight 的 EMA，取 top-k 时用 EMA 版本


class SqrtSoftplusRouter(nn.Module):
    """
    V4 的路由器：Sqrt(Softplus) 亲和度 + top-k 选择。

    与 V3 Sigmoid 路由的对比：
      Sigmoid:    score = sigmoid(Wx)     -> 范围 [0,1]，容易饱和
      Softplus:   score = log(1+exp(Wx))  -> 范围 [0,+inf)，不饱和
      SqrtSoftplus: score = sqrt(softplus(Wx)) -> 压平大值，更均匀

    直觉：sqrt 让"特别适合某专家的 token"和"一般适合的 token"差距缩小，
          鼓励路由器使用更多不同的专家。
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Anticipatory Routing: EMA of gate weight
        self.register_buffer(
            "gate_ema",
            self.gate.weight.data.clone(),
        )
        self.ema_momentum = 0.9

    def _sqrt_softplus(self, x: torch.Tensor) -> torch.Tensor:
        """Sqrt(Softplus(x)) 亲和度函数"""
        return torch.sqrt(F.softplus(x))

    def forward(
        self,
        x: torch.Tensor,
        use_ema: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       [num_tokens, d_model]
            use_ema: 是否用 EMA 权重计算路由（Anticipatory Routing）
        Returns:
            topk_indices: [num_tokens, top_k]
            topk_weights: [num_tokens, top_k]
        """
        # Anticipatory Routing: 用 EMA 权重算路由，避免追尾效应
        if use_ema and self.training:
            gate_w = self.gate_ema
        else:
            gate_w = self.gate.weight

        # Sqrt(Softplus) 亲和度
        logits = F.linear(x, gate_w)  # [num_tokens, num_experts]
        affinity = self._sqrt_softplus(logits)

        # Top-k 选择 + 归一化
        topk_vals, topk_indices = torch.topk(affinity, self.top_k, dim=-1)
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        return topk_indices, topk_weights

    def update_ema(self):
        """训练后更新 EMA（每个 step 调用一次）"""
        with torch.no_grad():
            self.gate_ema.mul_(self.ema_momentum).add_(
                self.gate.weight.data, alpha=1 - self.ema_momentum
            )


class HashRouter(nn.Module):
    """
    Hash 路由：静态 token-id -> expert 映射，无需学习。

    V4 在最早几层使用 Hash 路由替代 dense FFN 或动态 MoE：
    - 早期 token 语义简单，不需要复杂的动态路由
    - Hash 是确定性的，零计算开销（vs 动态路由需要 gate 计算）
    - 保证绝对的负载均衡（hash 均匀分布）

    实现：expert_id = hash(token_id) % num_experts
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(
        self,
        token_ids: torch.Tensor,  # [num_tokens]  token 在词表中的 ID
    ) -> torch.Tensor:
        """
        Returns:
            expert_ids: [num_tokens]  每个 token 分配的专家 ID
        """
        # 简单 hash: 利用 Python 内置 hash 的分布性
        # 在生产中会使用更快的位运算 hash
        token_ids_cpu = token_ids.cpu().tolist()
        hash_ids = torch.tensor(
            [hash(tid) % self.num_experts for tid in token_ids_cpu],
            dtype=torch.long,
            device=token_ids.device,
        )
        return hash_ids


class V4MoELayer(nn.Module):
    """
    V4 风格的 MoE 层：Sqrt(Softplus) 路由 + Anticipatory Routing + Expert FFN。

    可以直接替换标准 FFN 层。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        use_aux_loss: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_aux_loss = use_aux_loss

        # V4 风格路由器
        self.router = SqrtSoftplusRouter(d_model, num_experts, top_k)

        # 专家池
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:         [B, T, d_model]
            token_ids: [B, T] token IDs (for hash routing, optional)
        Returns:
            output:  [B, T, d_model]
            aux_loss: scalar
        """
        B, T, d_model = x.shape
        x_flat = x.reshape(B * T, d_model)

        # ── 路由 ────────────────────────────────────────────────────────
        topk_indices, topk_weights = self.router(x_flat)
        # topk_indices: [B*T, top_k]

        # ── 专家分发 + 加权合并 ─────────────────────────────────────────
        output_flat = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            expert_mask = (topk_indices == expert_id)
            token_mask = expert_mask.any(dim=-1)

            if not token_mask.any():
                continue

            expert_input = x_flat[token_mask]
            expert_output = self.experts[expert_id](expert_input)

            weights = topk_weights[token_mask][expert_mask[token_mask]]
            output_flat[token_mask] += expert_output * weights.unsqueeze(-1)

        output = output_flat.reshape(B, T, d_model)

        # ── 辅助损失（简化版） ──────────────────────────────────────────
        if self.use_aux_loss:
            # 简单的负载均衡：鼓励每个专家被均匀使用
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.num_experts):
                expert_counts[i] = (topk_indices == i).any(dim=-1).sum().float()
            target = (B * T * self.top_k) / self.num_experts
            aux_loss = ((expert_counts - target) ** 2).mean() * 0.01
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return output, aux_loss


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_v4_moe():
    print("\n" + "=" * 60)
    print("Test: V4 Advanced MoE (SqrtSoftplus + Hash + Anticipatory)")
    print("=" * 60)

    d_model, d_ff, num_experts, top_k = 256, 512, 8, 2
    B, T = 2, 16

    # ── SqrtSoftplus vs Sigmoid 对比 ────────────────────────────────────
    x_test = torch.linspace(-5, 5, 100)
    sigmoid_out = torch.sigmoid(x_test)
    softplus_out = F.softplus(x_test)
    sqrt_softplus_out = torch.sqrt(F.softplus(x_test))

    print(f"  Affinity function comparison (x in [-5, 5]):")
    print(f"    Sigmoid:       range [{sigmoid_out.min():.2f}, {sigmoid_out.max():.2f}]")
    print(f"    Softplus:      range [{softplus_out.min():.2f}, {softplus_out.max():.2f}]")
    print(f"    SqrtSoftplus:  range [{sqrt_softplus_out.min():.2f}, {sqrt_softplus_out.max():.2f}]")
    print(f"    -> SqrtSoftplus doesn't saturate at 0, wider dynamic range")

    # ── V4 MoE 层测试 ──────────────────────────────────────────────────
    moe = V4MoELayer(d_model, d_ff, num_experts, top_k)
    moe.eval()

    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out, aux = moe(x)

    assert out.shape == (B, T, d_model)
    print(f"  Input: {x.shape} -> Output: {out.shape}, aux_loss: {aux.item():.6f}")

    # ── Hash 路由测试 ──────────────────────────────────────────────────
    hash_router = HashRouter(num_experts)
    token_ids = torch.randint(0, 50000, (32,))
    hash_experts = hash_router(token_ids)

    # 验证均匀性
    expert_counts = torch.bincount(hash_experts, minlength=num_experts)
    print(f"\n  Hash routing distribution (32 tokens, {num_experts} experts):")
    print(f"    Expert counts: {expert_counts.tolist()}")
    print(f"    Mean: {32/num_experts:.1f}, Std: {expert_counts.float().std().item():.1f}")

    # ── Anticipatory Routing 测试 ────────────────────────────────────
    moe.train()
    x2 = torch.randn(B, T, d_model)
    out2, _ = moe(x2)
    moe.router.update_ema()
    ema_diff = (moe.router.gate_ema - moe.router.gate.weight.data).norm().item()
    print(f"\n  Anticipatory Routing:")
    print(f"    Gate weight vs EMA diff: {ema_diff:.6f}")
    print(f"    EMA tracks weight with momentum 0.9")

    # ── 路由亲和度函数可视化对比 ──────────────────────────────────────
    print(f"\n  Affinity visualization (extreme input ranges):")
    gate = moe.router.gate
    extreme_test = torch.tensor([[-10.0], [-1.0], [0.0], [1.0], [10.0]])
    with torch.no_grad():
        logits_test = gate(extreme_test.expand(-1, d_model))
        sigmoid_aff = torch.sigmoid(logits_test).mean(dim=-1)
        sqrt_sp_aff = torch.sqrt(F.softplus(logits_test)).mean(dim=-1)
        print(f"    Input scale | Sigmoid mean | SqrtSoftplus mean")
        for i, scale in enumerate([-10, -1, 0, 1, 10]):
            print(f"    {scale:>10.0f}  | {sigmoid_aff[i]:>12.6f} | {sqrt_sp_aff[i]:>13.6f}")


# ==============================================================================
# 第9部分: Mini DeepSeek-V4 Block (All Components Integrated)
# ==============================================================================

class MiniDeepSeekV4Block(nn.Module):
    """
    整合 V4 所有核心创新的 mini Transformer Block：

    - CSA/HCA 混合注意力（替代 MLA）
    - mHC 流形约束超连接（替代标准残差）
    - V4 风格 MoE FFN（SqrtSoftplus + Anticipatory Routing）

    这是 V4 架构的完整微缩版。
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_experts: int = 8,
        top_k: int = 2,
        attn_type: str = "csa",
        index_dim: int = 32,
        top_k_index: int = 32,
        window_size: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ── Pre-Norm ─────────────────────────────────────────────────────
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        # ── CSA/HCA 混合注意力 ──────────────────────────────────────────
        if attn_type == "csa":
            self.attn = CompressedSparseAttention(
                d_model=d_model,
                num_heads=num_heads,
                compression_block=4,
                index_dim=index_dim,
                top_k_index=top_k_index,
                window_size=window_size,
                dropout=dropout,
            )
        else:
            self.attn = HeavilyCompressedAttention(
                d_model=d_model,
                num_heads=num_heads,
                compression_block=128,
                window_size=window_size,
                dropout=dropout,
            )

        # ── mHC 替代残差连接 ────────────────────────────────────────────
        self.mhc_attn = ManifoldHyperConnection()
        self.mhc_ffn = ManifoldHyperConnection()

        # ── V4 风格 MoE FFN ─────────────────────────────────────────────
        self.moe = V4MoELayer(d_model, d_ff, num_experts, top_k)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x:        [B, T, d_model] updated hidden states
            h:        [B, T, d_model] auxiliary hidden state (from mHC)
            aux_loss: scalar          MoE aux loss
        """
        # ── CSA/HCA + mHC 残差 ──────────────────────────────────────────
        x_norm = self.norm_attn(x)
        attn_out = self.attn(x_norm)
        attn_out = self.dropout(attn_out)
        x, h_attn = self.mhc_attn(x, attn_out)

        # ── V4 MoE FFN + mHC 残差 ──────────────────────────────────────
        x_norm = self.norm_ffn(x)
        moe_out, aux_loss = self.moe(x_norm)
        moe_out = self.dropout(moe_out)
        x, h_ffn = self.mhc_ffn(x, moe_out)

        return x, h_ffn, aux_loss


# ── 测试 ──────────────────────────────────────────────────────────────────────
def test_mini_v4_block():
    print("\n" + "=" * 60)
    print("Test: Mini DeepSeek-V4 Block (ALL components)")
    print("=" * 60)

    d_model, num_heads, d_ff = 256, 4, 512
    num_experts, top_k = 4, 2
    B, T = 2, 64

    for attn_type in ["csa", "hca"]:
        block = MiniDeepSeekV4Block(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_experts=num_experts,
            top_k=top_k,
            attn_type=attn_type,
            index_dim=16,
            top_k_index=8,
            window_size=8,
        )
        block.eval()

        x = torch.randn(B, T, d_model)
        with torch.no_grad():
            out, h, aux = block(x)

        assert out.shape == (B, T, d_model), f"Output: {out.shape}"
        assert h.shape == (B, T, d_model), f"Hidden: {h.shape}"
        params = sum(p.numel() for p in block.parameters())
        print(f"  [{attn_type.upper()}] Input: {x.shape} -> "
              f"Output: {out.shape}, Params: {params/1e3:.1f}K, "
              f"AuxLoss: {aux.item():.6f}")

    print(f"\n  Full V4 mini-block working: CSA/HCA + mHC + V4MoE")


if __name__ == "__main__":
    torch.manual_seed(42)

    print("=" * 62)
    print("  DeepSeek-V4 Full Algorithm Tutorial")
    print("  CSA + HCA + mHC + V4MoE")
    print("=" * 62)

    # Part 1-6: Hybrid Attention
    test_kv_compressor()
    # test_lightning_indexer()
    # test_csa()
    # test_hca()
    # test_hybrid_block()
    # test_v4_hybrid_stack()
    #
    # # Part 7: mHC
    # test_mhc()
    #
    # # Part 8: Advanced MoE
    # test_v4_moe()
    #
    # # Part 9: Integration
    # test_mini_v4_block()
    #
    # print("\n" + "=" * 60)
    # print("All tests passed!")
    # print("=" * 60)
    # print("""
    # V4 Key Innovations Summary:
    # --------------------------------------------------------------
    # 1. CSA: KV compress(4:1) + Lightning Indexer(sparse select) + sliding window
    #    -> Fine-grained retrieval, each query ~1K entries
    #
    # 2. HCA: KV compress(128:1) + dense attention + sliding window
    #    -> Global summary channel, prevents CSA from missing context
    #
    # 3. Lightning Indexer: low-dim projection + top-k scoring
    #    -> Lightweight pre-filter, much cheaper than full attention
    #
    # 4. mHC: Sinkhorn-Knopp constrained doubly stochastic mixing matrix
    #    -> Replaces standard residual, prevents deep-stack signal explosion
    #
    # 5. SqrtSoftplus routing: more uniform expert utilization than Sigmoid
    #    -> + Anticipatory Routing: EMA gate for stable training
    #
    # 6. Hybrid Layout: first 2 layers HCA -> CSA/HCA alternating
    #    -> Global context foundation + efficient sparse retrieval
    #
    # 7. vs V3 MLA: MLA = per-token dimension compression
    #    CSA/HCA = temporal compression + sparse selection
    # --------------------------------------------------------------
    # """)