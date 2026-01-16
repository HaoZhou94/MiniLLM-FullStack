import torch
import torch.nn as nn
import math
from src.models.rope import DynamicRoPE
from src.utils.logger import logger


class QwenGQA(nn.Module):
    """
    Qwen官方标准GQA实现（对齐Qwen-7B/14B源码）
    核心参数：
    - dim: 模型总维度
    - num_query_heads: Q头数（如Qwen-7B为28）
    - num_kv_heads: K/V头数（如Qwen-7B为4）
    - max_seq_len: 最大序列长度（用于KV缓存初始化）
    """
    def __init__(self, dim:int, num_query_heads:int, num_kv_heads:int, max_seq_len:int=8192):
        super().__init__()
        self.dim = dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_group = self.num_query_heads // self.num_kv_heads
        self.head_dim = dim // self.num_query_heads

        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError('num_query_heads must be divisible by num_kv_heads')
        if dim % self.num_query_heads != 0:
            raise ValueError('dim must be divisible by num_query_heads')


        # === 投影层设计 (Qwen官方结构) ===
        self.q_proj = nn.Linear(dim, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, self.num_query_heads * self.head_dim, bias=False)


        # === RoPE层 ====
        self.rope = DynamicRoPE(dim, self.num_query_heads)

        # === 标准实现 KV缓存初始化 (推理优化) ===
        self.max_seq_len = max_seq_len
        self.register_buffer('kv_cache_k', torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim))
        self.register_buffer('kv_cache_v', torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim))
        self.register_buffer('cache_position', torch.zeros(1, dtype=torch.long))

        logger.info(
            f"初始化Qwen GQA（标准实现）：num_query_heads={num_query_heads}, num_kv_heads={num_kv_heads}, "
            f"num_groups={self.num_groups}, head_dim={self.head_dim}"
        )


    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Qwen官方因果掩码生成逻辑"""
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,seq_len]


    def forward(
            self,
            x: torch.Tensor,
            use_cache: bool=False,
            positions: torch.Tensor=None
    ) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape
        device = x.device

        # === 1. Q/K/V投影 ===







        # === 2. 应用RoPE ===





        # === 3. KV缓存增量更新 ===




        # ===== 4. GQA核心：KV按组扩展（Qwen官方分组逻辑）=====
        # 扩展维度：[B, Hkv, L, D] → [B, Hq, L, D]




        # ===== 5. 注意力分数计算（Qwen官方缩放逻辑）=====




        # ===== 6. 加权求和 + 输出投影（Qwen官方结构）=====






# class QwenGQA(nn.Module):
#     """
#     Qwen官方标准GQA实现（对齐Qwen-7B/14B源码）
#     核心参数：
#     - dim: 模型总维度
#     - num_query_heads: Q头数（如Qwen-7B为28）
#     - num_kv_heads: K/V头数（如Qwen-7B为4）
#     - max_seq_len: 最大序列长度（用于KV缓存初始化）
#     """
#
#     def __init__(self, dim: int, num_query_heads: int, num_kv_heads: int, max_seq_len: int = 8192):
#         super().__init__()
#         # ===== 标准实现要点1：维度校验（Qwen官方强制规则）=====
#         self.dim = dim
#         self.num_query_heads = num_query_heads
#         self.num_kv_heads = num_kv_heads
#         self.num_groups = self.num_query_heads // self.num_kv_heads  # 分组数（Qwen-7B为7）
#         self.head_dim = dim // self.num_query_heads  # 按Q头数计算head_dim（官方标准）
#
#         if self.num_query_heads % self.num_kv_heads != 0:
#             raise ValueError(
#                 f"Qwen GQA要求num_query_heads({num_query_heads})必须是num_kv_heads({num_kv_heads})的整数倍")
#         if dim % self.num_query_heads != 0:
#             raise ValueError(f"模型维度{dim}必须能被num_query_heads({num_query_heads})整除")
#
#         # ===== 标准实现要点2：投影层设计（Qwen官方结构）=====
#         self.q_proj = nn.Linear(dim, self.num_query_heads * self.head_dim, bias=False)
#         self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
#         self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
#         self.o_proj = nn.Linear(self.num_query_heads * self.head_dim, dim, bias=False)
#
#         # ===== 标准实现要点3：RoPE层（对齐Qwen动态RoPE）=====
#         self.rope = DynamicRoPE(dim, self.num_query_heads)
#
#         # ===== 标准实现要点4：KV缓存初始化（推理优化）=====
#         self.max_seq_len = max_seq_len
#         self.register_buffer("kv_cache_k", torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim))
#         self.register_buffer("kv_cache_v", torch.zeros(1, self.num_kv_heads, max_seq_len, self.head_dim))
#         self.register_buffer("cache_position", torch.zeros(1, dtype=torch.long))
#
#         logger.info(
#             f"初始化Qwen GQA（标准实现）：num_query_heads={num_query_heads}, num_kv_heads={num_kv_heads}, "
#             f"num_groups={self.num_groups}, head_dim={self.head_dim}"
#         )
#
#     def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
#         """Qwen官方因果掩码生成逻辑"""
#         mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
#         return mask.unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,seq_len]
#
#     def forward(
#             self,
#             x: torch.Tensor,
#             use_cache: bool = False,  # 是否启用KV缓存（推理=True，训练=False）
#             positions: torch.Tensor = None  # 位置编码（推理时传入增量位置）
#     ) -> torch.Tensor:
#         batch_size, seq_len, _ = x.shape
#         device = x.device
#
#         # ===== 1. Q/K/V投影（Qwen官方投影逻辑）=====
#         q = self.q_proj(x).view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1,
#                                                                                                     2)  # [B, Hq, L, D]
#         k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, L, D]
#         v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, L, D]
#
#         # ===== 2. 应用RoPE（Qwen官方位置编码逻辑）=====
#         if positions is None:
#             positions = torch.arange(seq_len, device=device)
#         q = self.rope(q, positions)
#         k = self.rope(k, positions)
#
#         # ===== 3. KV缓存增量更新（Qwen推理核心优化）=====
#         if use_cache:
#             # 取出缓存位置
#             cache_pos = self.cache_position.item()
#             # 更新缓存（仅存储新token的KV）
#             self.kv_cache_k[:batch_size, :, cache_pos:cache_pos + seq_len, :] = k
#             self.kv_cache_v[:batch_size, :, cache_pos:cache_pos + seq_len, :] = v
#             # 推理时使用完整缓存的KV
#             k = self.kv_cache_k[:batch_size, :, :cache_pos + seq_len, :]
#             v = self.kv_cache_v[:batch_size, :, :cache_pos + seq_len, :]
#             # 更新缓存位置
#             self.cache_position += seq_len
#             # 重新计算有效序列长度
#             kv_seq_len = cache_pos + seq_len
#         else:
#             kv_seq_len = seq_len
#
#         # ===== 4. GQA核心：KV按组扩展（Qwen官方分组逻辑）=====
#         # 扩展维度：[B, Hkv, L, D] → [B, Hq, L, D]
#         k = k.repeat_interleave(self.num_groups, dim=1)
#         v = v.repeat_interleave(self.num_groups, dim=1)
#
#         # ===== 5. 注意力分数计算（Qwen官方缩放逻辑）=====
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
#         # 应用因果掩码
#         causal_mask = self._create_causal_mask(kv_seq_len, device)
#         attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
#         # Softmax归一化
#         attn_weights = nn.functional.softmax(attn_scores, dim=-1)
#
#         # ===== 6. 加权求和 + 输出投影（Qwen官方结构）=====
#         attn_output = torch.matmul(attn_weights, v)  # [B, Hq, L, D]
#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [B, L, Hq*D]
#         output = self.o_proj(attn_output)  # [B, L, dim]
#
#         return output
#
#     def reset_cache(self):
#         """Qwen官方KV缓存重置方法（推理时复用）"""
#         self.cache_position.zero_()
#         self.kv_cache_k.zero_()
#         self.kv_cache_v.zero_()
#
#
# # ==================== Qwen GQA 标准验证 ====================
# if __name__ == "__main__":
#     # 模拟Qwen-7B小尺度配置（适配Micro）
#     dim = 64
#     num_query_heads = 4  # Q头数
#     num_kv_heads = 2  # KV头数（分组数=2）
#     max_seq_len = 32
#
#     # 初始化标准GQA
#     gqa = QwenGQA(dim=dim, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads, max_seq_len=max_seq_len)
#     device = torch.device("cpu")
#     gqa = gqa.to(device)
#
#     # 1. 训练模式（禁用缓存）
#     print("=== Qwen GQA 训练模式验证 ===")
#     train_x = torch.randn(1, 4, dim, device=device)  # [B=1, L=4, D=64]
#     train_out = gqa(train_x, use_cache=False)
#     print(f"输入形状：{train_x.shape}")
#     print(f"输出形状：{train_out.shape}")  # [1,4,64]（标准输出）
#
#     # 2. 推理模式（启用KV缓存）
#     print("\n=== Qwen GQA 推理模式验证（KV缓存）===")
#     gqa.reset_cache()  # 重置缓存
#     # 第一步：输入第一个token
#     step1_x = torch.randn(1, 1, dim, device=device)
#     step1_out = gqa(step1_x, use_cache=True, positions=torch.tensor([0], device=device))
#     print(f"第一步输出形状：{step1_out.shape}")  # [1,1,64]
#     print(f"缓存位置：{gqa.cache_position.item()}")  # 1（缓存1个token）
#     # 第二步：输入第二个token
#     step2_x = torch.randn(1, 1, dim, device=device)
#     step2_out = gqa(step2_x, use_cache=True, positions=torch.tensor([1], device=device))
#     print(f"第二步输出形状：{step2_out.shape}")  # [1,1,64]
#     print(f"缓存位置：{gqa.cache_position.item()}")  # 2（缓存2个token）
#
#     # 3. 维度验证（Qwen官方标准）
#     print("\n=== Qwen GQA 维度验证 ===")
#     q = gqa.q_proj(train_x).view(1, 4, num_query_heads, gqa.head_dim).transpose(1, 2)
#     k = gqa.k_proj(train_x).view(1, 4, num_kv_heads, gqa.head_dim).transpose(1, 2)
#     print(f"Q投影维度：{q.shape}")  # [1,4,4,16]（Hq=4, D=16）
#     print(f"K原始维度：{k.shape}")  # [1,2,4,16]（Hkv=2, D=16）
#     k_expanded = k.repeat_interleave(gqa.num_groups, dim=1)
#     print(f"K扩展维度：{k_expanded.shape}")  # [1,4,4,16]（匹配Q头数，官方标准）