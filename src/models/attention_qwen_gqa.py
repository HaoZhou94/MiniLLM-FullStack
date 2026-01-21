import os
import sys
import torch
import torch.nn as nn
import math

from jinja2.compiler import F

current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, "../../"))
sys.path.append(project_root)

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
        self.num_groups = self.num_query_heads // self.num_kv_heads
        self.head_dim = dim // self.num_query_heads

        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError('num_query_heads must be divisible by num_kv_heads')
        if dim % self.num_query_heads != 0:
            raise ValueError('dim must be divisible by num_query_heads')


        # === 投影层设计 (Qwen官方结构) ===
        self.q_proj = nn.Linear(dim, self.num_query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_query_heads * self.head_dim, dim, bias=False)

        # === RoPE层 ====
        self.rope = DynamicRoPE(self.dim, self.num_query_heads)

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
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # === 2. 应用RoPE ===
        if positions is None:
            positions = torch.arange(seq_len, device=device)
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        if use_cache:
            # === 3. KV缓存增量更新 ===
            cache_pos = self.cache_position.item()
            self.kv_cache_k[:batch_size, :, cache_pos:cache_pos+seq_len, :] = k
            self.kv_cache_v[:batch_size, :, cache_pos:cache_pos+seq_len, :] = v
            """
            关键：只往“当前页码到当前页码+新token数”的位置填值
            你的例子：
            - cache_pos=0，seq_len=1 → 填0:1的位置（第0页）
            - k的形状是[1,2,1,16]（新token的K）
            - 赋值后：kv_cache_k[1,2,0:1,16] = 新K（第0页不再是0，其他页还是0）
            比喻：把新token的K/V写在笔记本第0页，其他页仍空白
            """

            # 3. 推理时使用完整缓存的KV（历史+新）
            k = self.kv_cache_k[:batch_size, :, :cache_pos+seq_len, :]
            v = self.kv_cache_v[:batch_size, :, :cache_pos+seq_len, :]
            """
            关键：读取“从开头到当前页码+新token数”的所有KV（历史+新）
            你的例子：
            - cache_pos+seq_len=1 → 读:1的位置（第0页）
            - k的形状变成[1,2,1,16]（缓存里的新K）
            比喻：翻笔记本，从第一页读到刚写完的第0页，拿到所有已写的内容
            """

            # 4. 更新缓存位置（页码往后翻）
            self.cache_position += seq_len

            # 5. 重新计算有效序列长度（缓存里已存的token数）
            kv_seq_len = cache_pos + seq_len  # 0+1=1 → 有效长度1
            """
            作用：告诉注意力计算，现在要用到的KV总长度是1（仅新token）
            """
        else:
            kv_seq_len = seq_len  # 不用缓存时，有效长度就是当前输入的长度


        # ===== 4. GQA核心：KV按组扩展（Qwen官方分组逻辑）=====
        # 扩展维度：[B, Hkv, L, D] → [B, Hq, L, D]
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        # ===== 5. 注意力分数计算（Qwen官方缩放逻辑）=====
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        """
        核心作用：给注意力分数加因果掩码，强制模型只能看前面的 token，不能看后面的；
        操作逻辑：把禁止看到的位置的注意力分数设为负无穷，Softmax 后这些位置概率为 0；
        ~causal_mask：取反掩码，让 “禁止看到的位置” 变成 True，从而被填充负无穷；
        必要性：保证模型按 “时间顺序” 生成文本，避免 “作弊”，是所有生成式大模型的标配。
        """
        try:
            casual_mask = self._create_causal_mask(kv_seq_len, device)
            import pdb; pdb.set_trace()

            casual_mask = casual_mask.expand(batch_size, self.num_query_heads, -1, -1)

            # 2. 对禁止看到的位置填充负无穷
            # attn_scores = attn_scores.masked_fill(~casual_mask, float("-inf"))
            attn_scores = attn_scores.masked_fill(~casual_mask[:, :, -seq_len:, :], float("-inf"))
            attn_weights = nn.functional.softmax(attn_scores, dim=-1)

            score_shape = attn_scores.shape
            mask_shape = casual_mask.shape
            print(f"- attn_scores形状: {score_shape}, casual_mask形状: {mask_shape}")

            # ===== 6. 加权求和 + 输出投影（Qwen官方结构）=====
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_query_heads * self.head_dim)
            output = self.o_proj(attn_output)
            return output

        except Exception as e:
            # ========== 错误定位核心：打印所有关键信息 ==========
            print("\n" + "="*80)
            print(f"❌ 异常类型：{type(e).__name__}")
            print(f"❌ 异常信息：{str(e)}")
            print("="*80)

            print("\n【维度匹配检查】")
                # 检查掩码和分数的维度是否匹配
            score_shape = attn_scores.shape
            mask_shape = casual_mask.shape
            print(f"- attn_scores形状: {score_shape}, casual_mask形状: {mask_shape}")


    def reset_cache(self):
        """Qwen官方KV缓存重置方法（推理时复用）"""
        self.cache_position.zero_()
        self.kv_cache_k.zero_()
        self.kv_cache_v.zero_()


# ==================== Qwen GQA 标准验证 ====================
if __name__ == "__main__":
#     # 模拟Qwen-7B小尺度配置（适配Micro）
    dim = 64
    num_query_heads = 4  # Q头数
    num_kv_heads = 2  # KV头数（分组数=2）
    max_seq_len = 32

#     # 初始化标准GQA
    gqa = QwenGQA(dim=dim, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads, max_seq_len=max_seq_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gqa = gqa.to(device)
#
#     # 1. 训练模式（禁用缓存）
    print("=== Qwen GQA 训练模式验证 ===")
    train_x = torch.randn(1, 4, dim, device=device)  # [B=1, L=4, D=64]
    train_out = gqa(train_x, use_cache=False)
    print(f"输入形状：{train_x.shape}")
    print(f"输出形状：{train_out.shape}")  # [1,4,64]（标准输出）

#     # 2. 推理模式（启用KV缓存）
    print("\n=== Qwen GQA 推理模式验证（KV缓存）===")
    gqa.reset_cache()  # 重置缓存
    # 第一步：输入第一个token
    step1_x = torch.randn(1, 1, dim, device=device)
    """
    核心作用是为 “序列中第 0 个位置的 token” 计算专属的旋转角度，通过旋转 Q/K 向量的奇偶维度，给向量注入 “我是第 0 个位置” 的位置语义
    """
    step1_out = gqa(step1_x, use_cache=True, positions=torch.tensor([0], device=device))
    print(f"第一步输出形状：{step1_out.shape}")  # [1,1,64]
    print(f"缓存位置：{gqa.cache_position.item()}")  # 1（缓存1个token）
    # 第二步：输入第二个token
    step2_x = torch.randn(1, 1, dim, device=device)
    step2_out = gqa(step2_x, use_cache=True, positions=torch.tensor([1], device=device))
    print(f"第二步输出形状：{step2_out.shape}")  # [1,1,64]
    print(f"缓存位置：{gqa.cache_position.item()}")  # 2（缓存2个token）

#     # 3. 维度验证（Qwen官方标准）
#     print("\n=== Qwen GQA 维度验证 ===")
#     q = gqa.q_proj(train_x).view(1, 4, num_query_heads, gqa.head_dim).transpose(1, 2)
#     k = gqa.k_proj(train_x).view(1, 4, num_kv_heads, gqa.head_dim).transpose(1, 2)
#     print(f"Q投影维度：{q.shape}")  # [1,4,4,16]（Hq=4, D=16）
#     print(f"K原始维度：{k.shape}")  # [1,2,4,16]（Hkv=2, D=16）
#     k_expanded = k.repeat_interleave(gqa.num_groups, dim=1)
#     print(f"K扩展维度：{k_expanded.shape}")  # [1,4,4,16]（匹配Q头数，官方标准）