
import os
import sys
import torch
import torch.nn as nn
import math

# 1. 获取当前文件（attention.py）的绝对路径
current_file_path = os.path.abspath(__file__)
# 2. 获取当前文件所在目录（src/models/）
current_dir = os.path.dirname(current_file_path)
# 3. 向上两级目录，找到项目根目录（MiniLLM-FullStack/）
project_root = os.path.dirname(os.path.dirname(current_dir))
# 4. 将项目根目录添加到Python的模块搜索路径
sys.path.append(project_root)
from src.models.rope import DynamicRoPE
from src.utils.logger import logger


class MaskedMultiHeadAttention(nn.Module):
    """掩码多头注意力（Decoder专用）"""
    def __init__(self, dim:int, num_heads:int, dropout:float=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 检测维度合法性
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        # Q/K/V投影层
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # 输出投影层
        self.out_proj = nn.Linear(dim, dim)

        # RoPE层
        self.rope = DynamicRoPE(dim, num_heads)

        logger.info(f"初始化注意力层：维度={dim}，头数={num_heads}")


    def _create_mask(self, seq_len:int) -> torch.Tensor:
        """生成未来掩码：遮挡未来Token"""
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).to(self.q_proj.weight.device)
        return ~mask # True表示遮挡

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: 输入特征, shape:[batch_size, seq_len, embed_dim]
        :return: torch.Tensor: 注意力输出，shape同x
        """

        batch_size, seq_len, dim = x.shape

        # 1. 线性投影并拆分多头：[batch_size, num_heads, seq_len, head_dim]
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


        # 2. 应用动态RoPE到Q和K(V不需要位置编码)
        pos = torch.arange(seq_len).to(x.device)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        """
        注意力分数的核心是：对每个 Token 的 q 向量，计算它与序列中所有 Token 的 k 向量的点积，最终得到一个[seq_len, seq_len]的分数矩阵（每个位置对所有位置的注意力权重）
        """
        # 3. 计算注意力机制分数: Q @ K^T / sqrt(head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 应用未来掩码
        mask = self._create_mask(seq_len)
        attn_scores = attn_scores.masked_fill(mask, -1e9)

        # 5.Softmax归一化+加权求和
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # 6. 拼接多头并投影
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.out_proj(attn_output)
        return out

if __name__ == "__main__":
    # 初始化注意力层（适配Qwen-Micro）
    attn = MaskedMultiHeadAttention(dim=64,num_heads=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = attn.to(device)

    # 构造测试输入：模拟Decoder的输入
    # 形状【batch_size=1， seq_len=4，dim=64】
    x = torch.randn(1, 4, 64).to(device)

    # 前向传播
    output = attn(x)
    print("Attention Output Shape:", output.shape)  # 应输出：[1, 4, 64]
