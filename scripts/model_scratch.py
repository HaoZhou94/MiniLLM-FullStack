import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

from llama.model import repeat_kv


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float().type_as(x))
        return output * self.weight


"""
RoPE 的核心是将位置信息编码到注意力机制的 query/key 向量中，通过对向量的偶数 / 奇数维度进行旋转操作实现，核心公式如下：

对于维度为d的向量x（位置为m），其旋转后的值为：
- 偶数位 (x_{2i}) : x_{2i} * cos(mθ_i) − x_{2i+1} * sin(mθ_i)
- 奇数位 (x_{2i+1}) : x_{2i} * sin(mθ_i) + x_{2i+1} * cos(mθ_i)

其中θ_i = 10000^(-2i/d)（这是 RoPE 的频率初始化规则），代码的核心就是预计算所有位置和维度对应的 cos(mθ_i) 和 sin(mθ_i)（通过复数极坐标形式简化计算）。
"""

def precompute_freqs_cis(dim: int, end:int, theta:float = 10000.0):
    """
    预计算RoPE所需的复数旋转因子（freqs_cis）
    Args:
        dim: 每个token的特征维度（必须是偶数，RoPE按两两维度旋转）
        end: 最大序列长度（要编码的最大位置数）
        theta: RoPE的基础频率常数，默认10000（原论文设定）
    Returns:
        freqs_cis: [end, dim//2]的复数张量，存储cos和sin（极坐标形式）
    """
    # 1. 计算每个维度对的基础频率 θ_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    # 2. 生成位置序列 t (0,1,2,...,end-1)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # 3. 计算每个位置+维度对的旋转角度 m*theta_i
    freqs = torch.outer(t, freqs)
    # 4. 将角度转换为复数形式(cos+i*sin)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis      # [seq_len, head_dim//2]


# xq/xk 的形状（通常是 [batch, seq_len, n_head, head_dim//2]）
def reshape_for_broadcast(freqs_cis: torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim-1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)  #[1, seq_len, 1, head_dim//2]



def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq_), xk_out.type_as(xk_)


class Attention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        # 基础参数定义
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads


        # 标准线性层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_kv_heads * self.head_dim, args.dim, bias=False)


        # 显存优化： KV Cache 预分配
        cache_shape = (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.register_buffer('cache_k', torch.zeros(cache_shape, dtype=torch.bfloat16))
        self.register_buffer('cache_v', torch.zeros(cache_shape, dtype=torch.bfloat16))


    def forward(self, x, start_pos, freqs_cis, mask=None):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        self.cache_k[:batch_size,:start_pos + seq_len] = xk
        self.cache_v[:batch_size,:start_pos + seq_len] = xv

       # 读取当前及历史所有的 KV
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        import pdb;pdb.set_trace()

       # GQA 核心：如果 KV 头数少于 Q，则进行重复扩展
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

       # 矩阵乘法 [batch, heads, seq, head_dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算 Attention Scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask

        # 数值稳定性：Softmax 前转为 float32
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # 加权求和并转换回原始维度
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 1. 基础参数定义
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # 2. 替换并行层为标准线性层
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # 3. 显存优化：KV Cache 预分配
        # 建议 max_seq_len 设为 2048 以节省 3080 显存
        cache_shape = (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=torch.bfloat16))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=torch.bfloat16))

    def forward(self, x, start_pos, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        
        # 投影
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 重塑维度以匹配 RoPE 处理 [batch, seq, heads, head_dim]
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 应用旋转位置编码 (RoPE)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # 更新静态 KV Cache
        # 这里直接对 slice 赋值，避免了 torch.cat 带来的额外显存申请
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # 读取当前及历史所有的 KV
        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # GQA 核心：如果 KV 头数少于 Q，则进行重复扩展
        keys = repeat_kv(keys, self.n_rep) 
        values = repeat_kv(values, self.n_rep)

        # 矩阵乘法 [batch, heads, seq, head_dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算 Attention Scores
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        
        # 数值稳定性：Softmax 前转为 float32
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # 加权求和并转换回原始维度
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.wo(output)
"""