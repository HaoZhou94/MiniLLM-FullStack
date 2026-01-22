import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DeepSeekMLA(nn.Module):
    def __init__(self,
            hidden_size: int = 2048,
            num_attention_heads: int = 16,
            attention_dropout: float = 0.0,
            hidden_dropout: float = 0.0,
            bias: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size,
        self.num_attention_heads = num_attention_heads,
        self.attention_head_size = hidden_size // num_attention_heads  # 每个头的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        # 确保hidden_size能被注意力头数整除（DeepSeek的核心约束）
        assert self.all_head_size == self.hidden_size, \
            f"hidden_size {hidden_size} 必须能被 num_attention_heads {num_attention_heads} 整除"
        # Q/K/V投影层（DeepSeek标准版使用独立的线性层）
        self.q_proj = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size, bias=bias)

        # 投影层
        self.o_proj = nn.Linear(self.all_head_size, hidden_size,bias=bias)



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Optional, Tuple
#
# class DeepSeekMLA(nn.Module):
#     """
#     DeepSeek标准版多头注意力（MLA）模块
#     适配DeepSeek-7B/1.3B等标准版模型的参数设计
#     """
#     def __init__(
#         self,
#         hidden_size: int = 2048,    # DeepSeek标准版隐藏层维度
#         num_attention_heads: int = 16,  # 注意力头数
#         attention_dropout: float = 0.0, # 注意力dropout概率
#         hidden_dropout: float = 0.0,    # 输出dropout概率
#         bias: bool = True              # 线性层是否使用偏置
#     ):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = hidden_size // num_attention_heads  # 每个头的维度
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#         # 确保hidden_size能被注意力头数整除（DeepSeek的核心约束）
#         assert self.all_head_size == self.hidden_size, \
#             f"hidden_size {hidden_size} 必须能被 num_attention_heads {num_attention_heads} 整除"
#
#         # Q/K/V投影层（DeepSeek标准版使用独立的线性层）
#         self.q_proj = nn.Linear(hidden_size, self.all_head_size, bias=bias)
#         self.k_proj = nn.Linear(hidden_size, self.all_head_size, bias=bias)
#         self.v_proj = nn.Linear(hidden_size, self.all_head_size, bias=bias)
#
#         # 输出投影层
#         self.o_proj = nn.Linear(self.all_head_size, hidden_size, bias=bias)
#
#         # Dropout层
#         self.attention_dropout = nn.Dropout(attention_dropout)
#         self.hidden_dropout = nn.Dropout(hidden_dropout)
#
#     def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
#         """
#         将投影后的Q/K/V拆分为多头
#         输入形状: [batch_size, seq_len, all_head_size]
#         输出形状: [batch_size, num_heads, seq_len, head_size]
#         """
#         x = x.view(batch_size, -1, self.num_attention_heads, self.attention_head_size)
#         return x.transpose(1, 2)  # 交换seq_len和num_heads维度
#
#     def _merge_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
#         """
#         将多头结果拼接回原维度
#         输入形状: [batch_size, num_heads, seq_len, head_size]
#         输出形状: [batch_size, seq_len, all_head_size]
#         """
#         x = x.transpose(1, 2)  # 恢复seq_len维度在前
#         return x.contiguous().view(batch_size, -1, self.all_head_size)
#
#     def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
#         """
#         生成因果掩码（下三角矩阵），禁止当前位置关注未来位置
#         输出形状: [1, 1, seq_len, seq_len]
#         """
#         mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
#         mask = mask.masked_fill(mask == 0, float('-inf'))  # 未来位置设为负无穷
#         mask = mask.masked_fill(mask == 1, float(0.0))     # 过去/当前位置设为0
#         return mask
#
#     def forward(
#         self,
#         hidden_states: torch.Tensor,  # 输入: [batch_size, seq_len, hidden_size]
#         attention_mask: Optional[torch.Tensor] = None,  # 可选的padding掩码
#         output_attentions: bool = False  # 是否返回注意力分数
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         batch_size, seq_len, _ = hidden_states.size()
#
#         # 1. 线性投影生成Q/K/V
#         q = self.q_proj(hidden_states)  # [batch_size, seq_len, all_head_size]
#         k = self.k_proj(hidden_states)
#         v = self.v_proj(hidden_states)
#
#         # 2. 拆分多头
#         q = self._split_heads(q, batch_size)  # [batch_size, num_heads, seq_len, head_size]
#         k = self._split_heads(k, batch_size)
#         v = self._split_heads(v, batch_size)
#
#         # 3. 缩放点积注意力计算
#         # 计算Q·K^T / sqrt(head_size)
#         attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, num_heads, seq_len, seq_len]
#         attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, device=attention_scores.device))
#
#         # 4. 应用因果掩码（核心：确保自回归特性）
#         causal_mask = self._create_causal_mask(seq_len, attention_scores.device)
#         attention_scores = attention_scores + causal_mask  # 叠加因果掩码
#
#         # 5. 应用padding掩码（如果有）
#         if attention_mask is not None:
#             # attention_mask形状: [batch_size, 1, 1, seq_len]
#             attention_scores = attention_scores + attention_mask
#
#         # 6. Softmax + Dropout
#         attention_probs = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
#         attention_probs = self.attention_dropout(attention_probs)
#
#         # 7. 加权求和（QK^T * V）
#         context_layer = torch.matmul(attention_probs, v)  # [batch_size, num_heads, seq_len, head_size]
#
#         # 8. 拼接多头结果
#         context_layer = self._merge_heads(context_layer, batch_size)  # [batch_size, seq_len, all_head_size]
#
#         # 9. 输出投影 + Dropout
#         output = self.o_proj(context_layer)  # [batch_size, seq_len, hidden_size]
#         output = self.hidden_dropout(output)
#
#         # 返回输出和可选的注意力分数
#         if output_attentions:
#             return output, attention_probs
#         return output, None
#
# # ------------------- 测试代码 -------------------
# if __name__ == "__main__":
#     # 初始化DeepSeek MLA模块（适配DeepSeek-1.3B参数）
#     mla = DeepSeekMLA(
#         hidden_size=2048,
#         num_attention_heads=16,
#         attention_dropout=0.1,
#         hidden_dropout=0.1
#     )
#
#     # 构造测试输入（batch_size=2, seq_len=10, hidden_size=2048）
#     batch_size, seq_len = 2, 10
#     hidden_states = torch.randn(batch_size, seq_len, 2048)
#
#     # 构造padding掩码（示例：第二个样本仅前5个token有效）
#     padding_mask = torch.ones(batch_size, seq_len)
#     padding_mask[1, 5:] = 0
#     padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
#     padding_mask = padding_mask.masked_fill(padding_mask == 0, float('-inf'))
#
#     # 前向传播
#     output, attn_scores = mla(hidden_states, attention_mask=padding_mask, output_attentions=True)
#
#     # 打印输出形状
#     print(f"输入形状: {hidden_states.shape}")
#     print(f"输出形状: {output.shape}")  # 应与输入一致: [2, 10, 2048]
#     print(f"注意力分数形状: {attn_scores.shape}")  # [2, 16, 10, 10]