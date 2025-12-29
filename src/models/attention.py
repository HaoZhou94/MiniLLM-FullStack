





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