import torch
import torch.nn as nn
import onnx
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / (self.head_dim ** 0.5)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        B, N, D = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # q: [B, H, N, head_dim], k: [B, H, N, head_dim]
        # (q @ k.T) * scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)

class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ByteTransformer(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4, d_ff: int = 512, seq_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, 256, bias=False)

    def forward(self, byte_indices: torch.Tensor) -> torch.Tensor:
        # byte_indices: [1, seq_len] int32
        x = self.embedding(byte_indices) # [1, seq_len, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # [1, seq_len, 256]
        # Reshape to 4D [1, 1, seq_len, 256] for Qualcomm NPU native format
        return logits.unsqueeze(1)

if __name__ == "__main__":
    # Test Tiny model
    tiny = ByteTransformer(d_model=128, n_layers=2, n_heads=4, d_ff=512)
    p_tiny = sum(p.numel() for p in tiny.parameters())
    print(f"Tiny parameters: {p_tiny:,}")

    # Test Big model
    big = ByteTransformer(d_model=512, n_layers=8, n_heads=8, d_ff=2048)
    p_big = sum(p.numel() for p in big.parameters())
    print(f"Big parameters: {p_big:,} ({p_big/1e6:.2f}M)")

    # Test forward with "你好Hi"
    text = "你好Hi"
    raw_bytes = list(text.encode("utf-8"))
    print(f"UTF-8 bytes of '{text}': {raw_bytes}")
    padded_bytes = raw_bytes + [0] * (512 - len(raw_bytes))
    input_tensor = torch.tensor([padded_bytes], dtype=torch.int32)
    
    with torch.no_grad():
        out_tiny = tiny(input_tensor)
        print(f"Tiny output shape: {out_tiny.shape}")
        out_big = big(input_tensor)
        print(f"Big output shape: {out_big.shape}")
