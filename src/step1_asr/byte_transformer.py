"""
Problem 1: NPU-Native Text Tokenizer and Byte-Level Transformer
==============================================================
Qualcomm Dragonwing IQ-9075 EVK Deployment
Eliminating CPU Tokenization loops (BPE, UTF-8 decode, Trie/Hash lookups)
by replacing them with a 100% static tensor graph:
  - Input: B in {0..255}^(1 x 512), dtype=int32
  - Embedding: Gather(W_byte[256, D], B) -> [1, 512, D]
  - Byte-Level Transformer: LayerNorm, MultiHeadAttention (MatMul, Softmax, Transpose, Reshape), FFN (GeLU)
  - Output: Logits in [1, 1, 512, 256] (next-byte prediction / language model head)
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import numpy as np
import onnx


class MultiHeadAttention(nn.Module):
    """
    NPU-Native Multi-Head Attention using strictly whitelisted HTP operations:
    MatMul, Softmax, Transpose, Reshape, Mul, Add.
    Zero Conv1d, zero dynamic branching, zero dynamic shapes.
    """
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
        # x: [B, N, D] where B=1, N=512, D=d_model
        B, N, D = x.shape
        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # q: [B, H, N, head_dim], k: [B, H, N, head_dim]
        # Attention scores: (Q @ K.T) * scale
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out_proj(out)


class FFN(nn.Module):
    """
    Feed-Forward Network with GELU activation:
    fc1 (d_model -> d_ff) -> GELU -> fc2 (d_ff -> d_model)
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    Standard Pre-LN Transformer Block:
    x = x + MHA(LN1(x))
    x = x + FFN(LN2(x))
    """
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
    """
    End-to-End Byte-Level Transformer Language Model.
    Directly consumes raw UTF-8 byte stream B in {0..255}^(1 x 512).
    """
    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        seq_len: int = 512,
        vocab_size: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Equation (1): E = Gather(W_byte[256, D], B)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, byte_indices: torch.Tensor) -> torch.Tensor:
        # byte_indices: [1, seq_len] int32
        x = self.embedding(byte_indices) # [1, seq_len, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # [1, seq_len, 256]
        # Format as 4D tensor [1, 1, seq_len, 256] for Qualcomm NPU DLC format
        return logits.unsqueeze(1)


def text_to_byte_tensor(text: str, seq_len: int = 512) -> Tuple[torch.Tensor, List[int]]:
    """
    Converts UTF-8 text string to fixed-shape integer tensor B in {0..255}^(1 x 512).
    Zero-pads or trims to seq_len.
    """
    raw_bytes = list(text.encode("utf-8"))
    if len(raw_bytes) > seq_len:
        trimmed = raw_bytes[:seq_len]
    else:
        trimmed = raw_bytes + [0] * (seq_len - len(raw_bytes))
    tensor = torch.tensor([trimmed], dtype=torch.int32)
    return tensor, raw_bytes


def export_model_to_onnx(model: nn.Module, output_path: str, seq_len: int = 512) -> str:
    """
    Exports ByteTransformer model to ONNX using opset 17.
    """
    model.eval()
    dummy_input = torch.zeros((1, seq_len), dtype=torch.int32)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["byte_indices"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False
    )
    
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Exported clean ONNX model to {output_path}")
    return output_path


def get_tiny_model() -> ByteTransformer:
    """Tiny D=128, L=2, H=4, D_FF=512 (~0.46M params)"""
    return ByteTransformer(d_model=128, n_layers=2, n_heads=4, d_ff=512)


def get_big_model() -> ByteTransformer:
    """Big D=512, L=8, H=8, D_FF=2048 (~25.45M params)"""
    return ByteTransformer(d_model=512, n_layers=8, n_heads=8, d_ff=2048)


if __name__ == "__main__":
    tiny = get_tiny_model()
    p_tiny = sum(p.numel() for p in tiny.parameters())
    print(f"Tiny Model params: {p_tiny:,}")
    export_model_to_onnx(tiny, "outputs/byte_transformer_tiny.onnx")

    big = get_big_model()
    p_big = sum(p.numel() for p in big.parameters())
    print(f"Big Model params: {p_big:,} ({p_big/1e6:.2f}M)")
    export_model_to_onnx(big, "outputs/byte_transformer_big.onnx")
