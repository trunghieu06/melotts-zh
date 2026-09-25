#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Problem 3: In-NPU Batching & Sliding-Window Chunking
Reference: MeloTTS.pdf (Section 4)

Replaces CPU `for start_idx in range(...)` slicing loop with static In-NPU tensor reshaping:
  z: [1, 192, 1536] 
  -> Reshape [1, 192, 24, 64]
  -> Permute [0, 2, 1, 3] -> [1, 24, 192, 64]
  -> Reshape [24, 192, 64] (Batched for Vocoder)
  
After Vocoder generates [24, 1, 32768]:
  -> Reshape [1, 1, 24 * 32768 = 786432]

HTP Whitelist Ops: Reshape, Transpose, Conv2d.
0% Retrain, 100% NPU Native on Qualcomm Hexagon HTP v73.
"""

import torch
import torch.nn as nn


class NPUChunkBatcher(nn.Module):
    def __init__(self, channels: int = 192, total_frames: int = 1536, chunk_size: int = 64):
        super().__init__()
        self.channels = channels
        self.total_frames = total_frames
        self.chunk_size = chunk_size
        self.num_chunks = total_frames // chunk_size  # e.g., 1536 // 64 = 24

    def batch_chunks(self, z: torch.Tensor, g: torch.Tensor = None):
        """
        Chuyển Mel-Latent [1, 192, 1536] thành Batch [24, 192, 64] cho NPU Vocoder.
        """
        # 1. Reshape & Transpose sang dạng Batch
        # [1, 192, 1536] -> [1, 192, 24, 64]
        z_4d = z.view(1, self.channels, self.num_chunks, self.chunk_size)
        # [1, 192, 24, 64] -> [24, 192, 64]
        z_batched = z_4d.permute(0, 2, 1, 3).reshape(self.num_chunks, self.channels, self.chunk_size)

        if g is not None:
            # g: [1, 256, 1] -> [24, 256, 1]
            g_batched = g.repeat(self.num_chunks, 1, 1)
            return z_batched, g_batched

        return z_batched

    def unbatch_audio(self, audio_chunks: torch.Tensor, samples_per_chunk: int = 32768) -> torch.Tensor:
        """
        Ghép mảng sóng âm sau khi NPU Vocoder xử lý xong:
        [24, 1, 32768] -> [1, 1, 786432]
        """
        # [24, 1, 32768] -> [1, 1, 24 * 32768]
        total_samples = self.num_chunks * samples_per_chunk
        audio_flat = audio_chunks.view(1, 1, total_samples)
        return audio_flat


if __name__ == "__main__":
    print("=" * 60)
    print("Testing NPUChunkBatcher vs CPU Slicing Loop...")
    print("=" * 60)

    channels = 192
    total_frames = 1536
    chunk_size = 64
    num_chunks = total_frames // chunk_size  # 24

    batcher = NPUChunkBatcher(channels=channels, total_frames=total_frames, chunk_size=chunk_size)

    # Dữ liệu giả lập
    z = torch.randn(1, channels, total_frames)
    g = torch.randn(1, 256, 1)

    # 1. Chạy NPU Batching
    z_batched, g_batched = batcher.batch_chunks(z, g)
    print(f"z_batched shape: {z_batched.shape} (Expected: [{num_chunks}, {channels}, {chunk_size}])")
    print(f"g_batched shape: {g_batched.shape} (Expected: [{num_chunks}, 256, 1])")

    # 2. Đối chiếu với CPU loop
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        cpu_slice = z[0, :, start:end]
        assert torch.allclose(z_batched[i], cpu_slice), f"Mismatch at chunk {i}"

    print("✅ Input slicing parity PASSED: 100% exact match!")

    # 3. Test ghép âm thanh
    fake_audio_chunks = torch.randn(num_chunks, 1, 32768)
    audio_full = batcher.unbatch_audio(fake_audio_chunks)
    print(f"audio_full shape: {audio_full.shape} (Expected: [1, 1, {num_chunks * 32768}])")

    # Kiểm tra ghép từng chunk
    for i in range(num_chunks):
        start = i * 32768
        end = start + 32768
        assert torch.allclose(audio_full[0, 0, start:end], fake_audio_chunks[i, 0]), f"Audio mismatch at chunk {i}"

    print("✅ Output audio re-assembly parity PASSED: 100% exact match!")
