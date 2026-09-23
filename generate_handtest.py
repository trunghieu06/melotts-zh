#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Hand-test Audio for MeloTTS-ZH
Synthesizes a custom Chinese sentence using both:
1. FP32 Baseline
2. Quantized W8A16 Model (Vocoder INT8 + Flow INT8 + UINT16 Audio Activation)
Saves outputs to directory: handtest/
"""

import os
import sys
import time
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from melo.api import TTS


def quantize_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Symmetric per-tensor INT8 quantization."""
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = max_val / 127.0
    q = torch.clamp(torch.round(tensor / scale), -128, 127)
    return q * scale


def quantize_uint16_activation(
    audio_arr: np.ndarray,
    scale: float = 1.3067196960037109e-05,
    zero_point: int = 35955
) -> np.ndarray:
    """Asymmetric UINT16 quantization (Qualcomm AI Hub metadata.json)."""
    q = np.clip(np.round(audio_arr / scale) + zero_point, 0, 65535).astype(np.float32)
    dequant = (q - zero_point) * scale
    return dequant.astype(np.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate handtest audio for MeloTTS-ZH")
    parser.add_argument("text_pos", nargs="?", default=None, help="Văn bản tiếng Trung")
    parser.add_argument("--text", type=str, default=None, help="Văn bản tiếng Trung")
    parser.add_argument("--name", type=str, default="work", help="Tên file tiền tố (mặc định: work)")
    args = parser.parse_args()

    text = args.text or args.text_pos or "我上班也没那么容易呢。每天还要看客户的脸色，还要看老板和同事的脸色，也累死我了。"
    prefix = args.name

    output_dir = Path("handtest")
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / f"{prefix}_fp32.wav"
    quant_path = output_dir / f"{prefix}_quant.wav"

    seed = 42

    print("=" * 70)
    print(f"🎙️ BẮT ĐẦU TỔNG HỢP GIỌNG ĐỌC THỬ NGHIỆM (HANDTEST)")
    print(f"📝 Câu văn bản: \"{text}\"")
    print(f"📁 Thư mục xuất: {output_dir.resolve()}")
    print("=" * 70)

    # 1. Sinh Audio FP32
    print("\n[1/2] Đang nạp mô hình FP32 và tổng hợp giọng đọc...")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model_fp32 = TTS(language="ZH", device="cpu")
    speaker_id = list(model_fp32.hps.data.spk2id.values())[0]

    t0 = time.perf_counter()
    model_fp32.tts_to_file(text, speaker_id, str(fp32_path), speed=1.0)
    time_fp32 = (time.perf_counter() - t0) * 1000
    print(f"✓ Đã tạo file FP32: {fp32_path} ({time_fp32:.1f} ms)")

    # 2. Sinh Audio Quantized W8A16
    print("\n[2/2] Đang áp dụng công thức Lượng tử hóa W8A16 (Qualcomm AI Hub)...")
    # Lượng tử hóa trọng số INT8 trên Vocoder
    dec_layers = 0
    for name, param in model_fp32.model.dec.named_parameters():
        if "weight" in name and param.dim() >= 2:
            param.data = quantize_int8(param.data)
            dec_layers += 1

    # Lượng tử hóa trọng số INT8 trên Flow
    flow_layers = 0
    for name, param in model_fp32.model.flow.named_parameters():
        if "weight" in name and param.dim() >= 2:
            param.data = quantize_int8(param.data)
            flow_layers += 1

    print(f"[+] Đã lượng tử hóa {dec_layers} layers Vocoder & {flow_layers} layers Flow.")

    # Đặt lại seed để giữ nhịp điệu và ngữ điệu đồng nhất
    torch.manual_seed(seed)
    np.random.seed(seed)

    t1 = time.perf_counter()
    model_fp32.tts_to_file(text, speaker_id, str(quant_path), speed=1.0)

    # Lượng tử hóa dòng sóng âm UINT16
    audio_data, sr = sf.read(str(quant_path))
    audio_quant = quantize_uint16_activation(audio_data)
    sf.write(str(quant_path), audio_quant, sr)
    time_quant = (time.perf_counter() - t1) * 1000
    print(f"✓ Đã tạo file Quantized: {quant_path} ({time_quant:.1f} ms)")

    # 3. Đo lường so sánh chất lượng âm thanh thực tế
    print("\n" + "=" * 70)
    print("📊 THỐNG KÊ CHI TIẾT 2 FILE ÂM THANH:")
    info_fp32 = sf.info(str(fp32_path))
    info_quant = sf.info(str(quant_path))

    print(f"- File FP32     : {fp32_path.name} | Thời lượng: {info_fp32.duration:.2f}s | Sample rate: {info_fp32.samplerate} Hz | Dung lượng: {os.path.getsize(fp32_path)/1024:.1f} KB")
    print(f"- File Quantized: {quant_path.name} | Thời lượng: {info_quant.duration:.2f}s | Sample rate: {info_quant.samplerate} Hz | Dung lượng: {os.path.getsize(quant_path)/1024:.1f} KB")

    y_ref, _ = librosa.load(str(fp32_path), sr=24000)
    y_deg, _ = librosa.load(str(quant_path), sr=24000)

    # Cosine Similarity trên Mel-Spectrogram
    min_len = min(len(y_ref), len(y_deg))
    mel_ref = librosa.feature.melspectrogram(y=y_ref[:min_len], sr=24000).flatten()
    mel_deg = librosa.feature.melspectrogram(y=y_deg[:min_len], sr=24000).flatten()
    cos_sim = np.dot(mel_ref, mel_deg) / (np.linalg.norm(mel_ref) * np.linalg.norm(mel_deg))

    # FastDTW Aligned Cosine Sim
    mel_2d_ref = librosa.feature.melspectrogram(y=y_ref, sr=24000)
    mel_2d_deg = librosa.feature.melspectrogram(y=y_deg, sr=24000)
    _, path = fastdtw(mel_2d_ref.T, mel_2d_deg.T, dist=euclidean)
    p_ref = [p[0] for p in path]
    p_deg = [p[1] for p in path]
    m1_aligned = mel_2d_ref[:, p_ref].flatten()
    m2_aligned = mel_2d_deg[:, p_deg].flatten()
    cos_aligned = np.dot(m1_aligned, m2_aligned) / (np.linalg.norm(m1_aligned) * np.linalg.norm(m2_aligned))

    print(f"- Cosine Similarity (Trực tiếp)  : {cos_sim:.4f}")
    print(f"- Cosine Similarity (DTW Aligned): {cos_aligned:.4f}")
    print("=" * 70)
    print("🎉 Hoàn tất sinh file handtest thành công!")


if __name__ == "__main__":
    main()
