#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Quantized Audio Baseline (MeloTTS-ZH W8A16 Recipe)
Mô phỏng công thức Lượng tử hóa W8A16 Vocoder & UINT16 Flow của Qualcomm AI Hub (metadata.json).
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
import soundfile as sf
import numpy as np

from melo.api import TTS


def quantize_int8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Symmetric per-tensor INT8 quantization (Trọng số 8-bit W8).
    Scale = max(|W|) / 127, dải giá trị [-128, 127].
    """
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = max_val / 127.0
    q = torch.clamp(torch.round(tensor / scale), -128, 127)
    return q * scale


def quantize_uint16_activation(audio_arr: np.ndarray, scale: float = 1.3067196960037109e-05, zero_point: int = 35955) -> np.ndarray:
    """
    Asymmetric UINT16 quantization (Dòng kích hoạt sóng âm theo metadata.json của Qualcomm).
    Q = clamp(round(Audio / scale) + zero_point, 0, 65535)
    Audio_quant = (Q - zero_point) * scale
    """
    q = np.clip(np.round(audio_arr / scale) + zero_point, 0, 65535).astype(np.float32)
    dequant = (q - zero_point) * scale
    return dequant.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Sinh file audio Quantized W8A16 chuẩn Qualcomm")
    parser.add_argument("--num-samples", type=int, default=5, help="Số câu cần sinh (mặc định: 5 câu)")
    parser.add_argument("--output-dir", type=str, default="output_eval/wav_quant", help="Thư mục lưu file audio quantized")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_file = Path("eval_dataset/baker_500_eval.txt")
    if not test_file.exists():
        print(f"❌ Không tìm thấy file test: {test_file}")
        sys.exit(1)

    with open(test_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split("|") for line in f if "|" in line]

    test_cases = lines[:args.num_samples]

    print("[*] Đang nạp mô hình MeloTTS gốc...")
    model = TTS(language="ZH", device="cpu")
    speaker_id = list(model.hps.data.spk2id.values())[0]

    print("[*] Áp dụng công thức Lượng tử hóa W8A16 chuẩn Qualcomm (metadata.json)...")
    # 1. Lượng tử hóa INT8 toàn bộ ma trận trọng số trong khối Vocoder HiFi-GAN (decoder.bin)
    quantized_layers = 0
    for name, param in model.model.dec.named_parameters():
        if "weight" in name and param.dim() >= 2:
            param.data = quantize_int8(param.data)
            quantized_layers += 1
    print(f"[+] Đã lượng tử hóa INT8 cho {quantized_layers} layer trọng số trong Vocoder.")

    # 2. Lượng tử hóa INT8 cho các tầng nơ-ron trong Flow (flow.bin)
    flow_layers = 0
    for name, param in model.model.flow.named_parameters():
        if "weight" in name and param.dim() >= 2:
            param.data = quantize_int8(param.data)
            flow_layers += 1
    print(f"[+] Đã lượng tử hóa INT8 cho {flow_layers} layer trọng số trong Normalizing Flow.")

    print(f"\n[*] Đang bắt đầu sinh {len(test_cases)} file âm thanh Quantized (W8A16)...")
    total_time = 0.0

    for idx, (item_id, text) in enumerate(test_cases, 1):
        wav_path = out_dir / f"{item_id}_quant.wav"
        t0 = time.perf_counter()

        # Tổng hợp âm thanh qua mạng nơ-ron đã nén W8
        model.tts_to_file(text, speaker_id, str(wav_path), speed=1.0)

        # Áp dụng lượng tử hóa dòng kích hoạt UINT16 trên sóng âm thực tế
        data, sr = sf.read(str(wav_path))
        data_quant = quantize_uint16_activation(data)
        sf.write(str(wav_path), data_quant, sr)

        dur_ms = (time.perf_counter() - t0) * 1000
        total_time += dur_ms
        print(f"[{idx}/{len(test_cases)}] ✓ Đã sinh {item_id}_quant.wav: {dur_ms:.1f} ms | Text: {text}")

    avg_ms = total_time / len(test_cases)
    print("\n" + "=" * 70)
    print(f"🎉 HOÀN THÀNH SINH {len(test_cases)} FILE AUDIO QUANTIZED W8A16")
    print(f"📁 Thư mục lưu trữ: {out_dir.resolve()}")
    print(f"⏱️ Thời gian trung bình: {avg_ms:.1f} ms / câu")
    print("=" * 70)


if __name__ == "__main__":
    main()
