#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export NPU-Native Modules to ONNX (Problems 2, 3, 4)
Target Device: Qualcomm Dragonwing IQ-9075 EVK (Hexagon HTP v73)
"""

import os
from pathlib import Path
import torch
import onnx
import onnxruntime as ort
import numpy as np

from npu_engine.duration_expansion import NPUDurationExpansion
from npu_engine.trimming import NPUArtifactTrimmer

def export_duration_expansion():
    print("[*] Exporting Problem 2: NPUDurationExpansion to ONNX...")
    model = NPUDurationExpansion(max_phonemes=512, max_frames=1536)
    model.eval()

    dummy_w_ceil = torch.ones(1, 1, 512, dtype=torch.float32)
    out_path = Path("onnx_models/npu_duration_expansion.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_w_ceil,
        str(out_path),
        input_names=["w_ceil"],
        output_names=["attn_squeezed"],
        opset_version=17,
        do_constant_folding=True
    )

    # Lưu dạng self-contained (không dùng external data .data)
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.save(onnx_model, str(out_path), save_as_external_data=False)
    data_file = out_path.with_name(out_path.name + ".data")
    if data_file.exists():
        data_file.unlink()
    print(f"  ✓ Exported self-contained: {out_path} ({out_path.stat().st_size / 1024:.2f} KB)")

    # Test ONNX Runtime
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    w_input = np.zeros((1, 1, 512), dtype=np.float32)
    w_input[0, 0, :4] = [2.0, 3.0, 1.0, 4.0]
    (out_attn,) = sess.run(None, {"w_ceil": w_input})
    print(f"  ✓ ORT verification: attn_squeezed shape = {out_attn.shape}")
    return str(out_path)

def export_artifact_trimmer():
    print("\n[*] Exporting Problem 4: NPUArtifactTrimmer to ONNX...")
    total_samples = 24 * 32768  # 786432
    model = NPUArtifactTrimmer(total_samples=total_samples, hop_size=512)
    model.eval()

    dummy_audio = torch.zeros(1, 1, total_samples, dtype=torch.float32)
    dummy_y_len = torch.tensor([150.0], dtype=torch.float32)
    out_path = Path("onnx_models/npu_artifact_trimmer.onnx")

    torch.onnx.export(
        model,
        (dummy_audio, dummy_y_len),
        str(out_path),
        input_names=["audio", "y_lengths"],
        output_names=["audio_clean"],
        opset_version=17,
        do_constant_folding=True
    )

    # Lưu dạng self-contained (không dùng external data .data)
    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx.save(onnx_model, str(out_path), save_as_external_data=False)
    data_file = out_path.with_name(out_path.name + ".data")
    if data_file.exists():
        data_file.unlink()
    print(f"  ✓ Exported self-contained: {out_path} ({out_path.stat().st_size / 1024:.2f} KB)")

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    audio_in = np.ones((1, 1, total_samples), dtype=np.float32)
    y_in = np.array([100.0], dtype=np.float32)
    (out_audio,) = sess.run(None, {"audio": audio_in, "y_lengths": y_in})
    valid = int(100 * 512)
    assert np.all(out_audio[0, 0, :valid] == 1.0)
    assert np.all(out_audio[0, 0, valid:] == 0.0)
    print(f"  ✓ ORT verification: audio_clean shape = {out_audio.shape}, trimming verified!")
    return str(out_path)

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 EXPORT CÁC MODULES NPU-NATIVE CHO QUALCOMM HEXAGON NPU")
    print("=" * 70)
    p2 = export_duration_expansion()
    p4 = export_artifact_trimmer()
    print("\n✅ Hoàn tất xuất ONNX cho cả Problem 2 và Problem 4!")
