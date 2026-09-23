#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantization, Compilation, Profiling and Discrepancy Evaluation for encoder.onnx
Recipe: W8A16 (Weights INT8, Activations INT16)
Target Device: Qualcomm Dragonwing IQ-9075 EVK (QCS9075 / HTP v73)
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort
import qai_hub as hub

def compute_metrics(ref: np.ndarray, deg: np.ndarray):
    ref_flat = ref.flatten().astype(np.float64)
    deg_flat = deg.flatten().astype(np.float64)

    mse = float(np.mean((ref_flat - deg_flat) ** 2))
    mae = float(np.max(np.abs(ref_flat - deg_flat)))

    norm_ref = np.linalg.norm(ref_flat)
    norm_deg = np.linalg.norm(deg_flat)
    cos_sim = float(np.dot(ref_flat, deg_flat) / (norm_ref * norm_deg + 1e-12))

    signal_power = np.sum(ref_flat ** 2)
    noise_power = np.sum((ref_flat - deg_flat) ** 2)
    snr = float(10.0 * np.log10(signal_power / (noise_power + 1e-12)))

    return {
        "cosine_similarity": cos_sim,
        "snr_db": snr,
        "mse": mse,
        "max_abs_diff": mae
    }

def main():
    print("=" * 65)
    print("🌟 QUALCOMM AI HUB: ENCODER (W8A16) FOR DRAGONWING IQ-9075 EVK")
    print("=" * 65)

    device = hub.Device("Dragonwing IQ-9075 EVK")
    print(f"[+] Target hardware: {device.name}")

    # 1. Load calibration data
    calib_file = Path("calibration_data/encoder_calib.npz")
    if not calib_file.exists():
        print(f"[!] File not found: {calib_file}")
        sys.exit(1)

    calib_data = np.load(calib_file)
    n = min(30, len(calib_data["sid"]))
    dataset_dict = {
        "sid": [calib_data["sid"][i] for i in range(n)],
        "bert": [calib_data["bert"][i] for i in range(n)],
        "ja_bert": [calib_data["ja_bert"][i] for i in range(n)],
        "x": [calib_data["x"][i] for i in range(n)],
        "tone": [calib_data["tone"][i] for i in range(n)],
        "language": [calib_data["language"][i] for i in range(n)],
        "x_lengths": [calib_data["x_lengths"][i] for i in range(n)],
        "noise_scale_w": [calib_data["noise_scale_w"][i] for i in range(n)],
        "sdp_ratio": [calib_data["sdp_ratio"][i] for i in range(n)],
        "length_scale": [calib_data["length_scale"][i] for i in range(n)],
    }
    print(f"[+] Loaded {n} calibration samples for Encoder.")

    # 2. Upload model
    model_path = Path("onnx_models/encoder.onnx")
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"[*] Uploading {model_path} ({size_mb:.1f} MB) to AI Hub...")
    uploaded_model = hub.upload_model(str(model_path), name="MeloTTS_ZH_Encoder_FP32")
    print(f"[✓] Uploaded Model ID: {uploaded_model.model_id}")

    # 3. Submit Quantize Job (W8A16: weights_dtype=INT8, activations_dtype=INT16)
    print("[*] Submitting W8A16 Quantize Job (weights_dtype=INT8, activations_dtype=INT16)...")
    quant_job = hub.submit_quantize_job(
        model=uploaded_model,
        calibration_data=dataset_dict,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="Encoder_Quantize_w8a16"
    )
    print(f"[+] Quantize Job ID: {quant_job.job_id}")
    print("[*] Waiting for AIMET W8A16 quantization to finish...")
    quant_job.wait()
    quantized_model = quant_job.get_target_model()
    print(f"[✓] Quantization complete! Quantized Model ID: {quantized_model.model_id}")

    # 4. Download quantized model
    out_dir = Path("quantized_models/encoder_w8a16")
    out_dir.mkdir(parents=True, exist_ok=True)
    q_zip = Path("quantized_models/encoder_w8a16.zip")
    quantized_model.download(str(q_zip))
    print(f"[✓] Downloaded quantized model zip to: {q_zip}")

    # 5. Local QDQ ONNX evaluation vs FP32
    print("[*] Unzipping and evaluating local QDQ ONNX model...")
    # Find actual downloaded zip
    possible_zips = list(Path("quantized_models").glob("encoder_w8a16*.zip"))
    actual_zip = possible_zips[0] if possible_zips else q_zip
    os.system(f"unzip -q -o '{actual_zip}' -d '{out_dir}'")
    
    onnx_files = list(out_dir.rglob("*.onnx"))
    test_in = {
        "sid": dataset_dict["sid"][0],
        "bert": dataset_dict["bert"][0],
        "ja_bert": dataset_dict["ja_bert"][0],
        "x": dataset_dict["x"][0],
        "tone": dataset_dict["tone"][0],
        "language": dataset_dict["language"][0],
        "x_lengths": dataset_dict["x_lengths"][0],
        "noise_scale_w": dataset_dict["noise_scale_w"][0],
        "sdp_ratio": dataset_dict["sdp_ratio"][0],
        "length_scale": dataset_dict["length_scale"][0]
    }
    
    sess_fp32 = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    fp32_outs = sess_fp32.run(None, test_in)
    fp32_m_p, fp32_logs_p, fp32_w_ceil = fp32_outs[0], fp32_outs[1], fp32_outs[2]

    if onnx_files:
        qdq_path = onnx_files[0]
        print(f"[+] Found QDQ ONNX: {qdq_path}")
        sess_qdq = ort.InferenceSession(str(qdq_path), providers=["CPUExecutionProvider"])
        qdq_outs = sess_qdq.run(None, test_in)
        qdq_m_p, qdq_logs_p, qdq_w_ceil = qdq_outs[0], qdq_outs[1], qdq_outs[2]

        m_p_metrics = compute_metrics(fp32_m_p, qdq_m_p)
        logs_p_metrics = compute_metrics(fp32_logs_p, qdq_logs_p)
        w_ceil_metrics = compute_metrics(fp32_w_ceil, qdq_w_ceil)

        print("\n" + "-" * 50)
        print("📊 LOCAL QDQ (W8A16) vs FP32 METRICS FOR ENCODER:")
        print(f"• m_p Cosine Sim      : {m_p_metrics['cosine_similarity']:.6f} | SNR: {m_p_metrics['snr_db']:.2f} dB")
        print(f"• logs_p Cosine Sim   : {logs_p_metrics['cosine_similarity']:.6f} | SNR: {logs_p_metrics['snr_db']:.2f} dB")
        print(f"• w_ceil Cosine Sim   : {w_ceil_metrics['cosine_similarity']:.6f} | SNR: {w_ceil_metrics['snr_db']:.2f} dB")
        print("-" * 50 + "\n")

    # 6. Compile for Dragonwing IQ-9075 EVK
    print(f"[*] Submitting Compile Job for {device.name} (options='--target_runtime precompiled_qnn_onnx')...")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        options="--target_runtime precompiled_qnn_onnx",
        name="Encoder_Compile_precompiled_qnn_onnx_IQ9075"
    )
    print(f"[+] Compile Job ID: {compile_job.job_id}")
    print("[*] Waiting for compilation to finish...")
    compile_job.wait()
    compiled_model = compile_job.get_target_model()
    print(f"[✓] Compilation complete! Target Model ID: {compiled_model.model_id}")

    # 7. Submit Profile & Inference Jobs
    print(f"[*] Submitting Profile Job on {device.name}...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="Encoder_Profile_IQ9075"
    )
    print(f"[+] Profile Job ID: {profile_job.job_id}")

    print(f"[*] Submitting Inference Job on {device.name}...")
    infer_inputs = {
        k: [v[0]] for k, v in dataset_dict.items()
    }
    infer_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=infer_inputs,
        name="Encoder_Inference_IQ9075"
    )
    print(f"[+] Inference Job ID: {infer_job.job_id}")

    print("[*] Waiting for on-device jobs...")
    profile_job.wait()
    infer_job.wait()

    profile_res = profile_job.download_profile()
    lat_us = profile_res.get("execution_summary", {}).get("estimated_inference_time", "N/A")
    peak_mem = profile_res.get("execution_summary", {}).get("estimated_inference_peak_memory", "N/A")
    infer_out = infer_job.download_output_data()

    print("\n" + "=" * 65)
    print("🏆 FINAL ON-DEVICE RESULTS FOR ENCODER (W8A16 on IQ-9075 EVK):")
    if isinstance(lat_us, (int, float)):
        print(f"• Latency on NPU    : {lat_us / 1000.0:.2f} ms")
    if isinstance(peak_mem, (int, float)):
        print(f"• Peak NPU Memory   : {peak_mem / (1024*1024):.2f} MB")
    print("=" * 65)

    # 8. Download compiled model
    c_out = Path("quantized_models/encoder_compiled_iq9075.onnx")
    compiled_model.download(str(c_out))
    print(f"[✓] Downloaded compiled encoder model to {c_out}")

if __name__ == "__main__":
    main()
