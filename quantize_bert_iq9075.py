#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Automated Pipeline for bert_wrapper on Qualcomm AI Hub
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
    print("🌟 QUALCOMM AI HUB: BERT_WRAPPER (INT8 w8a8) FOR IQ-9075 EVK")
    print("=" * 65)

    device = hub.Device("Dragonwing IQ-9075 EVK")
    print(f"[+] Target hardware: {device.name}")

    # 1. Load calibration data
    calib_file = Path("calibration_data/bert_calib.npz")
    if not calib_file.exists():
        print(f"[!] File not found: {calib_file}")
        sys.exit(1)

    calib_data = np.load(calib_file)
    n = min(30, len(calib_data["input_ids"]))
    dataset_dict = {
        "input_ids": [calib_data["input_ids"][i] for i in range(n)],
        "token_type_ids": [calib_data["token_type_ids"][i] for i in range(n)],
        "attention_mask": [calib_data["attention_mask"][i] for i in range(n)],
    }
    print(f"[+] Loaded {n} Chinese Baker calibration samples.")

    # 2. Upload model
    model_path = Path("onnx_models/bert_wrapper.onnx")
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"[*] Uploading {model_path} ({size_mb:.1f} MB) to AI Hub...")
    uploaded_model = hub.upload_model(str(model_path), name="MeloTTS_ZH_BERT_FP32")
    print(f"[✓] Uploaded Model ID: {uploaded_model.model_id}")

    # 3. Submit Quantize Job (INT8: w8a8)
    print("[*] Submitting INT8 Quantize Job (weights_dtype=INT8, activations_dtype=INT8)...")
    quant_job = hub.submit_quantize_job(
        model=uploaded_model,
        calibration_data=dataset_dict,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name="BERT_Quantize_INT8"
    )
    print(f"[+] Quantize Job ID: {quant_job.job_id}")
    print("[*] Waiting for AIMET INT8 quantization to finish...")
    quant_job.wait()
    quantized_model = quant_job.get_target_model()
    print(f"[✓] Quantization complete! Quantized Model ID: {quantized_model.model_id}")

    # 4. Download quantized model
    out_dir = Path("quantized_models/bert_wrapper_int8")
    out_dir.mkdir(parents=True, exist_ok=True)
    q_zip = Path("quantized_models/bert_wrapper_int8.zip")
    quantized_model.download(str(q_zip))
    print(f"[✓] Downloaded quantized model zip to: {q_zip}")

    # 5. Local QDQ ONNX evaluation vs FP32
    print("[*] Unzipping and evaluating local QDQ ONNX model...")
    os.system(f"unzip -q -o {q_zip} -d {out_dir}")
    
    # Find onnx in out_dir
    onnx_files = list(out_dir.rglob("*.onnx"))
    if onnx_files:
        qdq_path = onnx_files[0]
        print(f"[+] Found QDQ ONNX: {qdq_path}")
        test_in = {
            "input_ids": dataset_dict["input_ids"][0],
            "token_type_ids": dataset_dict["token_type_ids"][0],
            "attention_mask": dataset_dict["attention_mask"][0]
        }
        
        sess_fp32 = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        fp32_out = sess_fp32.run(None, test_in)[0]

        sess_qdq = ort.InferenceSession(str(qdq_path), providers=["CPUExecutionProvider"])
        qdq_out = sess_qdq.run(None, test_in)[0]

        local_metrics = compute_metrics(fp32_out, qdq_out)
        print("\n" + "-" * 50)
        print("📊 LOCAL QDQ (INT8) vs FP32 METRICS:")
        print(f"• Cosine Similarity : {local_metrics['cosine_similarity']:.6f} (Target: > 0.9850)")
        print(f"• SNR (dB)          : {local_metrics['snr_db']:.2f} dB (Target: > 20 dB)")
        print(f"• MSE               : {local_metrics['mse']:.6e}")
        print("-" * 50 + "\n")

    # 6. Compile for Dragonwing IQ-9075 EVK
    print(f"[*] Submitting Compile Job for {device.name} (options='--target_runtime precompiled_qnn_onnx')...")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        options="--target_runtime precompiled_qnn_onnx",
        name="BERT_Compile_precompiled_qnn_onnx_IQ9075"
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
        name="BERT_Profile_IQ9075"
    )
    print(f"[+] Profile Job ID: {profile_job.job_id}")

    print(f"[*] Submitting Inference Job on {device.name}...")
    infer_inputs = {
        "input_ids": [dataset_dict["input_ids"][0]],
        "token_type_ids": [dataset_dict["token_type_ids"][0]],
        "attention_mask": [dataset_dict["attention_mask"][0]]
    }
    infer_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=infer_inputs,
        name="BERT_Inference_IQ9075"
    )
    print(f"[+] Inference Job ID: {infer_job.job_id}")

    print("[*] Waiting for on-device jobs...")
    profile_job.wait()
    infer_job.wait()

    profile_res = profile_job.download_profile()
    lat_us = profile_res.get("execution_summary", {}).get("estimated_inference_time", "N/A")
    infer_out = infer_job.download_output_data()
    quant_hidden = list(infer_out.values())[0][0]

    ondevice_metrics = compute_metrics(fp32_out, quant_hidden)
    print("\n" + "=" * 65)
    print("🏆 FINAL ON-DEVICE RESULTS FOR BERT_WRAPPER (IQ-9075 EVK):")
    print(f"• Cosine Similarity : {ondevice_metrics['cosine_similarity']:.6f} (Target: > 0.9850)")
    print(f"• SNR (dB)          : {ondevice_metrics['snr_db']:.2f} dB (Target: > 20 dB)")
    print(f"• MSE               : {ondevice_metrics['mse']:.6e}")
    if isinstance(lat_us, (int, float)):
        print(f"• Latency on NPU    : {lat_us / 1000.0:.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    main()
