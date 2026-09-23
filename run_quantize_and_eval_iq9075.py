#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline Tự động hóa Toàn diện trên Qualcomm AI Hub
Mục tiêu Phần cứng: Qualcomm Dragonwing IQ-9075 EVK (QCS9075 / HTP v73)

Quy trình:
1. Nạp dữ liệu hiệu chuẩn từ calibration_data/
2. Lượng tử hóa:
   - decoder.onnx      -> w8a16 (Weights INT8, Activations INT16)
   - bert_wrapper.onnx -> int8 (Weights INT8, Activations INT8)
3. Biên dịch QNN Context Binary cho Dragonwing IQ-9075 EVK
4. Profiling đo độ trễ suy luận (Latency ms) và NPU Memory trên chip IQ-9075 thật
5. Chạy suy luận (Inference Job) đối chiếu sai số (Cosine Sim, SNR, MSE) với bản FP32 gốc
6. Tải và lưu trữ artifacts vào thư mục quantized_models/
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
import numpy as np

try:
    import qai_hub as hub
    import onnxruntime as ort
except ImportError as e:
    print(f"[!] Thiếu thư viện: {e}")
    sys.exit(1)


def compute_metrics(ref: np.ndarray, deg: np.ndarray):
    """Tính các chỉ số sai số kỹ thuật giữa tensor FP32 và Quantized."""
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


def run_pipeline_decoder(device, num_calib=30, output_dir=Path("quantized_models")):
    print("\n" + "=" * 65)
    print("🚀 [1/2] LƯỢNG TỬ HÓA & KIỂM ĐỊNH DECODER (VOCODER - w8a16)")
    print(f"• Thiết bị mục tiêu: {device.name}")
    print(f"• Quy cách chuẩn   : Weights INT8, Activations INT16 (w8a16)")
    print("=" * 65)

    # 1. Đọc dữ liệu calibration
    calib_file = Path("calibration_data/decoder_calib.npz")
    if not calib_file.exists():
        raise FileNotFoundError("Chưa tìm thấy calibration_data/decoder_calib.npz. Vui lòng chạy prepare_calibration_data.py trước.")
    
    calib_data = np.load(calib_file)
    z_samples = [calib_data["z"][i] for i in range(min(num_calib, len(calib_data["z"])))]
    g_samples = [calib_data["g"][i] for i in range(min(num_calib, len(calib_data["g"])))]
    dataset_dict = {"z": z_samples, "g": g_samples}
    print(f"[+] Đã nạp {len(z_samples)} mẫu calibration thực tế cho Decoder.")

    # 2. Upload model gốc
    model_path = Path("onnx_models/decoder.onnx")
    print(f"[*] Đang tải {model_path} lên Qualcomm AI Hub...")
    uploaded_model = hub.upload_model(str(model_path), name="MeloTTS_ZH_Decoder_FP32")
    print(f"[✓] Model uploaded. Model ID: {uploaded_model.model_id}")

    # 3. Submit Quantize Job (w8a16)
    print("[*] Gửi job Lượng tử hóa w8a16 lên AI Hub...")
    quant_job = hub.submit_quantize_job(
        model=uploaded_model,
        calibration_data=dataset_dict,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
        name="Decoder_Quantize_w8a16"
    )
    print(f"    Job ID: {quant_job.job_id}")
    print("    Đang chờ AI Hub tối ưu hóa trọng số (AIMET)...")
    quant_job.wait()
    quantized_model = quant_job.get_target_model()
    print(f"[✓] Lượng tử hóa thành công! Quantized Model ID: {quantized_model.model_id}")

    # 4. Compile cho Dragonwing IQ-9075 EVK
    print(f"[*] Đang biên dịch QNN Context Binary cho {device.name}...")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        name="Decoder_Compile_IQ9075",
        options="--target_runtime qnn_context_binary"
    )
    print(f"    Job ID: {compile_job.job_id}")
    compile_job.wait()
    compiled_model = compile_job.get_target_model()
    print(f"[✓] Biên dịch hoàn tất! Target Model ID: {compiled_model.model_id}")

    # 5. Profile đo độ trễ trên NPU thật
    print(f"[*] Gửi Profile Job đo hiệu năng thực tế trên {device.name}...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="Decoder_Profile_IQ9075"
    )
    print(f"    Job ID: {profile_job.job_id}")
    profile_job.wait()
    profile_summary = profile_job.download_profile()
    latency_ms = profile_summary.get("execution_summary", {}).get("estimated_inference_time", "N/A")
    print(f"[✓] Profiling hoàn tất! Độ trễ ước tính trên NPU: {latency_ms}")

    # 6. Chạy Inference kiểm tra đối chiếu sai số
    print("[*] Gửi Inference Job chạy thử nghiệm trên NPU Cloud...")
    test_input = {"z": [z_samples[0]], "g": [g_samples[0]]}
    infer_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=test_input,
        name="Decoder_Inference_IQ9075"
    )
    infer_job.wait()
    quant_outputs = infer_job.download_output_data()
    quant_audio = list(quant_outputs.values())[0][0] # [1, 1, 32768]

    # Chạy mô hình FP32 trên host để lấy Ground Truth
    print("[*] Chạy mô hình FP32 đối chứng trên máy (ONNX Runtime)...")
    sess_fp32 = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    fp32_audio = sess_fp32.run(None, {"z": z_samples[0], "g": g_samples[0]})[0]

    # Tính sai số
    metrics = compute_metrics(fp32_audio, quant_audio)
    print("\n" + "-" * 50)
    print("📊 KẾT QUẢ ĐỐI CHIẾU SAI SỐ DECODER (w8a16 vs FP32):")
    print(f"• Cosine Similarity : {metrics['cosine_similarity']:.5f} (Mục tiêu: > 0.9900)")
    print(f"• SNR (Tỷ số tín hiệu): {metrics['snr_db']:.2f} dB (Mục tiêu: > 25 dB)")
    print(f"• Mean Squared Error: {metrics['mse']:.6e}")
    print(f"• Max Absolute Diff : {metrics['max_abs_diff']:.6e}")
    print("-" * 50)

    # Tải model về
    save_path = output_dir / "decoder_w8a16_quantized.onnx"
    quantized_model.download(str(save_path))
    print(f"[✓] Đã lưu model lượng tử hóa về máy: {save_path}")

    return {
        "model": "decoder",
        "precision": "w8a16",
        "metrics": metrics,
        "latency_summary": latency_ms,
        "saved_path": str(save_path)
    }


def run_pipeline_bert(device, num_calib=30, output_dir=Path("quantized_models")):
    print("\n" + "=" * 65)
    print("🚀 [2/2] LƯỢNG TỬ HÓA & KIỂM ĐỊNH BERT_WRAPPER (INT8)")
    print(f"• Thiết bị mục tiêu: {device.name}")
    print(f"• Quy cách chuẩn   : Weights INT8, Activations INT8 (INT8 thuần)")
    print("=" * 65)

    # 1. Đọc dữ liệu calibration
    calib_file = Path("calibration_data/bert_calib.npz")
    if not calib_file.exists():
        raise FileNotFoundError("Chưa tìm thấy calibration_data/bert_calib.npz. Vui lòng chạy prepare_calibration_data.py trước.")
    
    calib_data = np.load(calib_file)
    n = min(num_calib, len(calib_data["input_ids"]))
    input_ids_samples = [calib_data["input_ids"][i] for i in range(n)]
    token_type_samples = [calib_data["token_type_ids"][i] for i in range(n)]
    attn_mask_samples = [calib_data["attention_mask"][i] for i in range(n)]
    
    dataset_dict = {
        "input_ids": input_ids_samples,
        "token_type_ids": token_type_samples,
        "attention_mask": attn_mask_samples
    }
    print(f"[+] Đã nạp {n} mẫu văn bản tiếng Trung chuẩn Baker cho BERT.")

    # 2. Upload model gốc
    model_path = Path("onnx_models/bert_wrapper.onnx")
    print(f"[*] Đang tải {model_path} (~406 MB) lên Qualcomm AI Hub...")
    uploaded_model = hub.upload_model(str(model_path), name="MeloTTS_ZH_BERT_FP32")
    print(f"[✓] Model uploaded. Model ID: {uploaded_model.model_id}")

    # 3. Submit Quantize Job (INT8)
    print("[*] Gửi job Lượng tử hóa INT8 lên AI Hub...")
    quant_job = hub.submit_quantize_job(
        model=uploaded_model,
        calibration_data=dataset_dict,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name="BERT_Quantize_INT8"
    )
    print(f"    Job ID: {quant_job.job_id}")
    print("    Đang chờ AI Hub tối ưu hóa trọng số (AIMET)...")
    quant_job.wait()
    quantized_model = quant_job.get_target_model()
    print(f"[✓] Lượng tử hóa thành công! Quantized Model ID: {quantized_model.model_id}")

    # 4. Compile cho Dragonwing IQ-9075 EVK
    print(f"[*] Đang biên dịch QNN Context Binary cho {device.name}...")
    compile_job = hub.submit_compile_job(
        model=quantized_model,
        device=device,
        name="BERT_Compile_IQ9075",
        options="--target_runtime qnn_context_binary"
    )
    print(f"    Job ID: {compile_job.job_id}")
    compile_job.wait()
    compiled_model = compile_job.get_target_model()
    print(f"[✓] Biên dịch hoàn tất! Target Model ID: {compiled_model.model_id}")

    # 5. Profile đo độ trễ trên NPU
    print(f"[*] Gửi Profile Job đo hiệu năng thực tế trên {device.name}...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="BERT_Profile_IQ9075"
    )
    print(f"    Job ID: {profile_job.job_id}")
    profile_job.wait()
    profile_summary = profile_job.download_profile()
    latency_ms = profile_summary.get("execution_summary", {}).get("estimated_inference_time", "N/A")
    print(f"[✓] Profiling hoàn tất! Độ trễ ước tính trên NPU: {latency_ms}")

    # 6. Chạy Inference kiểm tra đối chiếu sai số
    print("[*] Gửi Inference Job chạy thử nghiệm trên NPU Cloud...")
    test_input = {
        "input_ids": [input_ids_samples[0]],
        "token_type_ids": [token_type_samples[0]],
        "attention_mask": [attn_mask_samples[0]]
    }
    infer_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=test_input,
        name="BERT_Inference_IQ9075"
    )
    infer_job.wait()
    quant_outputs = infer_job.download_output_data()
    quant_hidden = list(quant_outputs.values())[0][0] # [1, 200, 768]

    # Chạy mô hình FP32 trên host
    print("[*] Chạy mô hình FP32 đối chứng trên máy (ONNX Runtime)...")
    sess_fp32 = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    fp32_hidden = sess_fp32.run(None, {
        "input_ids": input_ids_samples[0],
        "token_type_ids": token_type_samples[0],
        "attention_mask": attn_mask_samples[0]
    })[0]

    # Tính sai số
    metrics = compute_metrics(fp32_hidden, quant_hidden)
    print("\n" + "-" * 50)
    print("📊 KẾT QUẢ ĐỐI CHIẾU SAI SỐ BERT (INT8 vs FP32):")
    print(f"• Cosine Similarity : {metrics['cosine_similarity']:.5f} (Mục tiêu: > 0.9850)")
    print(f"• SNR (Tỷ số tín hiệu): {metrics['snr_db']:.2f} dB (Mục tiêu: > 20 dB)")
    print(f"• Mean Squared Error: {metrics['mse']:.6e}")
    print(f"• Max Absolute Diff : {metrics['max_abs_diff']:.6e}")
    print("-" * 50)

    # Tải model về
    save_path = output_dir / "bert_wrapper_int8_quantized.onnx"
    quantized_model.download(str(save_path))
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Đã lưu model lượng tử hóa về máy: {save_path} ({size_mb:.2f} MB)")

    return {
        "model": "bert_wrapper",
        "precision": "int8",
        "metrics": metrics,
        "latency_summary": latency_ms,
        "saved_path": str(save_path),
        "file_size_mb": size_mb
    }


def main():
    parser = argparse.ArgumentParser(description="Chạy Pipeline Lượng tử hóa và Kiểm định trên Qualcomm AI Hub cho IQ-9075 EVK")
    parser.add_argument("--model", type=str, default="decoder", choices=["decoder", "bert", "all"],
                        help="Chọn mô hình cần chạy: 'decoder', 'bert', hoặc 'all'")
    parser.add_argument("--num_calib", type=int, default=30,
                        help="Số lượng mẫu calibration đưa lên AI Hub (mặc định: 30 mẫu)")
    parser.add_argument("--device_name", type=str, default="Dragonwing IQ-9075 EVK",
                        help="Tên thiết bị mục tiêu trên Qualcomm AI Hub")
    args = parser.parse_args()

    out_dir = Path("quantized_models")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🌟 QUALCOMM AI HUB AUTOMATED QUANTIZATION PIPELINE")
    print(f"• Thiết bị mục tiêu: {args.device_name}")
    print(f"• Mô hình chọn     : {args.model}")
    print(f"• Số mẫu Calib     : {args.num_calib}")
    print("=" * 65)

    try:
        target_device = hub.Device(args.device_name)
    except Exception as e:
        print(f"[!] Không tìm thấy thiết bị '{args.device_name}': {e}")
        print("    Đang chọn thiết bị thay thế có HTP v73: 'Snapdragon X Elite CRD'...")
        target_device = hub.Device("Snapdragon X Elite CRD")

    results = []
    start_all = time.time()

    if args.model in ["decoder", "all"]:
        res_dec = run_pipeline_decoder(target_device, num_calib=args.num_calib, output_dir=out_dir)
        results.append(res_dec)

    if args.model in ["bert", "all"]:
        res_bert = run_pipeline_bert(target_device, num_calib=args.num_calib, output_dir=out_dir)
        results.append(res_bert)

    total_time = time.time() - start_all

    print("\n" + "=" * 65)
    print(f"🎉 TỔNG KẾT TOÀN BỘ CÔNG VIỆC TRÊN QUALCOMM AI HUB (Tổng thời gian: {total_time:.1f}s)")
    print("=" * 65)
    for r in results:
        m = r["metrics"]
        print(f"[{r['model'].upper()} - {r['precision']}]")
        print(f"  • Cosine Similarity : {m['cosine_similarity']:.5f}")
        print(f"  • SNR               : {m['snr_db']:.2f} dB")
        print(f"  • File lưu tại      : {r['saved_path']}")
        print()
    print("Tất cả mô hình lượng tử hóa đã sẵn sàng để tích hợp vào hệ thống!")
    print("=" * 65)


if __name__ == "__main__":
    main()
