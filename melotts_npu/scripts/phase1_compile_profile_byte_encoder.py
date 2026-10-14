"""
Phase 1: Biên dịch, Profile và Kiểm chứng Inference mô hình melo_byte_encoder_npu.onnx
trên bo mạch phần cứng Qualcomm Dragonwing IQ-9075 EVK qua Qualcomm AI Hub.
"""

import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort
import qai_hub as hub
from pathlib import Path

# Thêm đường dẫn src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
from step1_asr.byte_transformer import text_to_byte_tensor

MODEL_ONNX_PATH = PROJECT_ROOT / "melo_byte_encoder_npu.onnx"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DLC_PATH = OUTPUT_DIR / "melo_byte_encoder_npu.dlc"
RESULTS_JSON_PATH = OUTPUT_DIR / "phase1_byte_encoder_npu_results.json"

DEVICE_NAME = "Dragonwing IQ-9075 EVK"
TEST_TEXT = "你好，很高兴认识你。"

def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def main():
    print("=" * 70)
    print("🚀 GIAI ĐOẠN 1: BIÊN DỊCH VÀ KIỂM CHỨNG melo_byte_encoder_npu TRÊN NPU")
    print(f"🎯 Thiết bị mục tiêu: {DEVICE_NAME}")
    print(f"📦 Mô hình ONNX: {MODEL_ONNX_PATH} ({MODEL_ONNX_PATH.stat().st_size / 1e6:.2f} MB)")
    print("=" * 70)

    # 1. Kiểm tra mô hình ONNX
    if not MODEL_ONNX_PATH.exists():
        print(f"❌ Không tìm thấy file: {MODEL_ONNX_PATH}")
        sys.exit(1)

    print("\n[Bước 1/5] ☁️ Upload mô hình lên Qualcomm AI Hub...")
    hub_model = hub.upload_model(str(MODEL_ONNX_PATH))
    print(f"✅ Upload thành công! Hub Model ID: {hub_model.model_id}")

    device = hub.Device(DEVICE_NAME)

    # 2. Submit Compile Job
    print("\n[Bước 2/5] ⚙️ Khởi tạo Compile Job sang QNN DLC (Compute Unit: NPU)...")
    compile_options = "--target_runtime qnn_dlc --compute_unit npu"
    compile_job = hub.submit_compile_job(
        model=hub_model,
        device=device,
        options=compile_options,
        name="melo_byte_encoder_compile_npu"
    )
    print(f"⏳ Job ID: {compile_job.job_id} | URL: {compile_job.url}")
    print("Đang chờ NPU compiler biên dịch đồ thị...")
    compile_job.wait()

    compile_status = compile_job.get_status().code
    print(f"Trạng thái Compile: {compile_status}")
    if compile_status != "SUCCESS":
        print(f"❌ Biên dịch thất bại! Logs: {compile_job.download_job_logs()}")
        sys.exit(1)

    compiled_model = compile_job.get_target_model()
    print(f"✅ Biên dịch thành công! Compiled Model ID: {compiled_model.model_id}")

    # Tải về file DLC
    print(f"💾 Đang tải tệp QNN DLC về: {TARGET_DLC_PATH}...")
    try:
        compiled_model.download(str(TARGET_DLC_PATH))
        print(f"✅ Đã lưu: {TARGET_DLC_PATH} ({TARGET_DLC_PATH.stat().st_size / 1e6:.2f} MB)")
    except Exception as e:
        print(f"⚠️ Chú ý: Không tải được trực tiếp qua download(): {e}")

    # 3. Submit Profile Job trên phần cứng thật
    print("\n[Bước 3/5] ⏱️ Khởi tạo Profile Job trên phần cứng thật Dragonwing IQ-9075 EVK...")
    profile_job = hub.submit_profile_job(
        model=compiled_model,
        device=device,
        name="melo_byte_encoder_profile_npu"
    )
    print(f"⏳ Job ID: {profile_job.job_id} | URL: {profile_job.url}")
    print("Đang chờ bo mạch EVK thực thi đo đạc (chu kỳ xung nhịp HTP & RAM)...")
    profile_job.wait()

    profile_status = profile_job.get_status().code
    print(f"Trạng thái Profile: {profile_status}")
    if profile_status != "SUCCESS":
        print(f"❌ Profile thất bại! Logs: {profile_job.download_job_logs()}")
        sys.exit(1)

    profile_data = profile_job.download_profile()
    exec_summary = profile_data.get("execution_summary", {})
    estimated_inference_time = exec_summary.get("estimated_inference_time", None) # microseconds
    inference_time_ms = estimated_inference_time / 1000.0 if estimated_inference_time else None
    peak_memory_bytes = exec_summary.get("estimated_inference_peak_memory", None)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024) if peak_memory_bytes else None

    exec_detail = profile_data.get("execution_detail", [])
    total_ops = len(exec_detail)
    npu_ops = 0
    cpu_ops = 0
    for op in exec_detail:
        comp_unit = op.get("compute_unit", "").upper()
        if "NPU" in comp_unit or "HTP" in comp_unit:
            npu_ops += 1
        else:
            cpu_ops += 1

    npu_ratio = (npu_ops / total_ops * 100) if total_ops > 0 else 0.0

    print("\n📊 --- KẾT QUẢ ĐO ĐẠC PHẦN CỨNG THẬT TRÊN IQ-9075 EVK ---")
    print(f"⚡ Phân bổ Toán tử NPU : {npu_ops}/{total_ops} ops ({npu_ratio:.1f}%)")
    print(f"💻 CPU Fallback         : {cpu_ops} ops ({'0% CPU Fallback' if cpu_ops == 0 else 'Cảnh báo: có CPU fallback'})")
    print(f"⏱️ Độ trễ suy luận (ms): {inference_time_ms:.3f} ms" if inference_time_ms else "⏱️ Độ trễ: N/A")
    print(f"💾 Peak Memory (RAM)    : {peak_memory_mb:.2f} MB" if peak_memory_mb else "💾 Peak Memory: N/A")

    # 4. Submit Inference Job với dữ liệu tiếng Trung thật
    print("\n[Bước 4/5] 🔬 Khởi tạo Inference Job trên NPU với dữ liệu câu nói thật...")
    print(f"📝 Câu kiểm tra: '{TEST_TEXT}'")
    input_tensor, raw_bytes = text_to_byte_tensor(TEST_TEXT, seq_len=512)
    print(f"Dãy UTF-8 bytes ({len(raw_bytes)} bytes): {raw_bytes}")

    input_data = {"byte_indices": [input_tensor.numpy()]}
    infer_job = hub.submit_inference_job(
        model=compiled_model,
        device=device,
        inputs=input_data,
        name="melo_byte_encoder_infer_npu"
    )
    print(f"⏳ Job ID: {infer_job.job_id} | URL: {infer_job.url}")
    print("Đang chờ NPU EVK tính toán và trả về output...")
    infer_job.wait()

    infer_status = infer_job.get_status().code
    print(f"Trạng thái Inference: {infer_status}")
    if infer_status != "SUCCESS":
        print(f"❌ Inference thất bại! Logs: {infer_job.download_job_logs()}")
        sys.exit(1)

    # 5. So sánh với ONNX Runtime Local (FP32 Reference)
    print("\n[Bước 5/5] 📐 Đối soát kết quả NPU vs ONNX Runtime FP32 cục bộ...")
    npu_outputs = infer_job.download_output_data()
    out_name = list(npu_outputs.keys())[0]
    npu_feat = npu_outputs[out_name][0] # np.ndarray
    print(f"Output tensor từ NPU shape: {npu_feat.shape}, dtype: {npu_feat.dtype}")

    # Chạy ONNX Runtime local
    ort_sess = ort.InferenceSession(str(MODEL_ONNX_PATH))
    ref_feat = ort_sess.run(None, {"byte_indices": input_tensor.numpy()})[0]
    print(f"Output tensor local shape : {ref_feat.shape}, dtype: {ref_feat.dtype}")

    cos_sim = compute_cosine_similarity(npu_feat, ref_feat)
    max_abs_err = float(np.max(np.abs(npu_feat - ref_feat)))
    mean_abs_err = float(np.mean(np.abs(npu_feat - ref_feat)))

    print("\n🎯 --- KẾT QUẢ KIỂM CHỨNG ĐỘ CHÍNH XÁC ---")
    print(f"✨ Cosine Similarity : {cos_sim:.6f} (Mục tiêu >= 0.99)")
    print(f"📉 Max Absolute Error: {max_abs_err:.6e}")
    print(f"📉 Mean Absolute Err : {mean_abs_err:.6e}")

    # Lưu toàn bộ báo cáo
    report = {
        "device": DEVICE_NAME,
        "onnx_model": str(MODEL_ONNX_PATH),
        "test_text": TEST_TEXT,
        "test_bytes_length": len(raw_bytes),
        "compile_job": {
            "id": compile_job.job_id,
            "url": compile_job.url,
            "status": compile_status,
            "options": compile_options
        },
        "profile_job": {
            "id": profile_job.job_id,
            "url": profile_job.url,
            "status": profile_status,
            "latency_ms": inference_time_ms,
            "peak_memory_mb": peak_memory_mb,
            "npu_ops": npu_ops,
            "total_ops": total_ops,
            "cpu_ops": cpu_ops,
            "npu_ratio_percent": npu_ratio
        },
        "inference_job": {
            "id": infer_job.job_id,
            "url": infer_job.url,
            "status": infer_status,
            "output_shape": list(npu_feat.shape)
        },
        "accuracy_metrics": {
            "cosine_similarity": cos_sim,
            "max_absolute_error": max_abs_err,
            "mean_absolute_error": mean_abs_err
        }
    }

    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    print(f"\n📄 Đã lưu kết quả chi tiết tại: {RESULTS_JSON_PATH}")
    print("=" * 70)
    print("🎉 HOÀN THÀNH GIAI ĐOẠN 1: MÔ HÌNH SẴN SÀNG CHO GIAI ĐOẠN GHÉP NỐI!")
    print("=" * 70)

if __name__ == "__main__":
    main()
