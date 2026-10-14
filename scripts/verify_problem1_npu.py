"""
Script kiểm chứng và đánh giá Problem 1 (Byte-Level Transformer trên NPU Qualcomm IQ-9075 EVK)
================================================================================================
Thực hiện:
1. Đọc và theo dõi trạng thái các Job: Compile, Profile, Inference trên AI Hub.
2. Trích xuất chỉ số phần cứng thật:
   - Tỷ lệ NPU Acceleration (HTP % và số lượng ops trên NPU, CPU fallback = 0).
   - Latency (thời gian suy luận tính bằng ms).
   - Peak Memory (bộ nhớ đỉnh tính bằng MB).
3. Đánh giá tính đúng đắn (Correctness):
   - Chạy ONNX Runtime local với chuỗi UTF-8: "你好Hi" -> bytes [228, 189, 160, 229, 165, 189, 72, 105] + padding.
   - So sánh Cosine Similarity giữa logits sinh ra trên NPU (QNN DLC) và FP32 ONNX Runtime.
"""

import os
import sys
import json
import numpy as np
import onnxruntime as ort
import qai_hub as hub
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from step1_asr.byte_transformer import text_to_byte_tensor

def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def inspect_profile_job(job_id: str):
    """Trích xuất chi tiết latency, RAM và tỷ lệ op chạy trên NPU từ Profile Job"""
    print(f"\n📊 --- Báo Cáo Profile Job: {job_id} ---")
    p_job = hub.get_job(job_id)
    status = p_job.get_status().code
    print(f"Trạng thái: {status} (URL: {p_job.url})")
    if status != "SUCCESS":
        return None

    profile_data = p_job.download_profile()
    exec_summary = profile_data.get("execution_summary", {})
    estimated_inference_time = exec_summary.get("estimated_inference_time", None) # microseconds
    inference_time_ms = estimated_inference_time / 1000.0 if estimated_inference_time else None
    peak_memory_bytes = exec_summary.get("estimated_inference_peak_memory", None)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024) if peak_memory_bytes else None

    # Operators on NPU vs CPU
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

    print(f"⏱️ Latency ước tính     : {inference_time_ms:.3f} ms" if inference_time_ms else "⏱️ Latency: N/A")
    print(f"💾 Peak Memory (RAM)    : {peak_memory_mb:.2f} MB" if peak_memory_mb else "💾 Peak Memory: N/A")
    print(f"⚡ Phân bổ toán tử NPU : {npu_ops}/{total_ops} ops (CPU Fallback: {cpu_ops} ops)")
    if total_ops > 0:
        npu_ratio = (npu_ops / total_ops) * 100
        print(f"🎯 Tỷ lệ NPU Native     : {npu_ratio:.1f}%")

    return {
        "job_id": job_id,
        "latency_ms": inference_time_ms,
        "peak_memory_mb": peak_memory_mb,
        "npu_ops": npu_ops,
        "total_ops": total_ops,
        "cpu_ops": cpu_ops,
    }

def verify_inference_job(job_id: str, onnx_model_path: str, text: str = "你好Hi"):
    """Lấy output từ Inference Job và so khớp với ONNX Runtime FP32 Reference"""
    print(f"\n🔬 --- Kiểm Chứng Kết Quả Inference Job: {job_id} ---")
    i_job = hub.get_job(job_id)
    status = i_job.get_status().code
    print(f"Trạng thái: {status} (URL: {i_job.url})")
    if status != "SUCCESS":
        return None

    # 1. Tải output từ NPU
    output_data = i_job.download_output_data()
    out_key = list(output_data.keys())[0]
    npu_logits = output_data[out_key][0] # np.ndarray
    print(f"QNN NPU Logits Shape: {npu_logits.shape}")

    # 2. Chạy ONNX Runtime cục bộ
    input_tensor, raw_bytes = text_to_byte_tensor(text, seq_len=512)
    sess = ort.InferenceSession(onnx_model_path)
    ref_logits = sess.run(None, {"byte_indices": input_tensor.numpy()})[0]
    print(f"Local Ref Shape     : {ref_logits.shape}")

    sim = compute_cosine_similarity(ref_logits, npu_logits)
    max_diff = float(np.max(np.abs(ref_logits - npu_logits)))
    print(f"Cosine Similarity   : {sim:.6f}")
    print(f"Max Absolute Diff   : {max_diff:.6e}")

    # Top-1 byte prediction tại vị trí kết thúc chuỗi
    seq_end_pos = len(raw_bytes) - 1
    npu_top_byte = int(np.argmax(npu_logits[0, 0, seq_end_pos, :]))
    ref_top_byte = int(np.argmax(ref_logits[0, 0, seq_end_pos, :]))
    print(f"Input text          : '{text}' (Bytes: {raw_bytes})")
    print(f"Top-1 byte tại pos {seq_end_pos}: NPU={npu_top_byte}, Local={ref_top_byte} (Khớp={npu_top_byte == ref_top_byte})")

    return {
        "job_id": job_id,
        "npu_shape": list(npu_logits.shape),
        "cosine_sim": sim,
        "max_diff": max_diff,
        "top_byte_npu": npu_top_byte,
        "top_byte_ref": ref_top_byte
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🏆 BÁO CÁO TỔNG HỢP KIỂM CHỨNG PROBLEM 1: BYTE-TRANSFORMER NPU")
    print("=" * 60)
    
    # Tiny
    print("\n[1] TINY MODEL (D=128, L=2):")
    inspect_profile_job("jpv2e9e7g")
    
    # Big
    print("\n[2] BIG MODEL (D=512, L=8, 25.45M PARAMS):")
    inspect_profile_job("jgd68zzlp")
    verify_inference_job("j57ok79rg", "outputs/byte_transformer_big.onnx", text="你好Hi")
    print("=" * 60)
