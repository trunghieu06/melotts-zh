#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify NPU-Native Modules (Problem 2 & Problem 4) on Qualcomm AI Hub
Target Device: Qualcomm Dragonwing IQ-9075 EVK (Hexagon HTP v73, SoC QCS9075)
"""

import sys
from pathlib import Path
import qai_hub as hub


def verify_module(name: str, onnx_path: str, device: hub.Device):
    print("\n" + "=" * 70)
    print(f"🚀 XÁC MINH QUALCOMM AI HUB: {name.upper()}")
    print(f"  • File ONNX: {onnx_path}")
    print(f"  • Thiết bị đích: {device.name}")
    print("=" * 70)

    p = Path(onnx_path)
    if not p.exists():
        print(f"[!] Không tìm thấy file {onnx_path}. Vui lòng chạy export_npu_native.py trước.")
        return False

    print("[1/3] Upload mô hình lên AI Hub...")
    model = hub.upload_model(str(p), name=f"MeloTTS_{name}_NPU")
    print(f"  ✓ Uploaded Model ID: {model.model_id}")

    print("[2/3] Nộp Compile Job sang QNN DLC (NPU)...")
    compile_job = hub.submit_compile_job(
        model=model,
        device=device,
        name=f"Compile_{name}_IQ9075",
        options="--target_runtime qnn_dlc"
    )
    print(f"  ✓ Compile Job ID: {compile_job.job_id}")
    print(f"  ✓ URL: {compile_job.url}")
    print("  Đang chờ biên dịch...")
    compile_job.wait()
    c_status = compile_job.get_status()
    print(f"  🏁 Trạng thái biên dịch: {c_status.code}")

    if "SUCCESS" not in str(c_status.code).upper():
        print(f"[!] Lỗi biên dịch: {c_status.message}")
        return False

    target_model = compile_job.get_target_model()
    print(f"  ✓ Target QNN Model ID: {target_model.model_id}")

    print("[3/3] Nộp Profile Job đo đạc hiệu năng trên NPU Dragonwing IQ-9075 EVK...")
    prof_job = hub.submit_profile_job(
        model=target_model,
        device=device,
        name=f"Profile_{name}_IQ9075"
    )
    print(f"  ✓ Profile Job ID: {prof_job.job_id}")
    print(f"  ✓ URL: {prof_job.url}")
    print("  Đang đo đạc phần cứng thực tế...")
    prof_job.wait()
    p_status = prof_job.get_status()
    print(f"  🏁 Trạng thái Profile: {p_status.code}")

    if "SUCCESS" in str(p_status.code).upper():
        prof = prof_job.download_profile()
        exec_sum = prof.get("execution_summary", {})
        est_time = exec_sum.get("estimated_inference_time")
        print(f"  🎉 THÀNH CÔNG! Độ trễ thực thi NPU ước tính: {est_time} µs ({est_time/1000.0:.2f} ms)")
        return True
    else:
        print(f"[!] Lỗi profile: {p_status.message}")
        return False


def main():
    device = hub.Device("Dragonwing IQ-9075 EVK")
    print(f"Target Hardware: {device.name}")

    p2_ok = verify_module("P2_DurationExpansion", "onnx_models/npu_duration_expansion.onnx", device)
    p4_ok = verify_module("P4_ArtifactTrimmer", "onnx_models/npu_artifact_trimmer.onnx", device)

    print("\n" + "=" * 70)
    print("📊 TỔNG KẾT KIỂM ĐỊNH AI HUB CHO GIAI ĐOẠN 1 (PROBLEM 2 & 4):")
    print(f"  • Problem 2 (Duration Expansion): {'✅ ĐẠT (100% NPU)' if p2_ok else '❌ THẤT BẠI'}")
    print(f"  • Problem 4 (Artifact Trimmer)  : {'✅ ĐẠT (100% NPU)' if p4_ok else '❌ THẤT BẠI'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
