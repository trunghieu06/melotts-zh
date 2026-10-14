"""
Đánh giá độ chính xác (Cosine Similarity) của MeloTTS-ZH
So sánh PyTorch FP32 vs NPU QNN (w8a16)
"""

import os
import json
import torch
import numpy as np
import qai_hub as hub
from pathlib import Path

# Cấu hình
MANIFEST = "data/mt/manifest.json"
MODEL_DIR = Path("melotts_zh-voice_ai-mixed_with_float-qualcomm_qcs9075")
DEVICE_NAME = "Dragonwing IQ-9075 EVK"

def cosine_sim(a, b):
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-9)

def main():
    print("🚀 Khởi tạo bài test Cosine Similarity cho MeloTTS-ZH...")
    
    # 1. Đọc câu đầu tiên từ manifest.json
    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)
    text = rows[0]["zh"]
    print(f"\n📝 Câu test (1/5): '{text}'")

    print("\n⏳ Đang load PyTorch FP32 model để lấy kết quả chuẩn (Reference)...")
    try:
        from melo.api import TTS
        model_fp32 = TTS(language="ZH", device="cpu")
        model_fp32.hps.data.disable_bert = True
        print("✅ Load PyTorch thành công!")
    except ImportError:
        print("❌ Lỗi: Chưa cài đặt thư viện MeloTTS gốc. Xin hãy cài đặt trước.")
        return

    print("\n⚠️ Đang phân tích kiến trúc để móc nối Tensors (Quá trình này cần 5-10 phút để chạy 3 Jobs trên Cloud...)")
    print("Vui lòng chờ AI Agent tự động gửi và tính toán độ chênh lệch!")
    
    # --- PHẦN CODE CHẠY TRÊN CLOUD SẼ ĐƯỢC CẬP NHẬT Ở BƯỚC TIẾP THEO ---
    # (Agent đang xử lý trích xuất Tensors bên trong MeloTTS để làm Input cho AI Hub)
    
if __name__ == "__main__":
    main()
