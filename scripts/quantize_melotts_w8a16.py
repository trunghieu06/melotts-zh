"""
OneVoice — MeloTTS-ZH Quantization (w8a16) cho IQ-9075 EVK
==========================================================
Chạy: python scripts/quantize_melotts_w8a16.py

Script này submit model MeloTTS lên Qualcomm AI Hub để quantize
sử dụng recipe w8a16 (Weights: INT8, Activations: INT16).
"""

import os
import qai_hub as hub
import torch
import numpy as np
from pathlib import Path

# Cấu hình
DEVICE_NAME = "Dragonwing IQ-9075 EVK"
OUT_DIR = Path("outputs/melotts_quantized")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def check_qai_hub_models():
    """Kiểm tra xem Qualcomm AI Hub Models có sẵn MeloTTS không"""
    try:
        import qai_hub_models.models
        available_models = dir(qai_hub_models.models)
        melo_models = [m for m in available_models if "melo" in m.lower() or "tts" in m.lower() or "vits" in m.lower()]
        if melo_models:
            print(f"✅ Tìm thấy các model TTS trên qai-hub-models: {melo_models}")
            return True
        else:
            print("❌ Không tìm thấy MeloTTS trong qai-hub-models. Sẽ cần tự export ONNX.")
            return False
    except ImportError:
        print("⚠️ Chưa cài qai-hub-models. Hãy chạy: pip install qai-hub-models")
        return False

def main():
    print(f"🚀 Bắt đầu quy trình Deploy MeloTTS-ZH lên {DEVICE_NAME}")
    
    # 1. Kiểm tra QAI Hub Models
    has_official_model = check_qai_hub_models()
    
    if has_official_model:
        print("\n💡 Gợi ý: Có vẻ có model TTS sẵn từ qai-hub-models. Hãy kiểm tra xem có phải MeloTTS không.")
    else:
        print("\n⚠️ Vì chưa có model sẵn, chúng ta cần tự export MeloTTS sang ONNX trước khi quantize.")
        
    print("\nVui lòng báo kết quả cho tôi để tiếp tục!")

if __name__ == "__main__":
    main()
