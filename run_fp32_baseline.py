# run_fp32_baseline.py
import os
import time
from pathlib import Path
from melo.api import TTS

output_dir = Path("output_eval/wav_fp32")
output_dir.mkdir(parents=True, exist_ok=True)

# Khởi tạo MeloTTS FP32 gốc
print("[*] Đang nạp mô hình MeloTTS gốc (FP32)...")
model = TTS(language="ZH", device="cpu")
speaker_id = list(model.hps.data.spk2id.values())[0]

# Đọc danh sách câu test
with open("eval_dataset/baker_500_eval.txt", "r", encoding="utf-8") as f:
    lines = [line.strip().split("|") for line in f if "|" in line]

# Chạy thử nghiệm trước 10 câu (hoặc bỏ [:10] để chạy hết 500 câu)
test_cases = lines
print(f"[*] Đang sinh audio chuẩn FP32 cho {len(test_cases)} câu...")

for item_id, text in test_cases:
    wav_path = output_dir / f"{item_id}_fp32.wav"
    t0 = time.perf_counter()
    model.tts_to_file(text, speaker_id, str(wav_path), speed=1.0)
    lat_ms = (time.perf_counter() - t0) * 1000
    print(f"✓ Đã xong {item_id}: {lat_ms:.1f} ms | Text: {text}")

print(f"\n[+] Đã sinh xong toàn bộ file audio FP32 đối chứng tại: {output_dir}")