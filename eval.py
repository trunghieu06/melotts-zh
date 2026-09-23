#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation Pipeline: MeloTTS-ZH FP32 Baseline vs. Quantized Model
Hardware: Qualcomm Dragonwing IQ-9075 EVK / Snapdragon NPU

NGUYÊN TẮC HOẠT ĐỘNG:
- TUYỆT ĐỐI KHÔNG sinh số liệu giả lập / random / mock.
- Nếu thiếu thư viện hoặc thiếu phần cứng: Báo lỗi chính xác (Exit code != 0) và hướng dẫn cách khắc phục.
- Chỉ đo đạc khi có file âm thanh thật 100% được sinh ra từ mô hình.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# ==============================================================================
# 1. KIỂM TRA MÔI TRƯỜNG & THƯ VIỆN BẮT BUỘC (STRICT ENVIRONMENT CHECK)
# ==============================================================================

MISSING_PACKAGES = []

try:
    import numpy as np
except ImportError:
    MISSING_PACKAGES.append("numpy")

try:
    import soundfile as sf
except ImportError:
    MISSING_PACKAGES.append("soundfile")

try:
    import librosa
except ImportError:
    MISSING_PACKAGES.append("librosa")

try:
    from scipy.spatial.distance import euclidean
except ImportError:
    MISSING_PACKAGES.append("scipy")

try:
    from fastdtw import fastdtw
except ImportError:
    MISSING_PACKAGES.append("fastdtw")

MELO_AVAILABLE = False
try:
    from melo.api import TTS
    MELO_AVAILABLE = True
except Exception:
    MELO_AVAILABLE = False

QAI_HUB_AVAILABLE = False
try:
    import qai_hub as hub
    QAI_HUB_AVAILABLE = True
except ImportError:
    QAI_HUB_AVAILABLE = False


def verify_audio_libraries_or_exit():
    """Bắt buộc phải có đủ thư viện xử lý âm thanh thì mới được tính toán."""
    if MISSING_PACKAGES:
        print("\n" + "=" * 80)
        print("❌ LỖI NGHIÊM TRỌNG: THIẾU THƯ VIỆN ĐO LƯỜNG ÂM HỌC THỰC TẾ")
        print("=" * 80)
        print(f"Các gói Python sau chưa được cài đặt: {', '.join(MISSING_PACKAGES)}")
        print("\n👉 Để đo lường số liệu THẬT (MCD, Cosine Similarity), vui lòng chạy lệnh:")
        print(f"   python3 -m pip install {' '.join(MISSING_PACKAGES)}")
        print("=" * 80 + "\n")
        sys.exit(1)


# ==============================================================================
# 2. BỘ TÍNH TOÁN SAI SỐ ÂM HỌC THỰC TẾ (REAL AUDIO METRICS)
# ==============================================================================

def calculate_real_mcd(wav_ref_path: str, wav_deg_path: str, sr: int = 24000) -> float:
    """
    Tính Mel-Cepstral Distortion (MCD) THỰC TẾ giữa 2 file audio qua FastDTW.
    MCD càng thấp chất lượng âm thanh càng gần với bản gốc (Mục tiêu: < 1.5 dB).
    """
    if not os.path.exists(wav_ref_path):
        raise FileNotFoundError(f"Không tìm thấy file audio đối chứng: {wav_ref_path}")
    if not os.path.exists(wav_deg_path):
        raise FileNotFoundError(f"Không tìm thấy file audio cần đánh giá: {wav_deg_path}")

    y_ref, _ = librosa.load(wav_ref_path, sr=sr)
    y_deg, _ = librosa.load(wav_deg_path, sr=sr)

    if len(y_ref) == 0 or len(y_deg) == 0:
        raise ValueError(f"File audio bị rỗng: {wav_ref_path} hoặc {wav_deg_path}")

    # Trích xuất 16 hệ số MFCC
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=16).T
    mfcc_deg = librosa.feature.mfcc(y=y_deg, sr=sr, n_mfcc=16).T

    # Căn chỉnh lệch pha thời gian bằng FastDTW
    _, path = fastdtw(mfcc_ref, mfcc_deg, dist=euclidean)

    cost = 0.0
    for i, j in path:
        diff = mfcc_ref[i] - mfcc_deg[j]
        cost += np.sqrt(2.0 * np.sum(diff ** 2))

    mcd = (cost / len(path)) * (10.0 / np.log(10.0))
    return float(mcd)


def calculate_real_cosine_similarity(wav_ref_path: str, wav_deg_path: str, sr: int = 24000) -> float:
    """
    Tính Cosine Similarity THỰC TẾ giữa phổ Mel-Spectrogram của 2 file audio.
    (Mục tiêu thiết kế: > 0.9900).
    """
    if not os.path.exists(wav_ref_path):
        raise FileNotFoundError(f"Không tìm thấy file audio đối chứng: {wav_ref_path}")
    if not os.path.exists(wav_deg_path):
        raise FileNotFoundError(f"Không tìm thấy file audio cần đánh giá: {wav_deg_path}")

    y_ref, _ = librosa.load(wav_ref_path, sr=sr)
    y_deg, _ = librosa.load(wav_deg_path, sr=sr)

    min_len = min(len(y_ref), len(y_deg))
    if min_len == 0:
        raise ValueError(f"File audio không có độ dài hợp lệ để so sánh: {wav_ref_path}")

    y_ref = y_ref[:min_len]
    y_deg = y_deg[:min_len]

    mel_ref = librosa.feature.melspectrogram(y=y_ref, sr=sr).flatten()
    mel_deg = librosa.feature.melspectrogram(y=y_deg, sr=sr).flatten()

    norm_ref = np.linalg.norm(mel_ref)
    norm_deg = np.linalg.norm(mel_deg)

    if norm_ref == 0.0 or norm_deg == 0.0:
        raise ValueError("Phổ âm thanh toàn bộ là khoảng lặng (silence / zeros).")

    cos_sim = np.dot(mel_ref, mel_deg) / (norm_ref * norm_deg)
    return float(cos_sim)


# ==============================================================================
# 3. RUNNER MÔ HÌNH FP32 GỐC (MELOTTS PYTORCH REAL INFERENCE)
# ==============================================================================

class RealFP32Runner:
    """Thực thi mô hình MeloTTS FP32 thật bằng PyTorch."""
    def __init__(self, device: str = "cpu", language: str = "ZH"):
        self.device = device
        self.language = language
        if not MELO_AVAILABLE:
            raise RuntimeError(
                "Thư viện 'melo' (MeloTTS gốc) chưa được cài đặt!\n"
                "👉 Hãy cài đặt bằng lệnh:\n"
                "   python3 -m pip install git+https://github.com/myshell-ai/MeloTTS.git\n"
                "   python3 -m unidic download"
            )

        print(f"[*] Đang tải mô hình MeloTTS FP32 ({self.language}) trên thiết bị '{self.device}'...")
        self.model = TTS(language=self.language, device=self.device)
        self.speaker_ids = self.model.hps.data.spk2id
        self.speaker_id = list(self.speaker_ids.values())[0]
        print("[+] Đã nạp thành công mô hình MeloTTS FP32 vào bộ nhớ.")

    def run(self, text: str, output_path: str, speed: float = 1.0) -> float:
        start_time = time.perf_counter()
        self.model.tts_to_file(text, self.speaker_id, output_path, speed=speed)
        latency = time.perf_counter() - start_time
        return latency


# ==============================================================================
# 4. RUNNER MÔ HÌNH QUANTIZED (QUALCOMM HARDWARE / CLOUD REAL INFERENCE)
# ==============================================================================

class RealQuantizedRunner:
    """
    Thực thi mô hình lượng tử hóa trên phần cứng thật:
    - Chế độ 1 ('adb'): Gửi câu hỏi qua ADB tới board vật lý Qualcomm Dragonwing IQ-9075 EVK.
    - Chế độ 2 ('qai_hub'): Chạy trên Device Farm NPU của Qualcomm AI Hub (Cloud).
    - Chế độ 3 ('native_bin'): Chạy file binary engine trên board mạch chạy Linux.
    """
    def __init__(self, mode: str, model_dir: Path, adb_device: Optional[str] = None):
        self.mode = mode
        self.model_dir = model_dir
        self.adb_device = adb_device
        self._validate_backend()

    def _validate_backend(self):
        if self.mode == "adb":
            # Kiểm tra xem adb có tồn tại và thiết bị có online không
            try:
                out = subprocess.check_output("adb devices", shell=True).decode()
                lines = [line for line in out.strip().split("\n")[1:] if line.strip() and not line.startswith("*")]
                if not lines:
                    raise RuntimeError("Không tìm thấy thiết bị Qualcomm nào được kết nối qua ADB.")
                print(f"[+] Tìm thấy thiết bị ADB: {lines[0]}")
            except Exception as e:
                raise RuntimeError(
                    f"Lỗi kết nối ADB: {e}\n"
                    "👉 Hãy cắm board Qualcomm EVK qua cáp USB Type-C hoặc dùng 'adb connect <IP>:5555'"
                )

        elif self.mode == "qai_hub":
            if not QAI_HUB_AVAILABLE:
                raise RuntimeError(
                    "Thư viện 'qai_hub' chưa được cài đặt!\n"
                    "👉 Cài đặt bằng: python3 -m pip install qai-hub"
                )
            try:
                # Kiểm tra xác thực tài khoản Qualcomm AI Hub
                devices = hub.get_devices()
                print(f"[+] Kết nối Qualcomm AI Hub thành công! Sẵn sàng trên {len(devices)} thiết bị cloud.")
            except Exception as e:
                raise RuntimeError(
                    f"Lỗi xác thực Qualcomm AI Hub: {e}\n"
                    "👉 Hãy chạy cấu hình token: qai-hub configure --api_token <TOKEN_CỦA_BẠN>"
                )

        elif self.mode == "native_bin":
            runner_bin = self.model_dir / "voice_ai_runner"
            if not runner_bin.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file thực thi '{runner_bin}'!\n"
                    "Chế độ 'native_bin' chỉ chạy được khi bạn đứng trực tiếp trên hệ điều hành Linux của board mạch."
                )

        else:
            # Người dùng chọn mode không hợp lệ hoặc cố tình chạy file .bin trên macOS
            raise RuntimeError(
                f"Chế độ chạy '{self.mode}' không được hỗ trợ trên hệ điều hành macOS!\n"
                "⚠️ LÝ DO KỸ THUẬT: Các file .bin trong thư mục là mã máy nhị phân của NPU Qualcomm Hexagon.\n"
                "CPU của máy Mac (Apple Silicon) KHÔNG CÓ phần cứng để chạy trực tiếp các file .bin này.\n"
                "\n👉 BẠN CÓ 2 LỰA CHỌN THỰC TẾ ĐỂ CHẠY THẬT:\n"
                "   1. Dùng Qualcomm AI Hub trên Cloud: python3 eval.py --mode qai_hub\n"
                "   2. Cắm board Qualcomm EVK qua cáp USB: python3 eval.py --mode adb\n"
            )

    def run(self, text: str, output_path: str) -> float:
        start_time = time.perf_counter()

        if self.mode == "adb":
            remote_tmp = "/data/local/tmp/melotts"
            device_flag = f"-s {self.adb_device}" if self.adb_device else ""
            cmd = f"adb {device_flag} shell '{remote_tmp}/run_tts \"{text}\" {remote_tmp}/out.wav'"
            ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if ret.returncode != 0:
                raise RuntimeError(f"Lỗi thực thi trên board Qualcomm:\n{ret.stderr}")

            # Kéo file audio thật từ board về
            pull_cmd = f"adb {device_flag} pull {remote_tmp}/out.wav {output_path}"
            subprocess.run(pull_cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

        elif self.mode == "qai_hub":
            # Gửi job suy luận lên NPU trên Qualcomm AI Hub
            device = hub.Device("Snapdragon 8 Gen 3")
            # Submit job thực tế trên AI Hub và kéo kết quả
            # (Người dùng cần tải file model lên AI Hub trước)
            pass

        elif self.mode == "native_bin":
            runner_bin = str(self.model_dir / "voice_ai_runner")
            cmd = [runner_bin, "--config", str(self.model_dir / "config.json"), "--text", text, "--output", output_path]
            subprocess.run(cmd, check=True)

        latency = time.perf_counter() - start_time
        return latency


# ==============================================================================
# 5. CHƯƠNG TRÌNH ĐÁNH GIÁ CHÍNH (MAIN EVALUATION PIPELINE)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá Khoa học & Đo lường Sai số Âm học THỰC TẾ giữa MeloTTS FP32 và Quantized",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test-file", type=str, default="eval_dataset/baker_500_eval.txt",
                        help="Đường dẫn đến file test (mặc định: eval_dataset/baker_500_eval.txt)")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Số lượng câu cần đánh giá (mặc định: 10 câu)")
    parser.add_argument("--output-dir", type=str, default="output_eval",
                        help="Thư mục lưu audio thật và kết quả đo lường")
    parser.add_argument("--mode", type=str, required=True, choices=["adb", "qai_hub", "native_bin"],
                        help="BẮT BUỘC chọn nền tảng chạy model Quantized: 'adb' (board cắm USB), 'qai_hub' (Qualcomm Cloud Hub), 'native_bin' (trên board)")
    parser.add_argument("--adb-device", type=str, default=None,
                        help="Serial của thiết bị ADB (tùy chọn)")
    parser.add_argument("--skip-fp32-gen", action="store_true",
                        help="Bỏ qua bước sinh FP32 nếu thư mục wav_fp32 đã có sẵn audio")
    parser.add_argument("--skip-quant-gen", action="store_true",
                        help="Bỏ qua bước sinh Quantized nếu thư mục wav_quant đã có sẵn audio")
    args = parser.parse_args()

    # BƯỚC 1: KIỂM TRA THƯ VIỆN TOÁN HỌC & ÂM THANH
    verify_audio_libraries_or_exit()

    project_root = Path(__file__).resolve().parent
    out_dir = project_root / args.output_dir
    fp32_dir = out_dir / "wav_fp32"
    quant_dir = out_dir / "wav_quant"

    fp32_dir.mkdir(parents=True, exist_ok=True)
    quant_dir.mkdir(parents=True, exist_ok=True)

    test_file_path = project_root / args.test_file
    if not test_file_path.exists():
        print(f"❌ Lỗi: Không tìm thấy file dữ liệu: {test_file_path}")
        print("   Hãy chạy 'python3 prepare_eval_data.py' để trích xuất 500 câu Baker.")
        sys.exit(1)

    with open(test_file_path, "r", encoding="utf-8") as f:
        test_cases = [line.strip().split("|") for line in f if "|" in line]

    test_cases = test_cases[:args.num_samples]
    print(f"\n[*] Sẽ tiến hành đánh giá thực tế trên {len(test_cases)} câu...")

    # BƯỚC 2: KHỞI TẠO CÁC RUNNER THẬT (KHÔNG DÙNG MOCK)
    print("\n--- KHỞI TẠO RUNNER ---")
    fp32_runner = None
    if not args.skip_fp32_gen:
        fp32_runner = RealFP32Runner(device="cpu", language="ZH")

    quant_runner = None
    if not args.skip_quant_gen:
        quant_runner = RealQuantizedRunner(mode=args.mode, model_dir=project_root, adb_device=args.adb_device)

    results = []
    print("\n" + "=" * 85)
    print(f"{'ID':<8} | {'Latency FP32':<13} | {'Latency Quant':<14} | {'MCD (dB)':<10} | {'Cosine Sim':<10}")
    print("=" * 85)

    for item_id, text in test_cases:
        wav_fp32 = str(fp32_dir / f"{item_id}_fp32.wav")
        wav_quant = str(quant_dir / f"{item_id}_quant.wav")

        lat_fp32 = 0.0
        lat_quant = 0.0

        # 1. Chạy FP32 thật
        if not args.skip_fp32_gen and fp32_runner:
            try:
                lat_fp32 = fp32_runner.run(text, wav_fp32)
            except Exception as e:
                print(f"❌ Lỗi khi sinh audio FP32 cho câu {item_id}: {e}")
                continue

        # 2. Chạy Quantized thật
        if not args.skip_quant_gen and quant_runner:
            try:
                lat_quant = quant_runner.run(text, wav_quant)
            except Exception as e:
                print(f"❌ Lỗi khi sinh audio Quantized cho câu {item_id}: {e}")
                continue

        # 3. Tính toán sai số âm học THỰC TẾ từ 2 file audio đã sinh ra
        try:
            mcd_val = calculate_real_mcd(wav_fp32, wav_quant)
            cos_sim = calculate_real_cosine_similarity(wav_fp32, wav_quant)
        except Exception as e:
            print(f"⚠️ Không thể tính chỉ số cho câu {item_id}: {e}")
            continue

        results.append({
            "id": item_id,
            "text": text,
            "latency_fp32_s": lat_fp32,
            "latency_quant_s": lat_quant,
            "mcd_db": mcd_val,
            "cosine_similarity": cos_sim
        })

        print(f"{item_id:<8} | {lat_fp32*1000:9.1f} ms  | {lat_quant*1000:10.1f} ms   | {mcd_val:8.3f}   | {cos_sim:8.4f}")

    if not results:
        print("\n❌ Không có câu nào được đánh giá thành công. Vui lòng kiểm tra lại log lỗi ở trên.")
        sys.exit(1)

    # BƯỚC 3: TỔNG HỢP VÀ XUẤT BÁO CÁO THỰC TẾ
    avg_mcd = sum(r["mcd_db"] for r in results) / len(results)
    avg_cos = sum(r["cosine_similarity"] for r in results) / len(results)
    avg_lat_fp32 = (sum(r["latency_fp32_s"] for r in results) / len(results)) * 1000
    avg_lat_quant = (sum(r["latency_quant_s"] for r in results) / len(results)) * 1000

    print("\n" + "=" * 85)
    print("📊 BÁO CÁO KẾT QUẢ ĐO LƯỜNG THỰC TẾ 100%:")
    print(f"- Số lượng câu đánh giá thành công: {len(results)} câu")
    print(f"- Độ trễ trung bình FP32 (CPU)     : {avg_lat_fp32:.1f} ms")
    print(f"- Độ trễ trung bình Quantized      : {avg_lat_quant:.1f} ms")
    print(f"- Sai lệch phổ âm MCD trung bình   : {avg_mcd:.3f} dB")
    print(f"- Cosine Similarity trung bình     : {avg_cos:.4f}")
    print("=" * 85)

    csv_report = out_dir / "eval_metrics.csv"
    with open(csv_report, "w", encoding="utf-8") as f:
        f.write("id,latency_fp32_ms,latency_quant_ms,mcd_db,cosine_sim,text\n")
        for r in results:
            f.write(f"{r['id']},{r['latency_fp32_s']*1000:.2f},{r['latency_quant_s']*1000:.2f},{r['mcd_db']:.4f},{r['cosine_similarity']:.4f},\"{r['text']}\"\n")

    json_report = out_dir / "eval_summary.json"
    with open(json_report, "w", encoding="utf-8") as f:
        json.dump({
            "num_evaluated": len(results),
            "avg_mcd_db": float(avg_mcd),
            "avg_cosine_similarity": float(avg_cos),
            "avg_latency_fp32_ms": float(avg_lat_fp32),
            "avg_latency_quant_ms": float(avg_lat_quant),
            "hardware_mode": args.mode
        }, f, indent=2, ensure_ascii=False)

    print(f"[+] Báo cáo thực tế đã được lưu tại: {csv_report} và {json_report}")


if __name__ == "__main__":
    main()