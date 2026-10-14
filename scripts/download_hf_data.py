"""
OneVoice — Script tải dữ liệu từ HuggingFace
=============================================
Chạy: python scripts/download_hf_data.py

Dataset sẽ được lưu vào thư mục data/ theo cấu trúc:
  data/asr/vimd/          — ViMD (đa giọng 63 tỉnh)
  data/asr/fleurs_vi/     — FLEURS Vietnamese (benchmark)
  data/asr/common_voice/  — Common Voice Vietnamese
  data/mt/flores/          — FLORES-200 (benchmark dịch)
  data/mt/phomt/          — PhoMT Vi-En (3M cặp câu)
"""

import os
import sys
from pathlib import Path

# Thư mục gốc của project
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def ensure_dirs():
    """Tạo cấu trúc thư mục nếu chưa có"""
    dirs = [
        DATA_DIR / "asr" / "vimd",
        DATA_DIR / "asr" / "fleurs_vi",
        DATA_DIR / "asr" / "common_voice",
        DATA_DIR / "mt" / "flores",
        DATA_DIR / "mt" / "phomt",
        DATA_DIR / "mt" / "opus_ko_vi",
        DATA_DIR / "noise" / "musan",
        DATA_DIR / "test_sealed",  # Bộ test niêm phong — KHÔNG dùng cho training
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  📁 {d.relative_to(PROJECT_ROOT)}")

def download_vimd():
    """
    ViMD — Vietnamese Multi-Dialect (102.56 giờ, 63 tỉnh thành)
    Dùng để fine-tune ASR cho đa giọng địa phương
    License: CC BY-NC-ND 4.0
    """
    print("\n" + "="*60)
    print("📥 [1/5] Tải ViMD — Đa giọng 63 tỉnh thành")
    print("="*60)
    
    save_path = DATA_DIR / "asr" / "vimd"
    
    try:
        from datasets import load_dataset
        
        print("  ⏳ Đang tải từ HuggingFace (có thể mất 10-30 phút)...")
        ds = load_dataset("nguyendv02/ViMD_Dataset", cache_dir=str(save_path))
        
        # In thông tin
        print(f"  ✅ Tải xong!")
        print(f"  📊 Thông tin dataset:")
        for split_name, split_data in ds.items():
            print(f"     - {split_name}: {len(split_data)} mẫu")
        
        # Kiểm tra các cột
        print(f"  📋 Các cột: {ds[list(ds.keys())[0]].column_names}")
        
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False

def download_fleurs_vi():
    """
    FLEURS Vietnamese — Benchmark chất lượng cao (12 giờ)
    Dùng để đánh giá ASR
    License: CC BY 4.0
    """
    print("\n" + "="*60)
    print("📥 [2/5] Tải FLEURS Vietnamese — Benchmark ASR")
    print("="*60)
    
    save_path = DATA_DIR / "asr" / "fleurs_vi"
    
    try:
        from datasets import load_dataset
        
        print("  ⏳ Đang tải...")
        ds = load_dataset("google/fleurs", "vi_vn", cache_dir=str(save_path))
        
        print(f"  ✅ Tải xong!")
        for split_name, split_data in ds.items():
            print(f"     - {split_name}: {len(split_data)} mẫu")
        
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False

def download_common_voice_vi():
    """
    Common Voice Vietnamese — Crowdsourced (~30-50 giờ)
    License: CC0 (Public Domain)
    ⚠️ Cần đăng nhập HuggingFace trước: huggingface-cli login
    """
    print("\n" + "="*60)
    print("📥 [3/5] Tải Common Voice Vietnamese")
    print("="*60)
    
    save_path = DATA_DIR / "asr" / "common_voice"
    
    try:
        from datasets import load_dataset
        
        print("  ⏳ Đang tải (cần HF login, có thể mất 20-40 phút)...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0", 
            "vi",
            cache_dir=str(save_path),
            trust_remote_code=True
        )
        
        print(f"  ✅ Tải xong!")
        for split_name, split_data in ds.items():
            print(f"     - {split_name}: {len(split_data)} mẫu")
        
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        if "gated" in str(e).lower() or "login" in str(e).lower():
            print("  💡 Cần đăng nhập HuggingFace trước:")
            print("     huggingface-cli login")
            print("     (Lấy token tại: https://huggingface.co/settings/tokens)")
        return False

def download_flores():
    """
    FLORES-200 — Benchmark dịch đa ngôn ngữ (3001 câu/ngôn ngữ)
    Dùng để đánh giá chất lượng MT
    License: CC BY-SA 4.0
    """
    print("\n" + "="*60)
    print("📥 [4/5] Tải FLORES-200 — Benchmark dịch máy")
    print("="*60)
    
    save_path = DATA_DIR / "mt" / "flores"
    
    try:
        from datasets import load_dataset
        
        # Tải các cặp ngôn ngữ cần thiết
        pairs = [
            ("vie_Latn", "eng_Latn", "Vi-En"),
            ("vie_Latn", "zho_Hans", "Vi-Zh"),
            ("vie_Latn", "kor_Hang", "Vi-Ko"),
        ]
        
        print("  ⏳ Đang tải...")
        ds = load_dataset("openlanguagedata/flores_plus", cache_dir=str(save_path))
        
        print(f"  ✅ Tải xong!")
        for split_name, split_data in ds.items():
            print(f"     - {split_name}: {len(split_data)} mẫu")
        
        # Kiểm tra các ngôn ngữ cần thiết
        sample = ds[list(ds.keys())[0]][0]
        langs_needed = ["vie_Latn", "eng_Latn", "zho_Hans", "kor_Hang"]
        available = [l for l in langs_needed if l in sample]
        missing = [l for l in langs_needed if l not in sample]
        
        print(f"  🌐 Ngôn ngữ có sẵn: {available}")
        if missing:
            print(f"  ⚠️ Thiếu: {missing}")
        
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        return False

def download_phomt():
    """
    PhoMT — Vi-En Machine Translation (3.02M cặp câu)
    License: VinAI Research License (Non-commercial)
    """
    print("\n" + "="*60)
    print("📥 [5/5] Tải PhoMT — Dữ liệu dịch Vi-En")
    print("="*60)
    
    save_path = DATA_DIR / "mt" / "phomt"
    
    try:
        from datasets import load_dataset
        
        print("  ⏳ Đang tải (3M cặp câu, có thể mất 15-30 phút)...")
        ds = load_dataset("ura-hcmut/PhoMT", cache_dir=str(save_path))
        
        print(f"  ✅ Tải xong!")
        for split_name, split_data in ds.items():
            print(f"     - {split_name}: {len(split_data)} mẫu")
        
        return True
    except Exception as e:
        print(f"  ❌ Lỗi: {e}")
        print("  💡 Thử tải từ GitHub: git clone https://github.com/VinAIResearch/PhoMT")
        return False

def print_summary(results):
    """In bảng tổng kết"""
    print("\n" + "="*60)
    print("📊 TỔNG KẾT TẢI DỮ LIỆU")
    print("="*60)
    
    for name, success in results.items():
        status = "✅ OK" if success else "❌ Lỗi"
        print(f"  {status}  {name}")
    
    print("\n" + "-"*60)
    print("📌 BƯỚC TIẾP THEO:")
    print("  1. Tải noise datasets (MUSAN): python scripts/download_noise.py")
    print("  2. Tải OPUS Ko-Vi: python scripts/download_opus.py")  
    print("  3. Kiểm tra data: python scripts/inspect_data.py")
    print("  4. Tiền xử lý: python scripts/preprocess_data.py")

def main():
    print("🚀 OneVoice — Bắt đầu tải dữ liệu từ HuggingFace")
    print(f"📁 Thư mục lưu: {DATA_DIR}")
    print()
    
    # Tạo thư mục
    print("📁 Tạo cấu trúc thư mục...")
    ensure_dirs()
    
    # Hỏi người dùng muốn tải gì
    print("\n" + "="*60)
    print("Chọn datasets muốn tải:")
    print("  1. ViMD (ASR đa giọng, ~5GB)")
    print("  2. FLEURS Vi (ASR benchmark, ~1GB)")
    print("  3. Common Voice Vi (ASR, ~3GB, cần HF login)")
    print("  4. FLORES-200 (MT benchmark, nhẹ ~50MB)")
    print("  5. PhoMT (MT Vi-En, ~2GB)")
    print("  a. Tải TẤT CẢ")
    print("  q. Thoát")
    print("="*60)
    
    choice = input("\nChọn (1-5, a, q): ").strip().lower()
    
    results = {}
    
    if choice == 'q':
        print("👋 Thoát.")
        return
    
    if choice in ('1', 'a'):
        results["ViMD"] = download_vimd()
    if choice in ('2', 'a'):
        results["FLEURS Vi"] = download_fleurs_vi()
    if choice in ('3', 'a'):
        results["Common Voice Vi"] = download_common_voice_vi()
    if choice in ('4', 'a'):
        results["FLORES-200"] = download_flores()
    if choice in ('5', 'a'):
        results["PhoMT"] = download_phomt()
    
    if not results and choice not in ('1','2','3','4','5','a'):
        print(f"  ⚠️ Lựa chọn '{choice}' không hợp lệ. Chạy lại script.")
        return
    
    print_summary(results)

if __name__ == "__main__":
    main()
