"""
Script để tải các file .bin (MeloTTS-ZH) lên Qualcomm AI Hub
và chạy Profile (Đo đạc Latency/Bộ nhớ) trên IQ-9075 EVK.
Sau khi chạy, script sẽ in ra các URL để bạn xem kết quả trên Web.
"""

import qai_hub as hub
from pathlib import Path
import time

def submit_and_get_url(name, model_path, device_name="Dragonwing IQ-9075 EVK"):
    print(f"\n🚀 Đang tải lên và submit job cho: {name}")
    try:
        # Tải mô hình QNN (.bin) lên AI Hub
        model = hub.upload_model(model_path)
        
        # Submit Profile Job
        device = hub.Device(device_name)
        profile_job = hub.submit_profile_job(
            model=model,
            device=device,
            name=f"MeloTTS_{name}_Profile"
        )
        print(f"✅ Đã gửi! Xem trực tiếp tại Web:")
        print(f"🔗 {profile_job.url}")
        return profile_job
    except Exception as e:
        print(f"❌ Lỗi khi submit {name}: {e}")
        return None

def main():
    base_dir = Path("melotts_zh-voice_ai-mixed_with_float-qualcomm_qcs9075")
    
    if not base_dir.exists():
        print("Không tìm thấy thư mục tải về. Vui lòng chạy lệnh fetch trước.")
        return

    # Lấy 3 submodels quan trọng
    models_to_profile = {
        "Encoder": base_dir / "encoder.bin",
        "Flow": base_dir / "flow.bin",
        "Decoder_Vocoder": base_dir / "decoder.bin",
    }

    jobs = {}
    for name, path in models_to_profile.items():
        if path.exists():
            job = submit_and_get_url(name, path)
            jobs[name] = job
            time.sleep(2) # Đợi 1 chút tránh bị limit
        else:
            print(f"⚠️ Không tìm thấy file {path}")

    print("\n🎉 XONG! Bạn hãy click vào các link trên để mở giao diện Qualcomm AI Hub Workbench.")
    print("Tại đó, bạn sẽ thấy thiết bị mô phỏng IQ-9075 EVK đang chạy model của chúng ta!")

if __name__ == "__main__":
    main()
