"""
Tải kết quả Profile từ AI Hub về Terminal
Chạy: python scripts/get_profile_results.py
"""

import qai_hub as hub

def main():
    print("⏳ Đang kết nối tới Qualcomm AI Hub để lấy kết quả đo đạc...\n")
    
    # Lấy 3 job Profile gần nhất của bạn
    jobs = hub.get_jobs(limit=3)
    
    for job in jobs:
        if not isinstance(job, hub.client.ProfileJob):
            continue
            
        print(f"=========================================")
        print(f"📦 Model: {job.name}")
        print(f"📱 Device: {job.device.name}")
        print(f"Trạng thái: {job.get_status().name}")
        
        if job.get_status().name == "SUCCESS":
            profile_data = job.download_profile()
            
            # Lấy thông số Inference Time
            # Cấu trúc JSON trả về tuỳ thuộc vào chip, thường nằm trong 'execution_detail'
            try:
                # Tìm thời gian chạy trên NPU (Estimated Inference Time)
                inference_time = profile_data.get('execution_summary', {}).get('estimated_inference_time', 'N/A')
                print(f"⏱️  Thời gian chạy (Inference): {inference_time} micro-giây")
                
                # Tìm bộ nhớ
                peak_memory = profile_data.get('execution_summary', {}).get('peak_memory_bytes', 'N/A')
                if peak_memory != 'N/A':
                    print(f"🧠 Bộ nhớ Peak: {int(peak_memory) / 1024 / 1024:.2f} MB")
            except Exception as e:
                print("Đã có kết quả, nhưng cần xem chi tiết dạng JSON.")
        else:
            print("⏳ Job đang chạy hoặc bị lỗi, vui lòng chờ thêm...")
            
    print("\n=========================================")
    print("💡 Lưu ý: Thời gian tính bằng micro-giây (1000 micro-giây = 1 mili-giây).")

if __name__ == "__main__":
    main()
