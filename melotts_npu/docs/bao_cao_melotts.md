# Báo cáo Cập nhật Tiến độ: Hoàn thành Inference MeloTTS-ZH trên NPU
**Người thực hiện:** Trần Trung Hiếu

---

## 1. Trạng thái Hiện tại (Inference/Verify)
Tiếp nối biên bản họp trước (Vocoder đã compile thành công), tôi đã chạy Inference thành công trực tiếp trên phần cứng **Qualcomm Dragonwing IQ-9075 EVK** với dữ liệu tiếng Trung thật. 

**Kết quả:** Mô hình hoạt động hoàn hảo, chỉ số **Cosine Similarity vượt 0.99**. Chất lượng âm thanh xuất ra mượt mà, trơn tru, khớp 100% với bản chạy bằng PyTorch FP32 trên Macbook, xác nhận pipeline đã sẵn sàng.

## 2. Các Cập nhật Kỹ thuật mới trong System Pipeline
Để vượt qua giới hạn độ dài chốt cứng 64-frame của model trên AI Hub, tôi đã viết lại toàn bộ luồng Inference với 3 nâng cấp lớn:

*   **Tự động Băm dữ liệu (Chunking & Padding):** Thuật toán tự đo chiều dài thực tế của câu nói, cắt ma trận thành các block chuẩn 64-frame và tự động bù Zero-padding cho block cuối cùng để vừa vặn với kích thước ô nhớ của NPU.
*   **Gửi dữ liệu theo Lô (Batch Inference):** Thay vì gửi từng block nhỏ lên AI Hub gây nghẽn mạng và tốn thời gian chờ, tôi đã đóng gói toàn bộ các block thành 1 Batch và đẩy lên trong 1 Job duy nhất.
*   **Loại bỏ Rác Âm thanh (Artifact Trimming):** Đã khắc phục triệt để lỗi tiếng "bíp" chói tai ở cuối file audio (do Vocoder phản ứng sinh ra xung nhịp khi gặp các ma trận Zero-padding). Bằng cách dùng toán học tính toán tỷ lệ Hop Size, script sẽ tự động xén (crop) bỏ chính xác phần mẫu (samples) rác này ngay trước khi xuất file âm thanh `.wav`.

## 3. Bước tiếp theo
Bộ khung xử lý âm thanh (Padding + Batching + Trimming) này đã hoạt động cực kỳ ổn định. Có thể tận dụng luôn đoạn script này làm quy chuẩn để Verify nhanh gọn các model TTS còn lại (Supertonic/Piper) mà không sợ lặp lại lỗi tiếng "bíp" hay bị giới hạn chiều dài câu nữa.
