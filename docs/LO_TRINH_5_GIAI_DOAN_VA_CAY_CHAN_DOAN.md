# TÀI LIỆU KỸ THUẬT: LỘ TRÌNH 5 GIAI ĐOẠN (5-STAGE ROADMAP) VÀ CÂY CHẨN ĐOÁN
*(Chuẩn hóa quy trình R&D mô hình MeloTTS Byte-Encoder trên Qualcomm NPU)*

---

## 1. Bảng Đối Chiếu: Lộ Trình Cũ vs Lộ Trình 5 Giai Đoạn Mới

| Giai đoạn | Lộ trình cũ (3 bước cảm tính) | Lộ trình mới (5 giai đoạn khoa học) | Tiêu chí vượt qua (Exit Gate) |
| :---: | :--- | :--- | :--- |
| **Giai đoạn 0** | *Không có (trích xuất ngay)* | **Kiểm định Teacher & Chốt cấu hình:**<br>• Khóa đúng `spk2id` (ID=1).<br>• Đo phổ & std của $z$ theo `noise_scale` $\in \{0.0, 0.3, 0.667\}$.<br>• Đo năng lượng nền $z=0$ (dB thật).<br>• Nghe kiểm tra Teacher Baseline. | Teacher phát âm tự nhiên, không rè, cấu hình $z$ được chốt cứng 100%. |
| **Giai đoạn 1** | Overfit 5–10 câu (200 epochs), kỳ vọng "chứng minh căn chỉnh". | **Overfit Kiểm Chứng Pipeline ($\ge 1.000$ bước):**<br>• Có LR Warmup.<br>• Đo $\text{NMSE}$, chỉ số Buzz Ratio (tại $258.4\text{ Hz}, 430.7\text{ Hz}$).<br>• Nghe bản Raw 512 frames và bản fade-out theo $L_{\text{est}}$. | NMSE $< 0.05$, Buzz Ratio $\approx 1.0$ (không có đỉnh cộng hưởng nhân tạo). |
| **Giai đoạn 2** | Train 1.000 câu, chỉ nhìn train loss. | **Thí nghiệm Learning Curve (500 $\to$ 3.000 câu):**<br>• Tập Validation cố định và khử trùng n-gram.<br>• Khớp bộ ước lượng độ dài tuyến tính $L_{\text{est}} = a \cdot N_{\text{chars}} + b$.<br>• Vẽ bản đồ Cross-Attention (kiểm tra vệt chéo căn chỉnh). | Val NMSE giảm đều khi tăng dữ liệu, Attention xuất hiện vệt chéo đơn điệu. |
| **Giai đoạn 3** | Đoán thời gian 35 phút cho 20k câu. | **Mở Rộng Dữ Liệu Lớn ($\ge 20.000$ câu):**<br>• Trích xuất trực tiếp trên GPU bằng `infer` theo lô.<br>• Cache lưu dưới dạng `float16` theo shard (tiết kiệm 75% RAM). | Val NMSE tiệm cận giới hạn trên của Teacher. |
| **Giai đoạn 4** | Xuất ONNX từ RAM epoch cuối. | **Xuất ONNX & Kiểm Định Triển Khai NPU:**<br>• Nạp đúng Checkpoint `best.pth`.<br>• So khớp số học ONNX vs PyTorch (`assert_allclose`).<br>• Đo độ nhạy lượng tử hóa FP16 trước khi nạp EVK. | Sai số ONNX $< 10^{-4}$, không có toán tử rơi xuống CPU trên HTP v73. |

---

## 2. Cây Chẩn Đoán Khi Giai Đoạn 1 Còn Méo Hoặc Buzz (Diagnostic Tree)

Khi chạy Giai đoạn 1 (Overfit), nếu âm thanh phát ra chưa như ý, **tuyệt đối không vội kết luận kiến trúc sai**, mà đi tuần tự theo 5 bước chẩn đoán sau:

```
[BƯỚC 1: Kiểm tra Teacher Baseline]
  ├─ Âm thanh Teacher có bị rè/bíp không?
  │    ├─ CÓ: Lỗi nằm ở cấu hình Teacher (sai spk2id, noise_scale quá thấp làm tụt std, hoặc tắt BERT làm lệch phân phối).
  │    │      ==> KHẮC PHỤC Ở TEACHER TRƯỚC, KHÔNG TRAIN TIẾP STUDENT.
  │    └─ KHÔNG: Teacher hoàn toàn sạch ==> Chuyển sang Bước 2.
  │
[BƯỚC 2: Kiểm tra NMSE trên tập câu train]
  ├─ NMSE có thấp không (< 0.05)?
  │    ├─ CAO: Mạng chưa hội tụ ==> Tăng số bước (1.500 bước), kiểm tra LR warmup, gradient clipping.
  │    └─ THẤP: Mạng đã khớp số liệu tốt ==> Chuyển sang Bước 3.
  │
[BƯỚC 3: Xác định vị trí tiếng ồn (Vùng nói vs Vùng đuôi)]
  ├─ Tiếng ồn/buzz chỉ xuất hiện ở nửa sau câu (vùng padding)?
  │    ├─ CÓ: Vấn đề padding vocoder bias (z=0 vẫn phát âm nền ~-25dB).
  │    │      ==> Khắc phục bằng fade-out theo độ dài ước lượng L_est.
  │    └─ KHÔNG: Tiếng ồn nằm ngay trong lúc đang nói ==> Chuyển sang Bước 4.
  │
[BƯỚC 4: Kiểm tra Jitter chu kỳ frame]
  ├─ z_pred có bị rung lắc giữa các frame liền kề không?
  │    ├─ CÓ: Vocoder bị kích thích bởi bước nhảy frame 86.13 Hz.
  │    │      ==> Thêm Post-net Conv1d (kernel=5) hoặc hàm mất mát vi phân thời gian (delta loss).
  │    └─ KHÔNG: z_pred mượt mà ==> Chuyển sang Bước 5.
  │
[BƯỚC 5: Đánh giá dung lượng kiến trúc]
  └─ Chỉ khi cả 4 bước trên đều thỏa mãn mà âm thanh vẫn méo, mới kết luận kiến trúc thiếu dung lượng 
     (Cần tăng d_model lên 512, tăng số layer hoặc thêm giám sát căn chỉnh explicit).
```
