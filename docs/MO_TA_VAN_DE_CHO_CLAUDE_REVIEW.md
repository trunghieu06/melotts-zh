# TÀI LIỆU KỸ THUẬT: ĐẶT VẤN ĐỀ VÀ GIẢI PHÁP NOTEBOOK HUẤN LUYỆN BYTE-ENCODER CHO MELOTTS TRÊN NPU (QUALCOMM HEXAGON)
*(Dùng để gửi cho Claude / Chuyên gia AI kiểm tra và phản biện kiến trúc)*

---

## 1. Bối cảnh dự án & Mục tiêu phần cứng
* **Mục tiêu:** Chạy mô hình Text-to-Speech (MeloTTS-Chinese) **100% Native trên Qualcomm NPU (Hexagon HTP v73)** thuộc vi xử lý Qualcomm Snapdragon X-Elite / Dragonwing IQ-9075 EVK.
* **Rào cản của MeloTTS gốc trên NPU:**
  * MeloTTS sử dụng pipeline phức tạp: `Văn bản -> G2P/Bert -> Text Encoder (enc_p) -> Duration Predictor (DP) -> Flow (Wavenet Bijective) -> HiFi-GAN Vocoder (dec)`.
  * Các khối như Duration Predictor, G2P tiếng Trung, BERT, và các phép căn chỉnh động (dynamic alignment / while loop) **không thể biên dịch thành đồ thị tĩnh trên NPU**, dẫn tới việc CPU phải gánh 90% tính toán, gây nghẽn cổ chai (latency > 800ms).
* **Giải pháp kiến trúc (OneVoice Byte-to-Acoustic Latent):**
  * Thay thế toàn bộ cụm `G2P + BERT + Text Encoder + Duration Predictor + Flow` bằng **duy nhất một mạng Student Transformer tĩnh** (Byte-Encoder).
  * Đầu vào: Chuỗi byte UTF-8 `[1, 512]` (int32).
  * Đầu ra: Ma trận tiềm ẩn âm thanh $z$ `[1, 192, 512]` (float32).
  * Ma trận $z$ này được nạp trực tiếp vào Vocoder NPU (`decoder.bin` - đã compile HTP v73 thành công, chạy mất 21ms trên EVK).
  * **Toàn bộ pipeline trở thành 100% NPU tĩnh, 0% CPU fallback.**

---

## 2. Tử thi kỹ thuật: Tại sao phiên bản huấn luyện cũ thất bại thảm hại?
Trong lần triển khai đầu tiên, checkpoint `byte_encoder_best.pth` khi ghép vào NPU Vocoder chỉ phát ra **tiếng bíp chói tai ở tần số 430.7 Hz và 258.4 Hz (không có tiếng người)**. Phân tích nguyên nhân gốc rễ (Root Cause Analysis):

### Lỗi 1: Sai lệch không gian biểu diễn (Semantic Domain Mismatch)
* **Thực tế mô hình:** Vocoder `dec` của MeloTTS (HiFi-GAN) yêu cầu đầu vào là ma trận $z$ sau Flow ($\text{std} \approx 0.88 - 0.95$, đặc trưng âm học thô).
* **Sai lầm ở notebook cũ:** Tác giả bắt hook vào đầu ra của Text Encoder $m_p$ (`enc_p`, $\text{std} \approx 0.04$, đặc trưng ngôn ngữ trừu tượng) thay vì $z$.
* **Hậu quả:** Khi đưa một tensor có $\text{std} \approx 0.04$ vào Vocoder mà bỏ qua bước Flow kéo giãn mật độ, các tầng Transposed Convolution trong Vocoder bị kích thích vào trạng thái **tự dao động cộng hưởng (harmonic self-oscillation)** ở đúng tần số bước nhảy khung hình:
  $$\text{Tần số cơ sở} = \frac{44,100\text{ Hz}}{512} = 86.13\text{ Hz}$$
  Hài âm bậc 5: $86.13 \times 5 = 430.65\text{ Hz} \approx 430.7\text{ Hz}$ (tiếng bíp chói tai ghi nhận trên phần cứng).

### Lỗi 2: Sai lệch chiều dài và cắt cụt token (`min_len` indexing bug)
* Trong notebook cũ, hàm loss được viết:
  ```python
  min_len = min(L_phonemes, 512)
  loss = F.mse_loss(student_out[:, :, :min_len], teacher_target[:, :, :min_len])
  ```
* Trong khi chuỗi byte dài khoảng 30–50 bytes thì độ dài phoneme $L$ chỉ khoảng 15–25 frames. 
* Hệ quả: Byte số $i$ bị ép học khớp vào Phoneme số $i$ (lệch hoàn toàn ngữ nghĩa), và toàn bộ các frame sau `min_len` nhận gradient bằng 0.

### Lỗi 3: Nghẽn cổ chai CPU (CPU Bottleneck 10 tiếng)
* Notebook cũ gọi `tts_to_file` trên CPU 30,000 lần trực tiếp trong vòng lặp huấn luyện của GPU, khiến GPU chạy cầm chừng 2% công suất và tốn hơn 10 tiếng vô ích.

---

## 3. Thiết kế của Notebook mới (`Train_Byte_Encoder_MeloTTS.ipynb`)
Notebook mới tái cấu trúc toàn diện quy trình:

1. **Bắt đúng đích ma trận $z$:**
   * Sử dụng `register_forward_pre_hook` trên `teacher.model.dec` để lấy chính xác tham số $z$ đầu vào của Vocoder ($\text{std} \approx 0.94$).
2. **Offline Pre-caching (Tăng tốc 100x):**
   * Chạy Teacher MeloTTS trích xuất ma trận $z$ của 1,500 câu tiếng Trung chuẩn (Alpaca-ZH) một lần duy nhất ra tệp `melo_z_dataset_cache.pt`.
   * Vòng lặp huấn luyện sau đó nạp ma trận thẳng từ RAM vào GPU, huấn luyện 25 epochs chỉ mất **2–3 phút**.
3. **Kiến trúc Transformer thuần NPU HTP v73 Whitelist:**
   * 8 Transformer Blocks ($d_{model}=512$, $n_{heads}=8$, $d_{ff}=2048$, GELU activation, LayerNorm).
   * Không dùng các toán tử bị NPU cấm (như Dynamic Slice, Dynamic Reshape, Einops, Non-zero boolean mask).
4. **Hàm mất mát Masked MSE:**
   * Chỉ tính sai số MSE trên các frame âm thanh thực tế (`valid_len`), vùng padding 0 không nhận gradient.
5. **Widget nghe thử trực tiếp trong Notebook:**
   * Chạy suy luận câu thử nghiệm, đẩy qua Vocoder và phát widget `IPython.display.Audio`.
   * Tính toán biến đổi Fourier (FFT) đo đạc trực tiếp các đỉnh tần số để đảm bảo F0 nằm trong dải tiếng người (150–250 Hz) và không có đỉnh 430.7 Hz.
6. **Tự động xuất ONNX tĩnh:**
   * Xuất `melo_byte_encoder_npu.onnx` chuẩn opset 17, sẵn sàng chuyển đổi qua Qualcomm AI Hub.

---

## 4. Các câu hỏi cần Claude / Chuyên gia phản biện & đánh giá

1. **Khả năng căn chỉnh thời lượng nội sinh (Implicit Duration Modeling):**
   * Mô hình Student đi thẳng từ chuỗi byte rời rạc sang ma trận frame âm thanh $z$ kích thước cố định `[1, 192, 512]` mà không có bộ dự đoán độ dài tường minh (Explicit Duration Predictor). Liệu cơ chế Self-Attention trong 8 tầng Transformer có đủ năng lực học sự giãn nở thời gian (time-stretching) giữa số ký tự và số frame âm thanh hay không?
2. **Hàm mất mát (Loss Function):**
   * Hiện tại notebook đang dùng **Masked MSE** giữa $z_{\text{pred}}$ và $z_{\text{target}}$. Có nên bổ sung thêm **Cosine Similarity Loss** hoặc **L1 Loss** để chống lại hiện tượng mờ âm thanh (over-smoothing) đặc trưng của MSE trong miền tiềm ẩn không?
3. **Độ ổn định của Vocoder đối với xấp xỉ $z$:**
   * Vocoder HiFi-GAN vốn rất nhạy cảm với pha và biên độ của $z$. Mức sai số MSE bao nhiêu là ngưỡng an toàn để Vocoder không bị méo tiếng hoặc vỡ âm?
4. **Vấn đề mã hóa UTF-8 cho chữ Hán:**
   * Tiếng Trung mỗi chữ Hán tương ứng 3 bytes UTF-8. Liệu việc mô hình nhận đầu vào là các byte riêng lẻ có học tốt hơn so với việc gom nhóm BPE token không, và chiều dài 512 bytes có đủ đại diện cho câu nói thông thường trên NPU không?
