# HƯỚNG DẪN TOÀN DIỆN: GIẢI MÃ KIẾN TRÚC MÔ HÌNH VÀ LỘ TRÌNH KIẾN THỨC BẠN CẦN BIẾT
*(Tài liệu dành cho kỹ sư dự án: Đi từ con số 0 đến làm chủ hệ thống MeloTTS trên Qualcomm NPU)*

---

## 1. Bức Tranh Toàn Cảnh: Chúng Ta Đang Làm Gì?

### Mục tiêu dự án
Tạo ra một hệ thống **Chuyển văn bản thành giọng nói (Text-to-Speech - TTS)** tiếng Trung có khả năng:
1. Chạy **100% trên phần cứng NPU (Neural Processing Unit)** của Qualcomm (chip Snapdragon X-Elite / Dragonwing IQ-9075 EVK).
2. **Không dùng CPU** để tính toán AI (0% CPU fallback), giúp tiết kiệm pin tối đa và đạt tốc độ phản hồi thời gian thực (< 50ms).
3. Đọc được chữ Hán tự nhiên mà không cần các thư viện từ điển cồng kềnh.

### Tại sao bài toán này cực kỳ khó trên NPU?
* **CPU/GPU thông thường:** Rất linh hoạt. Bạn có thể viết vòng lặp `for`, câu lệnh điều kiện `if/else`, xử lý câu dài ngắn tùy ý (dynamic shape), dùng từ điển tra từ vựng.
* **Qualcomm NPU (Hexagon HTP):** Hoạt động giống như một **dây chuyền sản xuất cơ khí cố định**. 
  * Nó chỉ chấp nhận **kích thước cố định (Static Graph)**: Ví dụ đầu vào bắt buộc luôn là 128 tokens, đầu ra bắt buộc luôn là 512 frames.
  * Nó **cấm tuyệt đối** các toán tử động: Không vòng lặp `while`, không chia nhánh `if`, không dynamic slicing.
  * Nếu mô hình có bất kỳ toán tử nào không nằm trong danh sách hỗ trợ (Whitelist), NPU sẽ "ném" toán tử đó về cho CPU xử lý $\rightarrow$ Tốc độ chậm đi 20 đến 50 lần!

---

## 2. Kiến Trúc MeloTTS Gốc (Teacher Model) Hoạt Động Ra Sao?

MeloTTS thuộc họ mô hình **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)**. Quy trình tạo ra âm thanh của nó gồm 5 chặng liên tiếp:

```
[Văn bản chữ Hán] 
       │
       ▼
[Chặng 1: G2P + BERT] ──> Chuyển chữ Hán thành chuỗi âm vị (Phonemes) + vector ngữ cảnh
       │
       ▼
[Chặng 2: Text Encoder (enc_p)] ──> Nén âm vị thành vector tiềm ẩn ngữ nghĩa m_p (std ≈ 0.04)
       │
       ▼
[Chặng 3: Duration Predictor (DP)] ──> Dự đoán mỗi âm vị cần đọc trong bao nhiêu mili-giây
       │
       ▼
[Chặng 4: Flow (Normalizing Flow)] ──> Dãn nở m_p theo thời lượng thành ma trận âm học z (std ≈ 0.94)
       │
       ▼
[Chặng 5: HiFi-GAN Vocoder (dec)] ──> Biến ma trận z thành sóng âm thanh nghe được (.wav)
```

### Tại sao kiến trúc gốc này "thất bại" trên NPU?
1. **G2P (Grapheme-to-Phoneme) & BERT:** Chạy bằng Python thuần trên CPU, dùng từ điển tra cứu `jieba` và mô hình BERT khổng lồ (110 triệu tham số) $\rightarrow$ Không thể nhét lên NPU.
2. **Duration Predictor (Bộ dự đoán thời lượng):** Mỗi câu nói có số âm vị khác nhau và thời lượng đọc mỗi từ là một con số ngẫu nhiên biến thiên. Việc lặp lại một âm vị theo số frame dự đoán đòi hỏi thuật toán căn chỉnh động (Dynamic Alignment / While-loop) $\rightarrow$ NPU từ chối biên dịch.
3. **Normalizing Flow:** Chứa các tầng WaveNet nghịch đảo với cấu trúc đệ quy $\rightarrow$ Rất khó tối ưu trên NPU.

**Kết quả:** Nếu cố chạy MeloTTS gốc, chỉ có Chặng 5 (Vocoder) chạy được trên NPU (mất 21ms), còn Chặng 1 đến 4 bị đẩy về CPU chạy mất hơn **800ms – 1.5 giây** $\rightarrow$ Hệ thống bị nghẽn hoàn toàn.

---

## 3. Kiến Trúc OneVoice Student: "Đường Tắt Đồ Thị Tĩnh"

Để giải quyết bài toán trên, ý tưởng đột phá của kiến trúc **OneVoice Byte-Encoder** là:
> **Vứt bỏ toàn bộ Chặng 1, 2, 3, 4!** Thay thế tất cả bằng **DUY NHẤT MỘT MẠNG TRANSFORMER TĨNH** huấn luyện theo phương pháp chưng cất tri thức (Knowledge Distillation).

```
              ┌────────────────────────────────────────────────────────┐
              │           MÔ HÌNH STUDENT DUY NHẤT (NPU)               │
              │                                                        │
[Chuỗi Byte] ──> [Text Encoder] ────> [Cross-Attention] ──> Ma trận z  │ ──> [Vocoder NPU] ──> Sóng Âm
[1, 128] int32│  + Positional Emb      ▲ (512 Frame Queries) [1,192,512]│     (decoder.bin)     (.wav)
              │                        │                               │
              │             [512 Learned Queries]                      │
              └────────────────────────────────────────────────────────┘
```

### Chi tiết từng khối trong mô hình Student:

#### 1. Chuỗi Byte UTF-8 (`[1, 128]` int32)
* Thay vì cần thư viện G2P để phân tích chữ Hán sang Pinyin, ta chuyển thẳng câu tiếng Trung sang chuỗi byte UTF-8 (mỗi chữ Hán là 3 bytes, từ 0 đến 255).
* Máy tính chỉ cần bảng mã ASCII/UTF-8 tiêu chuẩn, không cần bất kỳ từ điển nào.

#### 2. Text Encoder + Positional Encoding
* Chuỗi byte được đưa qua tầng `nn.Embedding(256, 384)` để biến mỗi byte thành 1 vector 384 chiều.
* **Positional Encoding:** Cộng thêm vector vị trí để mô hình biết byte nào đứng trước, byte nào đứng sau (vì cơ chế Self-Attention bản chất không biết thứ tự nếu không có vị trí).
* Đi qua 4 tầng Self-Attention để tạo ra **Bộ nhớ ngữ cảnh văn bản (Text Memory)**.

#### 3. Bộ giải mã 512 Frame Queries (Giải quyết bài toán co dãn thời gian)
* **Vấn đề hóc búa:** 1 chữ Hán chỉ có 3 bytes, nhưng khi đọc ra loa thì kéo dài khoảng 15 frame âm thanh (0.17 giây). Làm sao để từ 3 bytes nở ra thành 15 frames mà không dùng vòng lặp động?
* **Giải pháp Frame Queries:**
  * Mô hình được cấp sẵn **512 "Câu hỏi khung giờ" (Frame Queries)** cố định: Query 0 đại diện cho mili-giây thứ 0, Query 1 đại diện cho mili-giây thứ 11.6, ..., Query 511 đại diện cho mili-giây thứ 5,900.
  * Từng Query này sẽ nhìn vào Bộ nhớ ngữ cảnh văn bản thông qua **Cross-Attention**:
    * Query ở 0.5 giây đầu sẽ tự hỏi: *"Ở thời điểm này tôi phải đọc ký tự nào trong câu?"* $\rightarrow$ Nó chú ý vào byte của chữ thứ nhất.
    * Query ở 1.5 giây sau sẽ tự hỏi: *"Bây giờ đến chữ nào?"* $\rightarrow$ Nó chú ý vào byte của chữ thứ hai.
  * **Toàn bộ phép tính này là phép nhân ma trận thuần túy (Static MatMul)**, hoàn toàn tương thích 100% với tập lệnh của Qualcomm NPU!

#### 4. Đầu ra Ma trận Tiềm ẩn Âm học $z$ (`[1, 192, 512]`)
* 512 Frame Queries sau khi gom đủ thông tin từ văn bản sẽ được chiếu qua một tầng Linear để trở thành ma trận $z$ kích thước `[1, 192, 512]`.
* Ma trận này được nạp thẳng vào Vocoder `decoder.bin` trên NPU để xuất ra âm thanh.

---

## 4. Giải Phẫu Vấn Đề "Tiếng Bíp Chói Tai 430.7 Hz"

Trong lần thử nghiệm đầu tiên, mô hình phát ra tiếng rít cực kỳ khó chịu ở tần số 430.7 Hz. Đây là bài học kinh điển về xử lý tín hiệu âm thanh số:

### Nguyên nhân 1: Nhầm lẫn giữa Không gian $m_p$ và Không gian $z$
* **$m_p$ (Text Space):** Là đặc trưng ngôn ngữ trừu tượng, có độ lệch chuẩn rất nhỏ ($\text{std} \approx 0.04$).
* **$z$ (Acoustic Latent Space):** Là đặc trưng âm thanh thực sự sau khi đã đi qua bộ biến đổi Flow, có độ dao động lớn ($\text{std} \approx 0.94$).
* Ở bản cũ, tác giả lấy nhầm nhãn $m_p$ ép Student học. Khi nạp một ma trận phẳng lì ($\text{std} \approx 0.04$) vào Vocoder, các tầng tích chập chuyển vị (Transposed Convolutions) của Vocoder bị "đói biên độ", dẫn đến hiện tượng **tự kích dao động cộng hưởng (Harmonic Self-Oscillation)**.
* Tần số khung hình của Vocoder là:
  $$f_{\text{frame}} = \frac{44,100\text{ Hz}}{512} = 86.13\text{ Hz}$$
  Vocoder tự cộng hưởng ở hài âm bậc 5:
  $$86.13\text{ Hz} \times 5 = 430.65\text{ Hz} \approx 430.7\text{ Hz}$$
  $\rightarrow$ Tạo ra tiếng bíp chói tai liên tục!

### Nguyên nhân 2: Lỗi ép vị trí 1:1 (`min_len` bug)
* Ở bản cũ, người lập trình viết: `loss = (student[:, :, :min_len] - teacher[:, :, :min_len])**2`.
* Việc này ép Byte thứ 1 phải khớp với Frame thứ 1, Byte thứ 2 khớp với Frame thứ 2... Trong khi thực tế Byte thứ 1 phải tương ứng với các Frame từ 0 đến 15. Kết quả là mô hình học sai hoàn toàn ngữ âm.

### Nguyên nhân 3: Vùng Padding bị thả nổi
* Một câu nói thường dài khoảng 150–300 frames. Từ frame 300 đến 512 là vùng đệm (padding).
* Nếu không ép vùng padding này về 0, mạng Student sẽ sinh ra các giá trị ngẫu nhiên ở đuôi. Khi nạp vào Vocoder, phần đuôi phát ra tiếng rè hoặc tiếng buzz chu kỳ frame.

---

## 5. Những Cải Tiến Đã Đưa Vào Notebook V3

1. **Bắt đúng đích ma trận $z$:** Bắt hook vào hàm `dec(z, g=g)` của Teacher, lấy đúng ma trận sau Flow có $\text{std} \approx 0.94$.
2. **Khử hoàn toàn tính ngẫu nhiên:** Gọi Teacher với `noise_scale=0.0`, `noise_scale_w=0.0`, `sdp_ratio=0.0` để ma trận $z$ làm nhãn là hoàn toàn tất định (không bị nhiễu Gaussian).
3. **Bảo vệ Hook bằng `call_count`:** Đảm bảo mỗi câu chỉ gọi Vocoder đúng 1 lần, loại bỏ hoàn toàn việc câu bị cắt nhỏ thành nhiều mảnh.
4. **Kiểm tra Baseline & Nền Vocoder khi $z=0$:** Đo trực tiếp năng lượng của Vocoder và áp dụng kỹ thuật **Fade-out tuyến tính ở đuôi** để triệt tiêu tiếng ồn nền khi câu kết thúc.
5. **Quy trình 2 pha (Two-Stage Verification):**
   * *Pha 1 (Overfit 5 câu):* Ép mạng học thuộc lòng 5 câu trong 500 bước để chắc chắn Cross-Attention hoạt động và nghe thử ngay.
   * *Pha 2 (Full Training & Validation):* Huấn luyện diện rộng và đối chiếu song song audio của Teacher vs Student trên câu chưa học.
6. **So khớp số học ONNX vs PyTorch:** Đảm bảo file `.onnx` xuất ra chạy trên ONNX Runtime cho kết quả sai số $< 10^{-4}$ so với PyTorch.

---

## 6. Lộ Trình Kiến Thức Bạn Cần Tìm Hiểu (Roadmap)

Để hoàn toàn làm chủ và tự tin phát triển tiếp các bài toán AI âm thanh trên vi xử lý nhúng, đây là 4 khối kiến thức nền tảng bạn nên tích lũy:

### Khối 1: Xử Lý Tín Hiệu Âm Thanh Số (Audio DSP Cơ Bản)
* **Sampling Rate (Tần số lấy mẫu - ví dụ 44.1 kHz):** 1 giây âm thanh được chia thành 44,100 mẫu số.
* **Hop Length & Frame Rate (Bước nhảy khung):** Tại sao lại là con số 512? Vì cứ 512 mẫu âm thanh gộp thành 1 frame (tương đương $44,100 / 512 \approx 86.13$ khung hình/giây).
* **F0 (Fundamental Frequency - Tần số cơ bản):** Cao độ giọng nói con người (nam thường 85–180 Hz, nữ 165–255 Hz). Tiếng bíp 430.7 Hz là âm nhân tạo ngoài dải F0 bình thường.
* **Mel-Spectrogram & STFT:** Cách biểu diễn âm thanh dưới dạng ảnh nhiệt phổ (nhiệt kế tần số theo thời gian).
* 📖 *Tài liệu gợi ý:* Khóa học miễn phí "Audio Signal Processing for Machine Learning" của Valerio Velardo (YouTube).

### Khối 2: Kiến Trúc Transformer & Attention Mechanism
* **Self-Attention:** Từng từ trong câu tương tác với nhau thế nào để hiểu ngữ cảnh.
* **Cross-Attention:** Cơ chế một chuỗi truy vấn (Queries) lấy thông tin từ một chuỗi khóa (Keys/Values). Đây là chìa khóa biến chuỗi ký tự thành chuỗi âm thanh.
* **Positional Encoding:** Tại sao Transformer cần mã hóa vị trí (Learned vs Sinusoidal).
* 📖 *Tài liệu gợi ý:* Bài viết minh họa kinh điển "The Illustrated Transformer" của Jay Alammar.

### Khối 3: Mô Hình Sinh Giọng Nói Hiện Đại (Modern TTS)
* **Kiến trúc VITS (Conditional Variational Autoencoder with Adversarial Learning):** Hiểu rõ vai trò của Text Encoder, Duration Predictor, Normalizing Flow, và Vocoder.
* **HiFi-GAN Vocoder:** Mô hình biến đổi ma trận ẩn/mel-spectrogram thành sóng âm thanh thô bằng các tầng Transposed Convolution và Multi-Period Discriminator.
* **Knowledge Distillation (Chưng cất tri thức):** Cách huấn luyện một mạng Student nhỏ, nhanh bắt chước hành vi của mạng Teacher phức tạp.
* 📖 *Tài liệu gợi ý:* Paper gốc "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech" (Kim et al., 2021).

### Khối 4: Triển Khai AI Trên Chip Nhúng (NPU & Qualcomm AI Hub)
* **Static Graph vs Dynamic Graph:** Tại sao chip NPU đòi hỏi kích thước cố định `[1, 128]` và `[1, 192, 512]`.
* **Toán tử NPU (HTP Whitelist):** Các phép toán được tăng tốc bằng phần cứng DSP/Hexagon Tensor Processor (Conv, MatMul, LayerNorm, GELU) và các phép toán bị cấm (Dynamic Loop, ArgMax, Non-zero).
* **Lượng tử hóa (Quantization):** Sự khác biệt giữa tính toán dấu phẩy động (FP32, FP16) và số nguyên (INT8, INT16). Với mô hình hồi quy tiềm ẩn $z$, FP16 là tiêu chuẩn vàng để không bị vỡ tiếng.
* 📖 *Tài liệu gợi ý:* Qualcomm AI Engine Direct SDK Documentation (QNN Documentation).
