# 🚀 MeloTTS-ZH Edge Deployment 

**Mô hình:** MeloTTS (Module Tiếng Trung - ZH)  
**Phần cứng đích:** Qualcomm Dragonwing IQ-9075 EVK (NPU - Hexagon Tensor Processor)

Tài liệu này tổng hợp kiến thức về kiến trúc và chiến lược lượng tử hóa (quantization) nhằm triển khai tối ưu mô hình Text-to-Speech MeloTTS lên thiết bị nhúng (Edge Device), đặc biệt là NPU của Qualcomm. Đây là cẩm nang hữu ích cho quá trình phát triển C++/Python inference pipeline và xử lý sự cố.

---

## 🚧 1. Thách thức trên NPU & Hướng giải quyết

Khác với việc thực thi trên máy chủ GPU, quá trình đưa MeloTTS xuống NPU của Qualcomm (chip IQ-9075) đối mặt với 3 rào cản chính:

1. **Dynamic Shapes (Kích thước mảng biến thiên):** NPU yêu cầu kích thước mảng đầu vào (input tensor shape) phải cố định (Static). Tuy nhiên, độ dài câu nói và âm thanh trong thực tế luôn biến thiên.
2. **Quantization Loss (Suy hao do Lượng tử hóa):** Tính toán trên NPU yêu cầu sử dụng số nguyên (Integer). Việc ép kiểu từ số thực (FP32) xuống 8-bit (INT8) làm mất mát dữ liệu, dẫn đến chất lượng âm thanh đầu ra bị nhiễu nghiêm trọng.
3. **Unsupported Ops (Toán tử không hỗ trợ):** Các toán tử biến đổi mảng phức tạp hoặc mang tính ngẫu nhiên (Stochastic) trong MeloTTS không được NPU Compiler hỗ trợ.

💡 **Giải pháp Kiến trúc:** Thay vì chạy mô hình nguyên khối (Monolithic), dự án áp dụng chiến lược **Phân tách Mô hình (Component-wise Deployment)**.

---

## 🧩 2. Chiến lược Phân tách Mô hình (Splitting Strategy)

Mô hình gốc được bóc tách thành 4 khối chức năng riêng biệt, xuất ra các file QNN Context Binary (`.bin`) độc lập:

### 🅰️ Khối Ngữ cảnh (BERT)
- **Files:** `bert_zh_tokenizer.bin`, `bert_wrapper.bin`, `bert_normalizer.bin`
- **Vai trò:** Trích xuất đặc trưng ngữ nghĩa từ câu tiếng Trung bằng RoBERTa.
- **Chiến lược:** Do khối này có dung lượng lớn (~300MB), ưu tiên chạy trên **CPU hoặc GPU** để tránh chiếm dụng băng thông của NPU. Nếu đủ bộ nhớ, có thể xem xét chạy trên NPU với định dạng INT8.

### 🅱️ Khối Mã hóa Văn bản (Text Encoder)
- **File:** `encoder.bin`
- **Vai trò:** Nhận âm vị, thanh điệu và vector BERT, sau đó chuyển đổi thành phân phối thống kê ẩn.
- **Chiến lược:** Dữ liệu nhỏ, chứa nhiều toán tử chuẩn, có thể chạy mượt mà trên NPU.

### 🅲 Khối Biến đổi Không gian (Normalizing Flow)
- **File:** `flow.bin`
- **Vai trò:** Điều chỉnh phân phối thống kê.
- **Giải quyết nút thắt:** Bằng cách cô lập khối này, bước **Duration Predictor (Dự đoán thời lượng)** được thực thi trên môi trường Python/C++ (chạy trên CPU). Phép nhân bản âm vị (Dynamic Length Expansion) diễn ra tại đây giúp dữ liệu đẩy vào các khối tiếp theo luôn duy trì được tính chất **Static Shape**.

### 🅳 Khối Tạo Sóng âm (Vocoder)
- **File:** `decoder.bin`
- **Vai trò:** Dựa trên kiến trúc HiFi-GAN (ConvTranspose1d), chuyển đổi ma trận ẩn thành sóng âm 24kHz.
- **Chiến lược:** Đây là khối nặng nhất về cường độ tính toán (MACs) và là **đích ngắm chính để offload xuống Hexagon NPU**, tận dụng tối đa sức mạnh 100 TOPS của chip IQ-9075.

---

## 🎛️ 3. Chiến lược Lượng tử hóa (Quantization Recipe)

Để giải quyết bài toán suy giảm chất lượng âm thanh do ép kiểu, dự án áp dụng kỹ thuật **Mixed Precision Quantization (w8a16)** cho khối Vocoder (`decoder.bin`).

- **Weights (Trọng số):** Lượng tử hóa xuống **INT8** để giảm dung lượng file và tiết kiệm RAM.
- **Activations (Kích hoạt):** Giữ ở mức **INT16** (16-bit) khi luồng dữ liệu chảy qua các layer.
- **Kết quả:** Vocoder cực kỳ nhạy cảm với sai số cộng dồn. Việc dùng `w8a16` giúp chỉ số **Cosine Similarity vượt 0.99** so với bản gốc FP32, mang lại âm thanh trong trẻo, không lẫn tiếng ồn.

---

## 🛠️ 4. Thủ thuật xử lý Logic Inference (Inference Pipeline Tricks)

Khi lập trình C++/Python để ghép nối các file `.bin` trên phần cứng, cần đặc biệt lưu ý 2 kỹ thuật sau:

### ✂️ Chunking & Zero-Padding
Để thỏa mãn yêu cầu Static Shape của NPU (ví dụ: NPU chỉ nhận khối 64-frame), dữ liệu truyền vào `decoder.bin` phải được cắt thành các đoạn (chunk) kích thước 64. Ở đoạn cuối cùng nếu thiếu frame, phải **độn thêm số 0 (Zero-Padding)** cho đủ kích thước. Nên gửi dữ liệu theo Batch để tối ưu tốc độ xử lý.

### 🧹 Artifact Trimming (Cắt rác âm thanh)
Khối Vocoder khi xử lý vùng Zero-Padding sẽ sinh ra xung nhịp (thường là các tiếng "bíp" chói tai ở phần đuôi audio). Cần sử dụng công thức sau để xén (crop) bỏ đoạn dư thừa trước khi lưu thành file `.wav` hoàn chỉnh:
```text
Số_sample_hợp_lệ = Số_frame_thật * hop_size
```
