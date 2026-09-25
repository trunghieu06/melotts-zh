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
Số_sample_hợp_lệ = Số_frame_thật * hop_size  (với hop_size = 512)
```

---

## 🏃 5. Hướng dẫn Chạy Mô hình Lượng tử hóa (Inference Guide)

Để chạy suy luận trực tiếp mô hình đã lượng tử hóa (`decoder` W8A16, `flow` UINT16, `encoder`), sử dụng script [**`infer_quantized.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/infer_quantized.py):

### 🔊 1. Chạy suy luận một câu tiếng Trung bất kỳ:
```bash
python infer_quantized.py --text "你好，欢迎体验高通量化语音合成系统。" --output output.wav
```

### 📄 2. Chạy hàng loạt từ một tệp văn bản (`.txt`):
```bash
python infer_quantized.py --file eval_dataset/baker_500_eval.txt
```

### ⚙️ 3. Các tùy chọn dòng lệnh nâng cao:
| Tham số | Ý nghĩa | Mặc định |
| :--- | :--- | :---: |
| `--text`, `-t` | Câu văn bản tiếng Trung cần tổng hợp | Câu mẫu chào mừng |
| `--output`, `-o` | Tệp âm thanh `.wav` đầu ra | `output_quantized.wav` |
| `--npu-native` | **100% NPU-Native Pipeline:** Kích hoạt Binary Stencil Alignment, In-NPU Chunking & In-Graph Trimming (Problem 2, 3, 4 theo `MeloTTS.pdf`), loại bỏ hoàn toàn CPU loops | `False` |
| `--encoder-mode` | `fp32` (Chuẩn Qualcomm - 99.39% độ trung thực) hoặc `quantized` (W8A16) | `fp32` |
| `--speed` | Điều chỉnh tốc độ nói (ví dụ `1.2` để nói nhanh hơn) | `1.0` |
| `--file`, `-f` | Đọc danh sách câu từ tệp `.txt` | `None` |

Ví dụ chạy chế độ thuần NPU-Native:
```bash
python infer_quantized.py --text "你好，欢迎体验高通量化语音合成系统。" --npu-native --output output_npu_native.wav
```

### 📊 4. Chạy kiểm định đo đạc độ tương đồng (Benchmark):
Đối chiếu độ trung thực (% so với mô hình FP32 gốc) trên tập dữ liệu kiểm thử chuẩn 500 câu Baker CSMSC:
```bash
python benchmark_eval_dataset.py
```
*(Kết quả đánh giá và biểu đồ phân bổ được lưu tại `output_eval/benchmark_stitched/stitched_benchmark_report.json`)*

---

## 💻 6. Hướng dẫn Thiết lập & Tiếp tục trên Máy Khác (Setup on a New Machine)

Khi clone dự án về một máy tính mới (Linux/macOS), hãy làm theo các bước chuẩn mực dưới đây:

### Bước 1: Clone kho mã nguồn
```bash
git clone https://github.com/trunghieu06/melotts-zh.git
cd melotts-zh
```

### Bước 2: Tạo và kích hoạt môi trường ảo Python (Python 3.10 hoặc 3.11)
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt các gói phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ./MeloTTS
python -m unidic download
```

### Bước 4: Tạo lại các Submodel ONNX & NPU-Native Modules
Do các tệp nhị phân ONNX lớn không được lưu trực tiếp trên Git, bạn có thể tự sinh lại ngay lập tức:
```bash
# 1. Bóc tách 4 Submodel MeloTTS chuẩn
python export_onnx.py

# 2. Xuất các Module NPU-Native (Problem 2 & Problem 4)
python export_npu_native.py
```

### Bước 5: (Tùy chọn) Kiểm tra & Nộp job biên dịch lên Qualcomm AI Hub
```bash
qai-hub configure --api_token <YOUR_QUALCOMM_AI_HUB_API_TOKEN>
python verify_ai_hub_npu_native.py
```

---

## 🗂️ 7. Danh mục Toàn bộ Mã nguồn & Chức năng Từng File

| Tên tệp mã nguồn | Vai trò & Chức năng chính | Lệnh thực thi mẫu |
| :--- | :--- | :--- |
| [**`infer_quantized.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/infer_quantized.py) | **Script suy luận chính:** Hỗ trợ cả chế độ lai (Hybrid) và chế độ 100% NPU-Native (`--npu-native`), điều phối toàn bộ pipeline và xuất file WAV. | `python infer_quantized.py --npu-native` |
| [**`export_npu_native.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/export_npu_native.py) | **Xuất ONNX NPU-Native:** Xuất đồ thị tĩnh tự chứa cho `NPUDurationExpansion` (Problem 2) và `NPUArtifactTrimmer` (Problem 4). | `python export_npu_native.py` |
| [**`verify_ai_hub_npu_native.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/verify_ai_hub_npu_native.py) | **Kiểm định Qualcomm AI Hub:** Tự động upload, compile QNN DLC và profile P2 & P4 trên thiết bị Dragonwing IQ-9075 EVK. | `python verify_ai_hub_npu_native.py` |
| [**`npu_engine/`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/npu_engine/) | **Thư viện Toán tử NPU Tĩnh:** Chứa các module PyTorch thuần HTP Whitelist: `duration_expansion.py`, `chunking.py`, `trimming.py`, và bài test đối chiếu `test_pipeline_parity.py`. | `python -m npu_engine.test_pipeline_parity` |
| [**`benchmark_eval_dataset.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/benchmark_eval_dataset.py) | **Đánh giá độ tương đồng:** Ghép nối 4 submodels, đối chiếu với mô hình PyTorch FP32 gốc trên tập Baker CSMSC, tính Cosine Similarity & MCD. | `python benchmark_eval_dataset.py` |
| [**`generate_quick_test.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/generate_quick_test.py) | **Sinh mẫu đối chiếu:** Tự động tạo 10 câu âm thanh đối chiếu 3 chiều (FP32 Gốc vs Full-Quantized vs Standard-Quantized) vào `quick_test_full_quantized/`. | `python generate_quick_test.py` |
| [**`export_onnx.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/export_onnx.py) | **Xuất mô hình ONNX:** Bóc tách mô hình nguyên khối PyTorch thành 4 tệp `.onnx` theo đúng chuẩn giao diện phần cứng Qualcomm NPU (`metadata.json`). | `python export_onnx.py` |
| [**`prepare_calibration_data.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/prepare_calibration_data.py) | **Tạo dữ liệu hiệu chuẩn (PTQ):** Trích xuất các mẫu câu tiếng Trung thực tế từ Baker CSMSC để tạo file nén `.npz` nạp cho AI Hub Quantizer. | `python prepare_calibration_data.py` |
| [**`prepare_encoder_calib.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/prepare_encoder_calib.py) | **Hiệu chuẩn Encoder:** Chuẩn bị tập dữ liệu hiệu chuẩn riêng biệt cho khối Text Encoder. | `python prepare_encoder_calib.py` |
| [**`prepare_eval_data.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/prepare_eval_data.py) | **Chuẩn bị ngữ liệu kiểm thử:** Bóc tách 500 câu chuẩn từ Baker CSMSC vào `eval_dataset/baker_500_eval.txt`. | `python prepare_eval_data.py` |
| [**`quantize_bert_iq9075.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/quantize_bert_iq9075.py) | **Lượng tử hóa BERT:** Nộp job lượng tử hóa INT8 (`w8a8`) và biên dịch NPU cho `bert_wrapper` lên Qualcomm AI Hub (IQ-9075 EVK). | `python quantize_bert_iq9075.py` |
| [**`quantize_encoder_iq9075.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/quantize_encoder_iq9075.py) | **Lượng tử hóa Encoder:** Nộp job lượng tử hóa W8A16 cho `encoder` lên Qualcomm AI Hub. | `python quantize_encoder_iq9075.py` |
| [**`run_quantize_and_eval_iq9075.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/run_quantize_and_eval_iq9075.py) | **Quy trình tổng thể trên AI Hub:** Tự động upload, quantize, compile, profile và tải về các artifact `.dlc` / `.onnx`. | `python run_quantize_and_eval_iq9075.py` |
| [**`run_fp32_baseline.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/run_fp32_baseline.py) | **Sinh dữ liệu gốc FP32:** Sinh trọn vẹn 500 file audio chuẩn FP32 đối chứng tại `output_eval/wav_fp32/`. | `python run_fp32_baseline.py` |
| [**`run_quant_baseline.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/run_quant_baseline.py) | **Kiểm tra sơ bộ lượng tử hóa:** Chạy kiểm thử nhanh một số câu với các file lượng tử hóa ban đầu. | `python run_quant_baseline.py` |
| [**`eval.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/eval.py) | **Hàm tính toán chỉ số âm học:** Thư viện tính Mel-Cepstral Distortion (MCD) dùng FastDTW và Cosine Similarity phổ Mel. | `python eval.py` |
| [**`generate_handtest.py`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/generate_handtest.py) | **Tạo file nghe thủ công:** Sinh một số câu thử nghiệm đặc thù để kiểm tra bằng tai. | `python generate_handtest.py` |

---

## 📚 8. Các Báo cáo Kỹ thuật Chuyên sâu trong Repository

* [**`report_melotts_zh.md`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/report_melotts_zh.md): Báo cáo kỹ thuật tổng thể, phân tích kiến trúc, rào cản phần cứng NPU, kết quả lượng tử hóa trên AI Hub, và báo cáo đột phá kiến trúc 100% NPU-Native (Mục 6.5).
* [**`cpu_npu_ratio_analysis.md`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/cpu_npu_ratio_analysis.md): Phân tích chi tiết tỷ lệ tải trọng **95% NPU vs 5% CPU**, giải thích 5 lý do tại sao kiến trúc ban đầu phụ thuộc vào CPU Host.
* [**`MeloTTS.pdf`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/MeloTTS.pdf): Tài liệu nghiên cứu chuyên sâu về kiến trúc và 4 bài toán chuyển đổi mô hình TTS sang 100% NPU-Native.
* [**`quick_test_full_quantized/README.md`**](file:///Users/htti/Documents/Code.nosync/melotts-zh/quick_test_full_quantized/README.md): Bảng đối chiếu thời lượng và tai nghe của 10 câu mẫu (FP32 vs Full-Quantized vs Standard-Quantized).
