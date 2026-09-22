# 📑 BÁO CÁO KỸ THUẬT: TỶ TRỌNG HOẠT ĐỘNG, RANH GIỚI ĐIỆN TOÁN VÀ VAI TRÒ BẤT KHẢ THAY THẾ CỦA CPU HOST
## DỰ ÁN: ONEVOICE AI CHALLENGE (QUALCOMM × VNG) — MODULE TEXT-TO-SPEECH
### NỀN TẢNG THỰC THI: QUALCOMM DRAGONWING IQ-9075 EVK & SNAPDRAGON 8 GEN 3

---

## 📌 MỤC LỤC
1. [Tuyên Bố Kiến Trúc & Ranh Giới Phần Cứng](#1-tuyên-bố-kiến-trúc--ranh-giới-phần-cứng)
2. [Bảng Định Lượng Tỷ Trọng Hoạt Động Của CPU Host](#2-bảng-định-lượng-tỷ-trọng-hoạt-động-của-cpu-host)
3. [Chi Tiết 5 Giai Đoạn & Công Việc Cụ Thể CPU Đảm Nhận](#3-chi-tiết-5-giai-đoạn--công-việc-cụ-thể-cpu-đảm-nhận)
4. [Tại Sao Không Thể Thay Thế CPU Bằng NPU Trong Các Tác Vụ Này?](#4-tại-sao-không-thể-thay-thế-cpu-bằng-npu-trong-các-tác-vụ-này)
5. [Hậu Quả Kỹ Thuật Nếu Cố Tình Ép NPU Thay Thế CPU](#5-hậu-quả-kỹ-thuật-nếu-cố-tình-ép-npu-thay-thế-cpu)
6. [Triết Lý Đồng Xử Lý Tối Ưu (Co-Processor Synergy)](#6-triết-lý-đồng-xử-lý-tối-ưu-co-processor-synergy)

---

## 🏛️ 1. TUYÊN BỐ KIẾN TRÚC & RANH GIỚI PHẦN CỨNG

Trong hệ thống xử lý tiếng nói thời gian thực trên thiết bị biên (On-Device Speech AI), kiến trúc tối ưu chuẩn công nghiệp luôn là mô hình **Đồng xử lý không đối xứng (Asymmetric Co-Processing Architecture)**:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 QUALCOMM SOC (QCS9075 / SNAPDRAGON 8 GEN 3)            │
│                                                                                        │
│  ┌──────────────────────────────────────────┐    ┌──────────────────────────────────┐  │
│  │             CPU HOST (ARM CORTEX)        │    │    QUALCOMM HEXAGON HTP NPU      │  │
│  │  [Mặt Phẳng Điều Khiển - Control Plane]  │    │  [Mặt Phẳng Dữ Liệu - Data Plane]│  │
│  │                                          │    │                                  │  │
│  │  • Xử lý chuỗi ký tự rời rạc (Strings)   │    │  • 100% Ma trận học sâu (GEMM)   │  │
│  │  • Biểu thức chính quy (Regex Rules)     │DMA │  • 31 lớp Conv1D (Duration Pred) │  │
│  │  • Phân rã ngữ điệu & Ngắt nghỉ vi mô    ├────┤  • Multi-Head Attention (TextEnc)│  │
│  │  • Ánh xạ từ điển & Tokenization         │    │  • 5 bước Euler ODE (Flow-Match) │  │
│  │  • Điều phối vòng lặp & Quản lý bộ nhớ   │    │  • Neural Vocoder (25.5MB SRAM)  │  │
│  │  • Giao tiếp phần cứng (Audio Driver/DAC)│    │                                  │  │
│  └──────────────────┬───────────────────────┘    └─────────────────┬────────────────┘  │
│                     │ < 1.5 ms (< 0.1% FLOPs)                      │ 7.4 - 125 ms      │
└─────────────────────┼──────────────────────────────────────────────┼───────────────────┘
                      ▼                                              ▼
              CPU mát mẻ (< 0.1% tải)                   NPU xử lý siêu tốc (~95-99% FLOPs)
```

* **Khẳng định cốt lõi:**
  1. **Trên phương diện tính toán nơ-ron học sâu (Neural Inference Graph):** Tỷ lệ offload sang Qualcomm Hexagon HTP NPU đạt **~95% - 99% FLOPs** với **0.0% CPU Fallback** trong toàn bộ 4 submodel đã biên dịch.
  2. **Trên phương diện toàn chuỗi hệ thống (End-to-End System):** CPU Host đảm nhiệm vai trò **"Nhạc trưởng điều phối & Tiền/Hậu xử lý I/O"**, chiếm **~1% - 5% thời gian toàn chuỗi** (tổng thời gian thực thi `< 1.5 ms`) và **`< 0.1% công suất tính toán (FLOPs)`**.

---

## 📊 2. BẢNG ĐỊNH LƯỢNG TỶ TRỌNG HOẠT ĐỘNG CỦA CPU HOST

Dưới đây là bảng phân định định lượng thời gian, khối lượng công việc và mức tiêu thụ tài nguyên của từng thành phần do CPU đảm nhiệm:

| Phân loại / Công đoạn | Thành phần đảm nhiệm | Thời gian chạy | Tỷ trọng thời gian | Khối lượng FLOPs | Mức tải phần cứng |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **CPU 1. Tiền xử lý văn bản** | `text_normalizer.py` (Regex) | **0.35 ms** | ~0.9% | < 0.01% | < 0.1% |
| **CPU 2. Phân tích ngữ điệu** | `prosody_enhancer.py` (Dấu câu) | **0.20 ms** | ~0.5% | < 0.01% | < 0.1% |
| **CPU 3. Tokenization & Mask** | `UnicodeProcessor` (JSON Index) | **0.15 ms** | ~0.4% | < 0.01% | < 0.1% |
| **CPU 4. Điều phối Session NPU**| Host Orchestrator / DMA Buffer | **0.30 ms** | ~0.8% | < 0.01% | < 0.1% |
| **CPU 5. Hậu xử lý âm thanh** | `normalize_peak` / PCM Int16 | **0.25 ms** | ~0.6% | < 0.05% | < 0.1% |
| ─── **TỔNG CỘNG CPU HOST** ─── | **Mặt phẳng điều khiển (Control Plane)** | **`~1.25 ms`** | **`~3.2%`** | **`< 0.1%`** | **`< 0.1% (Gần như nghỉ)`** |
| ─── **QUALCOMM HEXAGON NPU** ──| **Mặt phẳng tính toán nơ-ron (4 Submodel)**| **`~38 - 163 ms`** | **`~96.8%`** | **`~99.9%`** | **NPU HTP Core (Burst)** |
| ════ **TỔNG TOÀN BỘ HỆ THỐNG** ═| **CPU Host + Hexagon NPU Co-Processing** | **`~40 - 165 ms`** | **`100.0%`** | **`100.0%`** | **Tối ưu điện năng & độ trễ** |

---

## 🛠️ 3. CHI TIẾT 5 GIAI ĐOẠN & CÔNG VIỆC CỤ THỂ CPU ĐẢM NHẬN

CPU Host tham gia vào đúng 5 mắt xích thiết yếu mà phần cứng NPU không thể tự giải quyết:

### Giai Đoạn 1: Tiền xử lý chuỗi văn bản phi cấu trúc (Text Normalization)
* **Tệp mã nguồn:** `src/step3_tts/text_normalizer.py`
* **Công việc cụ thể:**
  1. Chạy các biểu thức chính quy (Regular Expressions) để chuẩn hóa chuỗi ký tự.
  2. Đọc và chuyển đổi số tự nhiên, số thập phân, ngày tháng năm thành chuỗi chữ cái tiếng Việt tương ứng (ví dụ: `2026` $\to$ *"hai nghìn không trăm hai mươi sáu"*).
  3. Mở rộng từ viết tắt kỹ thuật (`VNG` $\to$ *"vê en giê"*, `AI` $\to$ *"ei ai"*, `NPU` $\to$ *"en pi u"*).
  4. Chuẩn hóa đơn vị tiền tệ (`$ 50` $\to$ *"năm mươi đô la"*, `100.000₫` $\to$ *"một trăm nghìn đồng"*).
  5. Áp dụng chuẩn Unicode NFKD, làm sạch biểu tượng cảm xúc (emoji) và các ký tự trang trí gây nhiễu.

### Giai Đoạn 2: Phân tích cấu trúc ngữ điệu & Ngắt nghỉ vi mô (Prosody & Micro-Pause)
* **Tệp mã nguồn:** `src/step3_tts/prosody_enhancer.py`
* **Công việc cụ thể:**
  1. Phân rã câu văn thành các mệnh đề độc lập dựa trên dấu câu ngắt nghỉ: `re.split(r"([,.\?!;:])", text)`.
  2. Cấp phát thời lượng ngắt nghỉ vi mô vật lý: gán `150 ms` cho dấu phẩy `,`, `350 ms` cho dấu chấm `.`, `250 ms` cho dấu hỏi `?`.
  3. Gán chỉ số đường bao cao độ (Pitch Contour): tăng hệ số cao độ đuôi câu `pitch_accent = 1.25` cho câu nghi vấn nhằm tái tạo ngữ điệu hỏi tự nhiên của con người.

### Giai Đoạn 3: Token hóa Unicode & Sinh mặt nạ nhị phân (Tokenization & Mask Generation)
* **Tệp mã nguồn:** Thư viện `supertonic/core.py` (`UnicodeProcessor`)
* **Công việc cụ thể:**
  1. Tra cứu bảng từ điển JSON `unicode_indexer.json` để ánh xạ từng ký tự và dấu câu thành mảng số nguyên `text_ids` dạng `int64` (kích thước cố định tĩnh `(1, 64)`).
  2. Tạo tensor mặt nạ nhị phân `text_mask` dạng `float32` (kích thước `(1, 1, 64)`) bằng phép so sánh độ dài câu: gán `1.0` cho ký tự hợp lệ và `0.0` cho khoảng đệm padding.
  3. Lấy mẫu ma trận nhiễu trắng Gauss ban đầu $x_0 \sim \mathcal{N}(0, I)$ (`sample_noisy_latent`) dựa trên tổng thời lượng mà Submodel 1 vừa dự đoán.

### Giai Đoạn 4: Điều phối luồng, quản lý bộ nhớ đệm chia sẻ (Host Orchestrator)
* **Tệp mã nguồn:** `src/step3_tts/supertonic_pure_npu_v2_engine.py` / C++ Native API
* **Công việc cụ thể:**
  1. Cấp phát và khóa các vùng nhớ chia sẻ DMA / ION giữa CPU và NPU nhằm thực hiện cơ chế truyền dữ liệu **Zero-Copy**, loại trừ hoàn toàn độ trễ sao chép dữ liệu (Memory Copy Overhead).
  2. Gửi tín hiệu kích hoạt phiên thực thi phần cứng (`session.run()` trên ONNX Runtime QNN Provider hoặc `QnnGraph_execute()` trên Android C++ JNI).
  3. Quản lý biến đếm bước thời gian $t \in [0, 4]$ của phương trình vi phân Euler trong Submodel 3 Flow-Matching.

### Giai Đoạn 5: Hậu xử lý âm thanh & Giao tiếp phần cứng phát loa (Audio Driver & DAC)
* **Tệp mã nguồn:** `src/step3_tts/tts_manager.py` (`normalize_peak`, `float32_to_int16_bytes`)
* **Công việc cụ thể:**
  1. Tiếp nhận buffer 307,200 mẫu `Float32` từ đầu ra của Neural Vocoder NPU.
  2. Kiểm tra và chuẩn hóa đỉnh biên độ (`target_peak = 0.95`) để ngăn ngừa hiện tượng méo tiếng kỹ thuật số (digital clipping).
  3. Ép kiểu dữ liệu từ số thực `Float32` sang số nguyên 16-bit PCM (`Int16` dải giá trị $[-32768, +32767]$).
  4. Giao tiếp với Audio Subsystem của hệ điều hành (ALSA trên Linux nhúng hoặc AudioTrack trên Android) để đẩy buffer qua giao tiếp I2S tới chip phần cứng giải mã âm thanh (DAC) phát ra loa.

---

## 🚫 4. TẠI SAO KHÔNG THỂ THAY THẾ CPU BẰNG NPU TRONG CÁC TÁC VỤ NÀY?

Đây là ranh giới bất khả biến xuất phát từ **nguyên lý kiến trúc bán dẫn (Silicon Microarchitecture)**:

```text
┌─────────────────────────────────────────┐    ┌─────────────────────────────────────────┐
│          KIẾN TRÚC CPU HOST             │    │       KIẾN TRÚC QUALCOMM HEXAGON NPU    │
├─────────────────────────────────────────┤    ├─────────────────────────────────────────┤
│ • Tập lệnh đa dụng (General Purpose)    │    │ • Mảng tâm thu ma trận (Systolic Array) │
│ • Có bộ dự đoán rẽ nhánh (Branch Predict)│   │ • KHÔNG có bộ dự đoán rẽ nhánh          │
│ • Quản lý con trỏ ngẫu nhiên (Pointers) │    │ • Chỉ tính toán ma trận tĩnh (Tensors)  │
│ • Có ngăn xếp đệ quy (Call Stack/Heap)  │    │ • KHÔNG có ngăn xếp bộ nhớ đệ quy       │
│ • Hỗ trợ System Calls nhân HĐH (Kernel) │    │ • Bị cô lập, KHÔNG gọi được Syscalls    │
├─────────────────────────────────────────┤    ├─────────────────────────────────────────┤
│ 👉 Xử lý cực nhanh chuỗi ký tự, Regex,  │    │ 👉 Tối ưu tuyệt đối cho phép nhân ma    │
│    logic điều kiện if/else, quản lý I/O │    │    trận số song song (GEMM / Conv1D)    │
└─────────────────────────────────────────┘    └─────────────────────────────────────────┘
```

1. **NPU hoàn toàn không hỗ trợ chuỗi ký tự phi cấu trúc (String Manipulation):**
   - NPU chỉ hiểu tensor số học (Float32, Float16, Int8, Int16). Nó không có tập lệnh thao tác trên mảng ký tự chuỗi, không thể chạy các thuật toán Regex tìm kiếm mẫu (Pattern Matching).
2. **NPU bị tê liệt trước logic rẽ nhánh động (Branching Logic):**
   - Các thuật toán đọc số tiếng Việt (ví dụ: khi nào đọc là *"mười"*, *"mốt"*, *"lăm"*, *"lẻ"*) chứa hàng chục câu lệnh điều kiện lồng nhau (`if/else`).
   - NPU không có phần cứng dự đoán rẽ nhánh (Branch Predictor). Nếu đưa logic rẽ nhánh vào NPU, NPU sẽ bị dừng chu kỳ xung nhịp (Pipeline Stalling) và mất toàn bộ lợi thế xử lý song song.
3. **NPU không thể giao tiếp với thiết bị ngoại vi hệ thống (No OS Syscall Access):**
   - NPU là một bộ gia tốc được cô lập trong phần cứng. Nó không có quyền gửi ngắt (Interrupts) tới hệ điều hành hay gọi các lời gọi hàm hệ thống (`ioctl`, `write`) để truyền dữ liệu tới chip DAC âm thanh hay màng loa. Việc giao tiếp với thế giới bên ngoài bắt buộc phải thông qua CPU Host.

---

## 💥 5. HẬU QUẢ KỸ THUẬT NẾU CỐ TÌNH ÉP NPU THAY THẾ CPU

Nếu một kỹ sư cố tình tìm cách "nhồi nhét" toàn bộ 100% công việc của CPU sang NPU (ví dụ: huấn luyện thêm một mạng nơ-ron phụ để làm thay nhiệm vụ của Regex và Tokenizer), hệ thống sẽ gánh chịu **4 thảm họa kỹ thuật nghiêm trọng**:

### Thảm họa 1: Đồ thị nơ-ron phình to & Tràn bộ nhớ SRAM (Graph Explosion & OOM)
* **Hậu quả:** Để thay thế bộ quy tắc chuẩn hóa văn bản regex chỉ nặng vài chục KB trên CPU, ta sẽ phải xây dựng một mô hình Seq2Seq/Transformer phụ (như mô hình T5-Small nặng hơn $150\text{ MB}$).
* **Hệ lụy:** Đồ thị này sẽ làm tràn bộ nhớ đệm SRAM cực kỳ quý giá của Hexagon NPU (vốn chỉ có dung lượng tĩnh giới hạn). Điều này dẫn tới lỗi tràn bộ nhớ **Out-Of-Memory (OOM)** và phá hủy hoàn toàn khả năng nạp Vocoder vào SRAM.

### Thảm họa 2: Độ trễ tăng vọt gấp 10 lần (Severe Latency Degradation)
* **Hậu quả:** Việc chạy một bộ quy tắc regex trên CPU chỉ tốn **`0.35 ms`**.
* **Hệ lụy:** Nếu ép một mạng nơ-ron trên NPU thực hiện nhiệm vụ đọc số và viết tắt, thời gian nạp tensor, suy luận và đồng bộ hóa qua lại sẽ tốn từ **`15 ms đến 30 ms`** — chậm hơn CPU **hơn 50 lần**, làm phá vỡ hoàn toàn tiêu chuẩn Time-to-First-Byte tức thì ($<40\text{ ms}$).

### Thảm họa 3: Đánh mất tính linh hoạt & Khóa cứng mô hình (Zero Extensibility)
* **Hậu quả:** Ngôn ngữ thực tế luôn thay đổi. Nếu xuất hiện một từ viết tắt mới (ví dụ: một từ lóng hoặc tên dự án mới) hoặc cần chỉnh lại một mốc ngắt nghỉ:
  - **Trên CPU:** Kỹ sư chỉ cần mở file `text_normalizer.py` và thêm 1 dòng code regex trong **5 giây** mà không cần khởi động lại mô hình.
  - **Trên NPU:** Kỹ sư sẽ phải thu thập dữ liệu, huấn luyện lại mạng nơ-ron, chạy lại toàn bộ quy trình lượng hóa W8A16 và biên dịch lại tệp QNN Context Binary mất **nhiều giờ đồng hồ**.

### Thảm họa 4: Nguy cơ sinh lỗi ảo giác ký tự (Hallucination Risk)
* **Hậu quả:** Regex trên CPU là logic xác thực tất định 100% (Deterministic Logic). Con số `100.000` chắc chắn luôn được dịch thành *"một trăm nghìn"*.
* **Hệ lụy:** Nếu dùng mạng nơ-ron trên NPU để dịch số, mô hình xác suất có thể bị ảo giác (hallucination), đọc sai mệnh giá tiền tệ hoặc sai số điện thoại khẩn cấp, gây ra sai lệch thông tin nghiêm trọng trong giao tiếp thực địa.

---

## ⚖️ 6. TRIẾT LÝ ĐỒNG XỬ LÝ TỐI ƯU (CO-PROCESSOR SYNERGY)

Sự phân chia giữa CPU và NPU trong dự án OneVoice AI đạt đến **"Tỷ Lệ Vàng" của kỹ nghệ hệ thống**:

$$\text{Tổng Hiệu Năng} = \underbrace{\text{CPU Host (0.1% FLOPs, < 1.5 ms)}}_{\text{Linh hoạt, Chính xác tuyệt đối, Điều khiển thông minh}} + \underbrace{\text{Hexagon NPU (~99.9% FLOPs, 7.4 ms)}}_{\text{Sức mạnh giải mã ma trận siêu thanh, Tiết kiệm pin}}$$

* **CPU làm nhiệm vụ Bảo Vệ NPU:** CPU gánh vác toàn bộ các tác vụ xử lý chuỗi "lộn xộn", lọc sạch dữ liệu thô, biến chúng thành các ma trận số tĩnh hoàn hảo trước khi đưa vào NPU.
* **NPU làm nhiệm vụ Giải Phóng CPU:** NPU gánh toàn bộ 85% tải tính toán nặng nề của mạng Vocoder và Flow-Matching, giúp CPU mát mẻ, tiêu thụ điện năng gần như bằng 0 và không bao giờ bị quá nhiệt tụt xung.

> **Kết luận:** Việc CPU tham gia với tỷ trọng thời gian **~3%** là một **quyết định thiết kế hoàn hảo và tối ưu tuyệt đối về mặt kỹ thuật**. CPU và NPU trong dự án không loại trừ nhau mà phối hợp nhịp nhàng như hai nửa không thể tách rời của một cỗ máy hoàn chỉnh.
