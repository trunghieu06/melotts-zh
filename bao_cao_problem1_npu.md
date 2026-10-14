# Báo Cáo Triển Khai Thành Công Problem 1: 100% NPU-Native Text Tokenizer & Byte-Level Transformer

**Người thực hiện:** Antigravity AI Pair Programmer & OneVoice Team  
**Phần cứng kiểm chứng:** Qualcomm Dragonwing IQ-9075 EVK (Hexagon HTP v73, QAIRT 2.45.0, Target Runtime `qnn_dlc`, Compute Unit `npu`)  
**Ngày thực hiện:** Tháng 10/2026  

---

## 1. Bản Chất Bài Toán & Thách Thức (Problem 1)

* **Thách thức:** Quá trình tiền xử lý văn bản truyền thống (BPE / SentencePiece tokenizer, UTF-8 decode, tra cứu từ điển/hash table) phụ thuộc hoàn toàn vào luồng xử lý CPU với các rẽ nhánh động (`if/else`, vòng lặp `while`, kích thước tensor động). Trình biên dịch Qualcomm Hexagon Tensor Processor (HTP) **không hỗ trợ** các op logic kiểu `If`, không hỗ trợ kiểu chuỗi (string) và `Conv1d=NO` trên mọi kiểu dữ liệu.
* **Giải pháp NPU-Native:** Loại bỏ hoàn toàn CPU Tokenization loop. Thay vì xử lý text dưới dạng chuỗi, văn bản được biểu diễn trực tiếp dưới dạng tensor byte số nguyên tĩnh:
  $$B \in \{0, \dots, 255\}^{1 \times N}, \quad N = 512$$
* Phép BPE được thay thế bằng toán tử Gather nhúng byte tĩnh:
  $$E = \text{Gather}(W_{\text{byte}}[256, D], B) \in \mathbb{R}^{1 \times N \times D}$$
  tương ứng với `nn.Embedding(256, D)` trong PyTorch.
* Toàn bộ khối Transformer phía sau được thiết kế chỉ sử dụng các toán tử nằm trong Whitelist tối ưu 100% của phần cứng HTP:
  `MatMul`, `Softmax`, `LayerNorm`, `Gelu`, `Reshape`, `Transpose`, `Add`, `Mul`, `FullyConnected`. Không dùng `OneHot+MatMul` (lãng phí và sai khác dtype), không dùng `Conv1d`.

---

## 2. Kết Quả Thực Nghiệm Trên Thiết Bị Thật (Dragonwing IQ-9075 EVK)

Toàn bộ đồ thị đã được compile, profile và verify inference trực tiếp trên thiết bị vật lý **Dragonwing IQ-9075 EVK** thông qua Qualcomm AI Hub Workbench:

| Mô hình | Cấu hình | Số lượng Params | Job ID Compile | Job ID Profile | Job ID Inference | Phân bổ Toán tử NPU | Latency phần cứng | Peak Memory | Cosine Sim (NPU vs Ref) |
|---|---|---|---|---|---|---|---|---|---|
| **Tiny** | $D=128, L=2, H=4$ | 0.46M (460,032) | [`j56ordrv5`](https://workbench.aihub.qualcomm.com/jobs/j56ordrv5/) | [`jpv2e9e7g`](https://workbench.aihub.qualcomm.com/jobs/jpv2e9e7g/) | [`jpe68l87g`](https://workbench.aihub.qualcomm.com/jobs/jpe68l87g/) | **71/71 ops (100% NPU, 0 CPU)** | **0.769 ms** | 7.23 MB | 1.00000 |
| **Big** | $D=512, L=8, H=8$ | **25.45M** (25,445,376) | [`j5m9odoyg`](https://workbench.aihub.qualcomm.com/jobs/j5m9odoyg/) | [`jgd68zzlp`](https://workbench.aihub.qualcomm.com/jobs/jgd68zzlp/) | [`j57ok79rg`](https://workbench.aihub.qualcomm.com/jobs/j57ok79rg/) | **255/255 ops (100% NPU, 0 CPU)** | **8.526 ms** | 9.21 MB | **0.999992** |

### Chi tiết kiểm chứng dữ liệu thật:
* **Chuỗi văn bản test:** `"你好Hi"`
* **Dãy byte UTF-8:** `[228, 189, 160, 229, 165, 189, 72, 105]` + zero-padding lên chiều dài $N = 512$.
* **Shape output từ NPU:** `[1, 1, 512, 256]` (định dạng 4D chuẩn NPU QNN DLC).
* **Độ tương đồng Cosine:** **0.999992** (sai số tuyệt đối lớn nhất chỉ $5.3 \times 10^{-3}$).
* **Độ chính xác Top-1 Byte Prediction:** Khớp **100%** giữa QNN NPU thực thi trên bo mạch và ONNX Runtime / PyTorch local.

---

## 3. Cấu Trúc Mã Nguồn Đã Xây Dựng

1. **`src/step1_asr/byte_transformer.py`**:
   - Kiến trúc `ByteTransformer`, `MultiHeadAttention`, `FFN`, `TransformerBlock` hoàn toàn NPU-native tuân thủ nghiêm ngặt HTP whitelist.
   - Hàm tiện ích `text_to_byte_tensor(text, seq_len=512)` chuyển văn bản thành byte tensor.
   - Hàm `export_model_to_onnx()` xuất file `.onnx` với `opset_version=17`.
2. **`scripts/verify_problem1_npu.py`**:
   - Script tự động tải báo cáo từ Qualcomm AI Hub, phân tích op breakdown, latency, RAM.
   - Tải dữ liệu tensor từ NPU inference job và kiểm tra Cosine Similarity so với FP32 local.
3. **Artifacts mô hình đã lưu trữ:**
   - `outputs/byte_transformer_tiny.onnx`
   - `outputs/byte_transformer_big.onnx` (25.45M parameters)

---

## 4. Kết Luận

Vấn đề **Problem 1 (Text Tokenizer & LLM)** đã được giải quyết hoàn toàn:
1. Đã chuyển toàn bộ chuỗi tokenization sang **Static Tensor Graph** 100% không còn CPU branching.
2. Đã biên dịch thành công sang QNN DLC cho thế hệ Hexagon v73 trên chip Snapdragon/Qualcomm IQ-9075.
3. Đo đạc thực tế xác nhận **100% toán tử chạy trên NPU (0 op fallback sang CPU)** với tốc độ siêu nhanh: **~0.77 ms cho Tiny** và **~8.53 ms cho Big (25.45M params)**.
