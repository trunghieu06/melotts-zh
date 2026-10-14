# Báo Cáo Kỹ Thuật Toàn Diện: Kiểm Điểm Lỗi Dự Án, Đánh Giá Thực Trạng & Định Hướng Khắc Phục

**Dự án:** OneVoice — Hệ thống TTS 100% NPU Qualcomm Dragonwing IQ-9075 EVK  
**Người thực hiện:** Antigravity AI Pair Programmer & OneVoice Team  
**Nguyên tắc báo cáo:** **Nói thẳng thắn, trung thực tuyệt đối, không ngụy biện, mổ xẻ tận gốc vấn đề kỹ thuật.**  

---

## TÓM TẮT THỰC TRẠNG (EXECUTIVE SUMMARY)

1. **Thực trạng âm thanh hiện tại:**  
   Khi ghép nối mô hình đã huấn luyện (`melo_byte_encoder_npu.onnx` / `.dlc`) với khối giải mã Vocoder (`decoder.bin`) trên NPU EVK, **âm thanh xuất ra không thể nghe được, toàn là âm rác, tiếng hú đơn âm hoặc tiếng bíp**. Dự án **chưa đạt được** mục tiêu tổng hợp tiếng nói dễ nghe từ chuỗi Byte chạy 100% trên NPU.
2. **Nguyên nhân cốt lõi:**  
   * Tệp huấn luyện trên Kaggle (`Train_Byte_Encoder_MeloTTS.ipynb` và `train_distill_byte_encoder.py`) mắc **3 lỗi toán học và kiến trúc nghiêm trọng**, khiến mô hình 25M tham số sau 10 tiếng huấn luyện thực chất chỉ học được các vector trung bình nhiễu, **độ tương đồng Cosine so với đặc trưng chuẩn của Teacher là số âm (-0.0718)**.
   * Bản chất bài toán Text-to-Speech (TTS) đòi hỏi cơ chế giãn nhịp thời lượng (Duration Expansion) và chuyển đổi không gian âm học (Flow). Việc cố gắng ép một mô hình Transformer thuần túy dự đoán thẳng từ Byte sang âm thanh mà không có cơ chế căn chỉnh độ dài (Alignment) là **sai về mặt nguyên lý thiết kế mô hình**.

---

## PHẦN 1: MỔ XẺ TOÀN DIỆN CÁC LỖI TRONG FILE TRAIN NOTEBOOK

Qua rà soát từng dòng lệnh trong `notebooks/Train_Byte_Encoder_MeloTTS.ipynb` (và file script `train_distill_byte_encoder.py`), dưới đây là các lỗi nghiêm trọng đã khiến toàn bộ quá trình train 10 tiếng bị vô hiệu hóa:

### 1.1. Lỗi Lệch Vị Trí & Cắt Cụt Chuỗi Token (`min_len` mismatch)
Tại **Cell 13** của Notebook, đoạn mã tính hàm mất mát (Loss) được viết như sau:

```python
# Trích xuất đặc trưng Teacher (L = số lượng âm vị phonemes)
t_feat = extract_teacher_feature(text)  # Shape: [1, 192, L]
L = t_feat.size(2)

# Student forward (chuỗi Byte UTF-8)
b_tensor = text_to_bytes(text, seq_len=512).to(DEVICE)
s_feat = student(b_tensor)  # Shape: [1, 192, 512]

# Cắt so sánh theo min_len
min_len = min(L, 512)
loss = criterion(s_feat[:, :, :min_len], t_feat[:, :, :min_len])
```

#### Hậu quả toán học:
* Trong tiếng Trung, 1 chữ Hán chiếm **3 byte UTF-8**, nhưng chỉ tương ứng với **1 đến 2 âm vị (phoneme)**.  
  *Ví dụ câu 10 chữ Hán:* Số byte UTF-8 là ~32 bytes, nhưng số âm vị chỉ có ~20 phonemes ($L = 20$).
* Do đó, `min_len = 20`. Lệnh cắt `[:min_len]` đã dẫn đến 2 thảm họa:
  1. **Lệch pha hoàn toàn giữa Byte và Âm vị:**  
     Vị trí 0 của Student (Byte 1 của chữ thứ nhất) bị ép so sánh với Âm vị 0.  
     Vị trí 1 của Student (Byte 2 của chữ thứ nhất) bị ép so sánh với Âm vị 1.  
     Vị trí 2 của Student (Byte 3 của chữ thứ nhất) bị ép so sánh với Âm vị 2 (vốn đã thuộc về chữ thứ hai!).  
     $\implies$ **Mô hình bị ép học một ánh xạ hỗn loạn, Byte của chữ này lại phải dự đoán đặc trưng của chữ khác!**
  2. **Bỏ rơi 50% câu văn phía sau (Zero Gradient):**  
     Các byte từ vị trí 21 đến 32 (nửa sau của câu) hoàn toàn không nằm trong `[:min_len]`, nên **không hề nhận được bất kỳ gradient nào để học**. Nửa sau câu văn hoàn toàn là rác ngẫu nhiên.

---

### 1.2. Lỗi Sai Lệch Không Gian Ngữ Nghĩa (Hook `enc_p` thay vì Latent $z$)
Tại **Cell 11** của Notebook:

```python
def extract_teacher_feature(text: str):
    captured = {}
    def hook_fn(module, args, output):
        captured["feat"] = output[0].detach()  # output[0] của enc_p
    handle = teacher.model.enc_p.register_forward_hook(hook_fn)
    ...
```

#### Hậu quả kiến trúc:
* Mô hình được huấn luyện để bắt chước `enc_p` (TextEncoder của VITS). Đầu ra của `enc_p` là $m_p$:
  * **Miền không gian:** Biểu diễn ngôn ngữ học tĩnh của từng âm vị.
  * **Độ lệch chuẩn (std):** Cực nhỏ, chỉ dao động quanh $0.04 - 0.06$.
  * **Độ dài:** Đúng bằng số lượng âm vị ($P$).
* Tuy nhiên, khối NPU Vocoder (`decoder.bin`) lại đòi hỏi đầu vào là $z$:
  * **Miền không gian:** Biểu diễn âm học tiềm ẩn của phổ Mel-Spectrogram sau khi đã đi qua khối **Normalizing Flow**.
  * **Độ lệch chuẩn (std):** Chuẩn hóa quanh $0.88 - 0.95$.
  * **Độ dài:** Chiều dài khung thời gian âm thanh ($T$, thường gấp 3 đến 5 lần $P$).
* **Hậu quả:** Sau khi train, ta lại lấy đầu ra $m_p$ này nhồi trực tiếp vào `decoder.bin`. Vocoder khi nhận một vector lạ lẫm, không đúng phân phối, có độ lệch chuẩn quá nhỏ sẽ lập tức rơi vào trạng thái dao động tự kích thích tại các tần số cộng hưởng của mạch tích chập chuyển vị:
  $$f_{\text{beep}} = \frac{44,100}{512} \times k = 86.13\text{ Hz} \times k \quad (k=3 \implies 258.4\text{ Hz},\; k=5 \implies 430.7\text{ Hz})$$
  Đây chính là nguồn gốc phát ra tiếng bíp chói tai át hoàn toàn giọng người.

---

### 1.3. Lỗi Nghẽn Cổ Chai CPU Khiến Quá Trình Huấn Luyện Mất 10 Tiếng
Tại **Cell 13**, bên trong vòng lặp mỗi batch của mỗi epoch:

```python
for epoch in range(1, EPOCHS + 1):
    for text in pbar:
        with torch.no_grad():
            t_feat = extract_teacher_feature(text) # GỌI CPU CHẠY MELOTTS TẠI MỖI BƯỚC!
```

* Mỗi lần gọi `extract_teacher_feature`, thư viện phải chạy toàn bộ pipeline CPU: phân tích từ tố `jieba`, chuyển bính âm `pypinyin`, suy luận `TextEncoder` và ghi file tạm ra `/tmp/dummy.wav`.
* Với 3,000 câu $\times$ 10 epochs = **30,000 lần gọi hàm CPU tuần tự**.
* Tốc độ trung bình: ~1.2 giây/câu $\implies$ Tổng thời gian mất $30,000 \times 1.2\text{s} \approx 36,000\text{s} = \mathbf{10\text{ tiếng}}$!
* **Thực tế lãng phí:** GPU chỉ hoạt động tính toán đạo hàm chưa đến **2% thời gian**, 98% thời gian còn lại GPU phải ngồi chờ CPU xử lý chuỗi ký tự. Nếu trích xuất trước (offline pre-caching) 3,000 câu ra file `.pt` một lần duy nhất (mất ~15 phút), thì toàn bộ 10 epoch huấn luyện trên GPU chỉ mất **dưới 3 phút**!

---

### 1.4. Bất Khả Thi Về Mặt Nhiệm Vụ (G2P Learning Capacity)
* Tiếng Trung là ngôn ngữ tượng hình. Để đọc được một chữ Hán, hệ thống cần:
  1. Nhận diện từ tố (Word Segmentation).
  2. Tra từ điển phiên âm Pinyin (kèm dấu thanh điệu 1, 2, 3, 4, thanh nhẹ).
  3. Xử lý từ đa âm (Polyphonic disambiguation).
* Mô hình `MeloByteEncoder` chỉ là một mạng Transformer 25.45M tham số, được huấn luyện vỏn vẹn trên **3,000 câu đơn giản trong 10 epochs**.
* Về mặt dung lượng thông tin, mô hình **không thể tự học thuộc toàn bộ từ điển phiên âm chữ Hán** trực tiếp từ các byte số vô nghĩa mà không có bất kỳ cơ chế giám sát ngữ âm (Phonetic Supervision) nào.

---

## PHẦN 2: ĐÁNH GIÁ TRUNG THỰC: NHỮNG GÌ ĐÃ LÀM ĐƯỢC VÀ CHƯA LÀM ĐƯỢC

### 2.1. Những Gì ĐÃ Làm Được (Điểm Sáng Thực Nghiệm)

1. **Chứng minh tính khả thi của 100% NPU Hardware:**
   * Toàn bộ mã nguồn đã được biên dịch thành công sang QNN DLC và thực thi trực tiếp trên phần cứng **Qualcomm Dragonwing IQ-9075 EVK** (HTP v73, QAIRT 2.45.0) qua nền tảng Qualcomm AI Hub.
   * Xác nhận **0% CPU Fallback**: Mọi toán tử trong đồ thị đều chạy thuần trên chip NPU.
2. **Khối Vocoder NPU (`decoder.bin`) hoạt động hoàn hảo khi có đầu vào đúng:**
   * Khi cấp ma trận $z$ chuẩn từ Teacher vào `decoder.bin` trên chip EVK thật (Job: [`jp4e374qg`](https://workbench.aihub.qualcomm.com/jobs/jp4e374qg/)), bo mạch xuất ra file âm thanh **hoàn toàn sạch sẽ, giọng người tự nhiên F0 = 249 Hz, không hề có tiếng bíp**. Điều này chứng minh Vocoder trên phần cứng EVK không bị lỗi.
3. **Hiện thực hóa thành công các giải pháp đồ thị tĩnh theo bài báo:**
   * **Problem 1:** Byte Transformer 25.45M tham số chạy trong **8.505 ms** (256/256 ops NPU).
   * **Problem 2:** Binary Stencil Duration Expansion ($M \times H$) chạy trong **0.254 ms** (12/12 ops NPU).
   * **Problem 3:** Sliding-Window Chunking 64-frame chạy trong **0.555 ms** (8/8 ops NPU).
   * **Problem 4:** Artifact Trimming & Dequantize chạy trong **0.163 ms** (9/9 ops NPU).
   * Tổng độ trễ phần cứng đạt mức cực nhanh: **~15.4 ms** (RTF = 0.0088, nhanh gấp 113 lần thời gian thực).

---

### 2.2. Những Gì CHƯA Làm Được (Thừa Nhận Thẳng Thắn)

1. **CHƯA có một mô hình Byte-to-Acoustic hoạt động được:**  
   Mô hình `melo_byte_encoder_npu.onnx` hiện tại là **không thể sử dụng** (unusable) cho việc tạo giọng nói thực tế.
2. **CHƯA giải quyết được vấn đề căn chỉnh độ dài (Alignment):**  
   Chưa tích hợp được cơ chế dự đoán độ dài động (Duration Predictor) từ tầng Byte sang tầng Frame một cách tự động trên NPU.
3. **Các giải pháp chắp vá trước đó (Option A, Fast Fine-Tuning) đều thất bại khi nghe thực tế:**  
   * Phép nội suy `nearest-neighbor` (Option A) tạo ra tiếng rít 430.7 Hz.
   * Phép fine-tune ép đầu ra 512 frame tạo ra tiếng hú 258.4 Hz.
   * Việc nối `flow.bin` với đầu ra lỗi chỉ biến tiếng bíp thành âm xào xạc vô nghĩa (garbage noise).

---

## PHẦN 3: BẢNG SO SÁNH CÁC PHƯƠNG ÁN & KẾT QUẢ THỰC TẾ

| Phương án | Bản chất kỹ thuật | Kết quả trên EVK thật | Nguyên nhân thất bại / Thành công |
|---|---|---|---|
| **Pipeline gốc (Giai đoạn 2 ban đầu)** | Lấy ByteEncoder nối thẳng sang Decoder NPU | ❌ Tiếng rít chói tai + nhiễu tĩnh kéo dài 5.94s | ByteEncoder xuất $m_p$ (std=0.04), Decoder cần $z$ (std=0.9). Padding 512 frame phát ra âm rác. |
| **Option A (Duration Expansion)** | Cắt 41 phonemes, nội suy `nearest` sang 151 frames | ❌ Tiếng bíp the thé 430.7 Hz (biên độ 8,613) | Bước nhảy bậc thang của phép nội suy kích thích họa ba bậc 5 của Vocoder ($5 \times 86.13\text{ Hz}$). |
| **Fast Fine-Tuning sang $z$** | Huấn luyện ByteEncoder dự đoán trực tiếp $z$ | ❌ Tiếng hú đơn âm 258.4 Hz (biên độ 12,855) | Mô hình không có Duration Predictor, các vị trí padding (zero) tạo ra vector hằng số lặp lại, kích thích họa ba bậc 3 ($3 \times 86.13\text{ Hz}$). |
| **Ghép Flow.bin + Decoder.bin với ByteEncoder** | Cho đầu ra ByteEncoder đi qua Flow QNN rồi vào Decoder | ❌ Toàn bộ âm thanh là tiếng rác vô nghĩa | Dữ liệu đầu vào từ ByteEncoder là rác (Cosine sim = -0.07), nguyên lý "Garbage In, Garbage Out". |
| **Kiểm chứng Vocoder với Teacher $z$ (Job `jp4e374qg`)** | Nạp ma trận $z$ chuẩn từ Teacher vào Decoder NPU | **✅ THÀNH CÔNG: Giọng người chuẩn 100%, 0% bíp** | Chứng minh phần cứng EVK và `decoder.bin` hoạt động hoàn hảo khi có dữ liệu đầu vào đúng. |

---

## PHẦN 4: ĐỊNH HƯỚNG KỸ THUẬT ĐỂ ĐẠT CHẤT LƯỢNG THẬT (ACTIONABLE ROADMAP)

Để có một hệ thống Voice AI chạy trên Qualcomm NPU vừa **0% CPU fallback** vừa **nói rõ tiếng người**, dự án cần chuyển hướng theo đúng chuẩn nghiên cứu học máy hiện đại:

### Phương Án 1: Giữ Đúng Thiết Kế Chuẩn Của Qualcomm MeloTTS (Khuyên Dùng Cho Demo Sớm)
Qualcomm đã cung cấp sẵn bộ nhị phân tối ưu cho MeloTTS-ZH trên NPU:
* `encoder.bin` (Nhận âm vị + thanh điệu, chạy trên NPU).
* `flow.bin` (Biến đổi Flow, chạy trên NPU).
* `decoder.bin` (Vocoder HiFi-GAN, chạy trên NPU).
* **Quy trình chuẩn:**  
  Giữ phần Frontend (G2P: Chữ Hán $\to$ Âm vị) chạy cực nhẹ trên CPU (~2 ms), toàn bộ phần suy luận nặng (Encoder, Flow, Decoder) chiếm 98% tính toán chạy trên NPU. Đây là cách tiếp cận tiêu chuẩn trong công nghiệp (giống như Whisper hay FastSpeech của Microsoft, Google).

### Phương Án 2: Nếu Bắt Buộc 100% Byte-to-Speech (Không Dùng Bất Kỳ G2P CPU Nào)
Nếu bài toán bắt buộc không có bất kỳ dòng code CPU nào từ Byte sang Audio, mô hình cần được thiết kế lại đúng chuẩn nghiên cứu khoa học:
1. **Kiến trúc mô hình:**
   * Cần có **Duration Predictor Module** độc lập (dự đoán số lượng frame cho từng byte/ký tự).
   * Dùng **CTC Loss** hoặc **Monotonic Alignment Search (MAS)** để tự động căn chỉnh giữa chuỗi Byte đầu vào và chuỗi âm thanh đầu ra.
2. **Quy trình Huấn Luyện Lại (Đúng Kỹ Thuật):**
   * **Bước 1:** Trích xuất trước toàn bộ ma trận đặc trưng ($m_p$, duration $d$, latent $z$) của 10,000 câu lưu sẵn ra file `.pt` (mất ~30 phút một lần duy nhất).
   * **Bước 2:** Huấn luyện trên GPU với dữ liệu pre-cached (mất ~15 phút cho 50 epochs).
   * **Bước 3:** Sử dụng hàm mất mát kết hợp: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{acoustic}} + \lambda_1 \mathcal{L}_{\text{duration}} + \lambda_2 \mathcal{L}_{\text{CTC}}$.

---

## TỔNG KẾT

* Sự cố âm thanh rác/tiếng bíp **không phải do phần cứng hay lỗi suy luận của chip Dragonwing IQ-9075 EVK**, mà do lỗi toán học nghiêm trọng trong file train notebook (`Train_Byte_Encoder_MeloTTS.ipynb`), khiến mô hình được huấn luyện sai lệch pha và sai không gian ngữ nghĩa.
* Báo cáo này ghi nhận trung thực 100% hiện trạng kỹ thuật, không che giấu lỗi, nhằm giúp đội ngũ OneVoice có cái nhìn chuẩn xác nhất để quyết định bước đi tiếp theo cho dự án.
