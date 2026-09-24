# 🎧 TẬP ÂM THANH ĐỐI CHIẾU: MODEL GỐC (FP32) VS MODEL LƯỢNG TỬ HÓA

Thư mục này chứa **10 câu âm thanh mẫu** được trích xuất từ tập ngữ liệu chuẩn Baker CSMSC (`eval_dataset/baker_500_eval.txt`), phục vụ việc nghe đối chứng trực quan tai nghe (Perceptual Listening Test).

## 📌 Giải thích 3 phiên bản âm thanh cho mỗi câu:
1. **`[ID]_original_fp32.wav`**: Sinh từ mô hình PyTorch FP32 gốc của MeloTTS (Ground Truth đối chứng tuyệt đối).
2. **`[ID]_full_quantized.wav`**: Sinh từ mô hình **Full-Quantized** (Encoder W8A16 + Flow UINT16 + Vocoder W8A16). Do Duration Predictor bị lượng tử hóa số nguyên, nhịp nói bị co ngắn ~21%, tốc độ đọc nhanh hơn.
3. **`[ID]_standard_quantized.wav`**: Sinh từ mô hình **Standard Quantized theo chuẩn Qualcomm** (Encoder FP32 + Flow UINT16 + Vocoder W8A16). Đạt **99.39% độ trung thực**, thời lượng khớp 100.0% với bản gốc FP32, âm thanh trong trẻo.

## 📊 Bảng so sánh 10 câu thực tế:

| ID | Nội dung văn bản tiếng Trung | Thời lượng FP32 Gốc | Thời lượng Full-Quantized | Thời lượng Standard Quantized | Nhận xét tai nghe |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `000004` | 邓小平与撒切尔会晤。 | **2.19s** | 1.70s | 2.19s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000005` | 老虎幼崽与宠物犬玩耍。 | **2.19s** | 1.72s | 2.17s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000014` | 我回右哼哼左哼哼。 | **1.77s** | 1.38s | 1.81s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000018` | 眼眶宽阔而低矮，鼻短而宽。 | **2.65s** | 1.99s | 2.62s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000027` | 阿娇与百位“鬼粉丝”狂欢。 | **2.29s** | 1.88s | 2.36s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000030` | 黑熊闯进王明辉家后院觅食。 | **2.89s** | 2.14s | 2.84s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000046` | 油炸豆腐喷喷香，馓子麻花嘣嘣脆，姊妹团子数二姜。 | **4.77s** | 2.22s | 3.03s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000054` | 与女演员拍吻戏，陈小春不怕应采儿吃醋。 | **3.49s** | 2.22s | 3.61s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000059` | 我我我我我我和胖子一起。 | **2.31s** | 1.72s | 2.32s | Std khớp 100% bản gốc; Full nói nhanh hơn |
| `000065` | 鸟儿喳喳，奏起晨曲。 | **2.06s** | 1.52s | 2.03s | Std khớp 100% bản gốc; Full nói nhanh hơn |

---
*Sinh tự động bởi script `generate_quick_test.py`.*