# OneVoice — Phân công nhiệm vụ chi tiết (đầy đủ giải thích + phương án + paper + quy trình test)

> Tài liệu này để **giao việc cho từng người/nhóm nhỏ** — mỗi mục viết đủ để người chưa từng nghe qua dự án
> vẫn hiểu phải làm gì, tại sao, chọn phương án nào, đọc paper nào, và test ra sao để biết đã xong.
> Tài liệu liên quan: [OneVoice_Ban_Thao_Du_Thi.md](OneVoice_Ban_Thao_Du_Thi.md) (bối cảnh cuộc thi đầy đủ),
> [OneVoice_Architecture_Analysis.md](OneVoice_Architecture_Analysis.md) (phân tích kiến trúc gốc).

## Mục lục
0. Audio Front-end (khử nhiễu, mic)
1. ASR — Nhận dạng giọng nói
2. MT — Dịch máy
3. TTS — Tổng hợp giọng nói
4. Integration/Streaming — Ghép nối hệ thống real-time
5. Data & Fine-tuning
6. Deploy & Quản lý phần cứng
7. Testing & Đánh giá tổng
8. Business/Demo

**Sơ đồ luồng dữ liệu chung** (mọi module đều nằm trên luồng này, 2 CHIỀU):
```
Mic → [0. Front-end] → audio sạch → [1. ASR] → text → [2. MT] → text dịch → [3. TTS] → audio → Loa
      (khử nhiễu, VAD)              (streaming)      (streaming)          (streaming)
Tất cả được nối bởi [4. Integration]. [5. Data] nuôi [1][2]. [6. Deploy] đưa tất cả lên chip thật.
[7. Testing] kiểm tra chéo tất cả. [8. Business] kể câu chuyện ra bên ngoài.
```

---

## 0. Audio Front-end (khử nhiễu, phát hiện giọng nói)

**Nhiệm vụ là gì:** Nhận âm thanh thô từ mic (có thể nhiều mic cùng lúc), lọc bỏ tiếng ồn xung quanh (tiếng máy, tiếng người khác), phát hiện chính xác lúc nào có người đang nói (Voice Activity Detection — VAD) để cắt thành từng đoạn câu hợp lý đưa cho ASR xử lý.

**Vì sao quan trọng:** Đây là tuyến phòng thủ đầu tiên. Nếu audio đầu vào bẩn (nhiễu, vọng, lẫn tiếng người khác), thì ASR sẽ nhận dạng sai, và MT/TTS phía sau **không có cách nào sửa lại** — lỗi cộng dồn xuôi dòng. Đây cũng là phần quyết định thiết bị có "sống được" ở nhà máy ồn hay không (khác biệt giữa demo phòng lab và demo thực tế).

**Input:** audio thô nhiều kênh (2–4 mic), 16kHz trở lên, liên tục theo thời gian thực.
**Output:** 1 luồng audio mono đã khử nhiễu + nhãn "đoạn nào có giọng nói" (segment timestamps) đẩy sang ASR.

### Các phương án
| Phương án | Mô tả | Khi nào chọn |
|---|---|---|
| A — Tối giản | 1 mic, WebRTC VAD + không khử nhiễu | Chỉ để có bản demo chạy được sớm nhất, KHÔNG dùng cho field test |
| **B — Khuyến nghị** | **GTCRN** (model khử nhiễu siêu nhẹ, 23.7K tham số) + **Silero VAD** | Mặc định cho mọi bản build — gần như miễn phí về tài nguyên |
| C — Chất lượng cao hơn | DeepFilterNet3 (2.3M tham số, full-band 48kHz) | Nếu CPU còn dư sau khi đo phương án B |
| D — Nâng cao | Thêm **beamforming** (MVDR/GSC) nếu có mic array thật (vd ReSpeaker 4-mic) | Khi đã có phần cứng thật, để định hướng người nói giữa nhiều nguồn ồn |

### Paper cần đọc (theo thứ tự)
1. **GTCRN** — https://github.com/Xiaobin-Rong/gtcrn — đọc để hiểu kiến trúc + lấy pretrained weight, đây là lựa chọn mặc định.
2. **DeepFilterNet2/3** — arXiv 2205.05474 — đọc nếu cân nhắc phương án C, so sánh trade-off chất lượng/tài nguyên.
3. **CHiME-3 beamforming** — arXiv 1509.06103 — để hiểu VÌ SAO beamforming giảm WER (số liệu: giảm WER tương đối 10–60% tùy điều kiện), làm căn cứ khi thuyết phục team đầu tư vào mic array.
4. **Spatial Speech Translation** — arXiv 2504.18715 — chỉ đọc nếu muốn làm tính năng nâng cao "định hướng nhiều người nói cùng lúc" (không bắt buộc, để dành cho giai đoạn sau).

### Quy trình thử nghiệm (từng bước)
1. Chuẩn bị bộ test nhiễu: tải MUSAN, DEMAND, WHAM! (noise công khai) + tự thu noise nhà máy thật (tiếng máy dập, băng chuyền, xe nâng...).
2. Trộn (mix) giọng nói sạch với noise ở nhiều mức tỉ lệ tín hiệu/nhiễu: SNR = 0, 5, 10, 15, 20 dB.
3. Chạy audio nhiễu qua front-end, đo các chỉ số khách quan: **PESQ** (chất lượng cảm nhận), **STOI** (độ dễ hiểu), **DNSMOS** (điểm chất lượng không cần audio gốc). So sánh trước/sau xử lý.
4. Đo **RTF** (Real-Time Factor = thời gian xử lý / thời gian audio) trên đúng CPU của thiết bị đích — bắt buộc RTF < 1, mục tiêu < 0.1 (xử lý nhanh hơn 10 lần so với audio thật).
5. Đo chất lượng VAD: cho audio có nhiễu, đo tỉ lệ cắt sai (miss rate: bỏ sót giọng nói thật; false trigger: cắt nhầm noise thành giọng nói).
6. Nếu có mic array: đo hiệu quả beamforming bằng cách so WER của ASR (mục 1) khi có vs không có beamforming, ở cùng điều kiện nhiễu.

**Tiêu chí hoàn thành (DoD):** RTF < 0.1 trên CPU thiết bị; PESQ cải thiện rõ rệt so với không xử lý; VAD hoạt động ổn định ở SNR ≥ 5dB (miss rate < 5%).

---

## 1. ASR — Nhận dạng giọng nói (Speech → Text)

**Nhiệm vụ là gì:** Chuyển audio (đã làm sạch từ mục 0) thành chữ viết, chạy theo kiểu **streaming** — tức là chữ hiện ra dần trong lúc người ta đang nói, không phải đợi họ nói xong hết câu mới hiện. Phải làm được **cho cả 4 ngôn ngữ, cả 2 chiều** (không chỉ Việt → chữ, mà cả Anh/Trung/Hàn → chữ khi người nước ngoài nói vào máy).

**Vì sao quan trọng:** Đây là input của tầng dịch (MT). Nhận dạng sai chữ nào thì dịch sai theo chữ đó — MT không có cách nào biết "người ta thực ra nói câu khác".

**Input:** đoạn audio đã VAD cắt, mono, sạch (từ mục 0).
**Output:** text — cả bản "partial" (đang đoán, có thể còn sửa) và bản "final" (chốt) — kèm theo biết đó là ngôn ngữ nào.

### Các phương án
| Phương án | Mô tả | Ưu | Nhược |
|---|---|---|---|
| A | Whisper-small multilingual dùng chung cho cả 4 ngôn ngữ | Đơn giản, 1 model, dễ bảo trì | Không tối ưu riêng — Việt sẽ kém hơn model chuyên biệt |
| **B — Khuyến nghị** | **PhoWhisper** cho tiếng Việt (chuyên biệt, SOTA) + **SenseVoice-Small** cho Anh/Trung/Hàn (non-autoregressive, siêu nhanh: 70ms cho 10 giây audio) | Mỗi ngôn ngữ dùng đúng "chuyên gia"; SenseVoice cực nhanh nên streaming gần như miễn phí | 2 model khác nhau, phải quản lý routing theo ngôn ngữ |
| C | Canary (model 1B, dịch tiếng nói→chữ trực tiếp, theo paper "Pocket Offline") | 1 model đa ngôn ngữ, có nghiên cứu học thuật hỗ trợ | Cần TỰ kiểm tra xem có tốt cho tiếng Việt không — rủi ro chưa rõ |

**Chính sách streaming (chọn theo model):**
- Model tự hồi quy (PhoWhisper/Whisper): dùng **LocalAgreement-2** — so sánh 2 lần đoán liên tiếp, phần nào giống nhau ở đầu câu thì chốt (confirm), phần cuối vẫn để ngỏ chờ thêm audio.
- Model non-autoregressive (SenseVoice): vì mỗi lần đoán cả câu chỉ tốn 70ms, có thể đơn giản là **đoán lại toàn bộ mỗi 300ms** — không cần thuật toán phức tạp.

### Paper cần đọc (theo thứ tự)
1. **PhoWhisper** — arXiv 2406.02555 — model chính cho tiếng Việt, đọc để biết cách fine-tune và benchmark WER gốc.
2. **ViMD** (Vietnamese Multi-Dialect) — arXiv 2410.03458 — dataset **63 tỉnh thành**, dùng để fine-tune chống lỗi giọng địa phương — đọc kỹ phần cách họ đo WER tách theo vùng miền.
3. **Bud500** — https://huggingface.co/datasets/linhtran92/viet_bud500 — 500 giờ audio Bắc/Trung/Nam, bổ sung cho ViMD.
4. **SenseVoice** — https://github.com/FunAudioLLM/SenseVoice — đọc README để hiểu kiến trúc non-autoregressive và cách nó nhanh hơn Whisper 15 lần.
5. **Whisper-Streaming / SimulStreaming** — arXiv 2307.14743 — đọc để hiểu cơ chế LocalAgreement, áp dụng cho PhoWhisper.
6. (Tùy chọn, nếu cân nhắc phương án C) **Pocket Offline Model** — arXiv 2606.03948 — đọc phần họ dùng Canary + AlignAtt ra sao.

### Quy trình thử nghiệm (từng bước)
1. **Baseline**: chạy model gốc (chưa fine-tune) trên bộ test sạch (Common Voice tiếng Việt, FLEURS) → đo WER (Word Error Rate)/CER (Character Error Rate cho tiếng Trung).
2. **Fine-tune**: train tiếp trên ViMD + Bud500 (đa giọng) kết hợp **multi-condition training** — trộn noise nhà máy vào audio lúc train (không chỉ test).
3. **Đo lại WER, TÁCH THEO VÙNG MIỀN** (Bắc/Trung/Nam) — không được gộp chung thành 1 con số duy nhất, vì như vậy sẽ giấu mất chỗ yếu (ví dụ model có thể rất tốt giọng Bắc nhưng tệ giọng miền Tây, nếu gộp lại vẫn ra số đẹp mà không biết).
4. **Đo theo ma trận nhiễu**: WER ở SNR = clean/20/15/10/5/0 dB — vẽ thành 1 bảng hoặc biểu đồ đường.
5. **Đo streaming**: đo độ trễ tới khi có chữ "partial" đầu tiên xuất hiện; đo "flicker rate" — partial bị model sửa đi sửa lại bao nhiêu lần trước khi chốt (càng ít càng tốt, vì partial nhảy lung tung sẽ làm MT phía sau bị rối).
6. **Đo trên thiết bị thật** sau khi quantize (int8) — so sánh WER trước/sau quantize, phải không tụt quá 1–2 điểm phần trăm.

**Tiêu chí hoàn thành (DoD):** WER tiếng Việt < 8% (audio sạch), < 15% (SNR 5dB, không "sập"); WER đồng đều giữa 3 miền (không có miền nào chênh lệch quá lớn); WER/CER tiếng Anh/Trung/Hàn đạt mức tương đương benchmark công khai của model gốc.

---

## 2. MT — Dịch máy (Text → Text)

**Nhiệm vụ là gì:** Dịch chữ đã nhận dạng được sang ngôn ngữ đích, cũng chạy kiểu streaming — dịch dần khi ASR đẩy "partial" sang, không đợi ASR chốt hết cả câu. Đây được xem là "bộ não" của hệ thống, quyết định chất lượng giao tiếp thực sự giữa 2 người.

**Vì sao quan trọng:** Kể cả ASR/TTS hoàn hảo, nếu dịch sai nghĩa hoặc sai thuật ngữ an toàn lao động thì thiết bị vô dụng, thậm chí nguy hiểm.

**Input:** text partial/final từ ASR (kèm biết ngôn ngữ nguồn là gì) + ngôn ngữ đích cần dịch tới.
**Output:** text đã dịch (partial/final), đẩy sang TTS.

### Các phương án
| Phương án | Mô tả | Khi nào dùng |
|---|---|---|
| A | NLLB-600M nguyên bản, chưa fine-tune | Baseline nhanh, có ngay tuần đầu |
| **B — Khuyến nghị (chế độ "Live")** | NLLB-600M fine-tune trên data domain nhà máy + back-translation cho Việt↔Trung/Hàn + chính sách streaming **AlignAtt** | Mặc định khi hội thoại đang diễn ra (streaming) |
| C (chế độ "Accurate", tùy chọn) | Model lớn hơn (HY-MT1.5-1.8B, hậu duệ của model vô địch WMT2025) chạy khi KHÔNG cần streaming — ví dụ bấm nút "dịch chính xác" cho câu quan trọng | Bổ sung, không bắt buộc phase đầu |
| D | Distill: dùng model dịch cực mạnh (Hunyuan-MT-7B) chạy trên máy tính mạnh để **sinh dữ liệu dịch mẫu chất lượng cao**, rồi lấy dữ liệu đó dạy lại cho NLLB-600M nhỏ | Cách nâng chất lượng NLLB-600M mà không cần chạy model 7B trên thiết bị nhỏ |

**Chính sách streaming (khi nào dịch, khi nào chờ thêm):**
- **wait-k**: luật cố định, dịch sau khi thấy đủ k từ. Đơn giản nhất nhưng cứng nhắc.
- **AlignAtt** (khuyến nghị): nhìn vào "độ chú ý" (attention) của model để tự quyết định lúc nào đã đủ thông tin để dịch — không cần train lại model, linh hoạt hơn wait-k.
- **AliBaStr** (nâng cao): học hẳn một bộ "quyết định đọc/viết" riêng — chất lượng tốt nhất nhưng phải train thêm.

### Paper cần đọc (theo thứ tự)
1. **NLLB-200** — https://huggingface.co/facebook/nllb-200-distilled-1.3B — đọc model card, hiểu các ngôn ngữ hỗ trợ sẵn (vie_Latn, zho_Hans, kor_Hang).
2. **AlignAtt / SimulSeamless** — arXiv 2406.14177 — đọc kỹ, đây là chính sách streaming chính sẽ dùng.
3. **On-Device Speech Translation (AliBaStr-MT)** — arXiv 2508.13358 — đọc để hiểu blueprint tổng thể + số liệu latency thực tế đạt được (giảm 37–39% so với không streaming).
4. **Việt-Trung low-resource (mBART + back-translation)** — arXiv 2501.19314 — đọc kỹ, đây là công thức cụ thể để làm cặp Việt↔Trung tốt hơn Google Translate.
5. **VietMix** — arXiv 2505.24472 — đọc để xử lý câu trộn ngôn ngữ (code-switching), ví dụ công nhân nói lẫn Việt-Anh.
6. **Hunyuan-MT-7B** — arXiv 2509.05209 — đọc nếu dùng làm teacher model sinh dữ liệu.
7. (Tùy chọn) **Agent-SiMT** — arXiv 2406.06910 — đọc nếu làm 2 chế độ Live/Accurate như một hệ thống "agent" có phân công rõ vai trò.

### Quy trình thử nghiệm (từng bước)
1. **Baseline**: chạy NLLB gốc trên bộ test công khai FLORES-200 → đo BLEU và COMET cho từng cặp, từng chiều (6 hướng: Vi→En, En→Vi, Vi→Zh, Zh→Vi, Vi→Ko, Ko→Vi).
2. **Fine-tune từng cặp riêng biệt**: Việt-Anh (dùng bộ PhoMT có sẵn), Việt-Trung (VLSP 2022 + back-translation), Việt-Hàn (corpus AI Hub Hàn Quốc + back-translation/pivot qua tiếng Anh nếu thiếu data).
3. So sánh BLEU/COMET **trước và sau fine-tune** — phải thấy cải thiện rõ, không chỉ bằng cảm tính.
4. **Head-to-head vs Google Translate**: tự soạn bộ test 500 câu domain nhà máy (an toàn lao động, QC, máy móc), dịch chuẩn bởi người, rồi so cả model của mình và Google Translate cùng bộ test này — ra bảng số trực tiếp.
5. **Test streaming**: dùng công cụ **SimulEval** đo Average Lag (AL) và LAAL, vẽ đường cong latency-quality bằng cách quét tham số δ của AlignAtt (δ càng cao càng chính xác nhưng càng trễ).
6. **Test thuật ngữ chuyên ngành**: bộ câu chứa từ khóa an toàn lao động, kiểm tra tỉ lệ dịch đúng thuật ngữ (không được dịch sai ý — ví dụ từ cảnh báo nguy hiểm).
7. **Test code-switch**: câu trộn Việt-Anh-Hàn trong cùng 1 câu.

**Tiêu chí hoàn thành (DoD):** BLEU/COMET ở Việt↔Trung/Hàn bằng hoặc vượt Google Translate trên bộ test domain; Average Lag đạt mục tiêu latency đề ra; thuật ngữ chuyên ngành dịch đúng > 95%.

---

## 3. TTS — Tổng hợp giọng nói (Text → Speech)

**Nhiệm vụ là gì:** Chuyển chữ đã dịch thành giọng nói phát ra loa, streaming theo từng cụm từ/mệnh đề (không đợi có hết cả câu dịch mới bắt đầu nói). Cần **cả 4 giọng** (Việt, Anh, Trung, Hàn) vì máy phải nói được cả 2 chiều.

**Vì sao quan trọng:** Đây là trải nghiệm cuối cùng người dùng nghe được — dù ASR và MT đúng hoàn hảo, giọng đọc dở/giật cục/sai thuật ngữ vẫn khiến sản phẩm bị đánh giá thấp.

**Input:** text đã dịch (partial/final theo cụm từ) + ngôn ngữ đích.
**Output:** audio phát trực tiếp ra loa.

### Các phương án
| Phương án | Mô tả | Lưu ý |
|---|---|---|
| A | Piper cho tất cả 4 ngôn ngữ | ⚠️ Piper **không có giọng Hàn tốt** — rủi ro |
| **B — Khuyến nghị** | **Piper** cho Việt + Anh, **MeloTTS** cho Trung + Hàn | MeloTTS hỗ trợ chính thức Hàn/Trung, chạy real-time trên CPU |
| C | viXTTS thay Piper cho tiếng Việt | Giọng tự nhiên hơn nhưng nặng hơn — cân nhắc RAM còn dư không |
| D (stretch, rủi ro cao) | Voice cloning — giữ giọng gốc người nói khi dịch (X-Voice, MiniMax-Speech) | Ấn tượng nhưng thêm độ phức tạp + rủi ro latency; để dành giai đoạn sau nếu còn thời gian |

### Paper cần đọc (theo thứ tự)
1. **Piper** — https://github.com/rhasspy/piper — đọc README, lấy voice tiếng Việt + Anh có sẵn.
2. **MeloTTS** — https://github.com/myshell-ai/MeloTTS — đọc README, chú ý bản MeloTTS-ONNX để chạy CPU nhanh.
3. (Tùy chọn nâng cấp Việt) **viXTTS** — https://huggingface.co/capleaf/viXTTS.
4. (Tùy chọn, stretch) **X-Voice** — arXiv 2605.05611, **MiniMax-Speech** — arXiv 2505.07916 — chỉ đọc nếu quyết định thử voice cloning.

### Quy trình thử nghiệm (từng bước)
1. **Nghe thử chủ quan (MOS — Mean Opinion Score)**: cho 5–10 người nghe từng giọng, từng ngôn ngữ, chấm điểm 1–5. Đây là bước không thể bỏ qua vì số liệu khách quan không phản ánh hết "nghe có tự nhiên không".
2. Đo **RTF** trên CPU thiết bị đích — phải rất nhỏ (< 0.05, tức 1 giây audio tổng hợp trong dưới 50ms).
3. Đo **độ trễ từ lúc có text đến lúc có audio đầu tiên** phát ra (vì chạy streaming theo cụm từ).
4. **Test đọc số, tên riêng, thuật ngữ chuyên ngành** — đây là chỗ TTS hay đọc sai nhất (ví dụ số đo, mã sản phẩm).
5. Nếu làm voice cloning (phương án D): đo độ giống giọng gốc (cosine similarity giữa embedding giọng gốc và giọng tổng hợp) + đo MOS độ tự nhiên riêng.

**Tiêu chí hoàn thành (DoD):** RTF < 0.05; MOS > 3.5/5; độ trễ audio đầu tiên < 100ms sau khi có text; đọc đúng số/thuật ngữ.

---

## 4. Integration/Streaming Pipeline — Ghép nối hệ thống real-time

**Nhiệm vụ là gì:** Là "nhạc trưởng" nối 4 khối (Front-end → ASR → MT → TTS) thành một hệ thống chạy **thật sự song song**, không phải nối tuần tự (chạy xong ASR mới chạy MT, chạy xong MT mới chạy TTS) — vì nối tuần tự sẽ khiến độ trễ **cộng dồn** của cả 4 tầng, chậm hơn nhiều so với chạy chồng lấn.

**Vì sao đây là phần khó nhất:** Không có "model" mới nào ở đây, nhưng đây là nơi **đa số đội thi sẽ gặp khó khăn nhất** — vì mỗi tầng chạy tốc độ khác nhau, ra dữ liệu (partial) ở nhịp khác nhau, phải đồng bộ mà không bị tràn bộ nhớ, không bị trễ dồn theo thời gian khi nói liên tục.

**Input:** audio thô liên tục từ mic.
**Output:** audio dịch liên tục phát ra loa, độ trễ thấp và ổn định.

### Các phương án kiến trúc
| Phương án | Mô tả | Đánh giá |
|---|---|---|
| A | Pipeline tuần tự đơn giản (chờ từng bước xong mới sang bước sau) | Dễ code, nhưng độ trễ = tổng cả 4 tầng cộng lại — KHÔNG dùng cho bản chính thức, chỉ để test nhanh từng khối riêng |
| **B — Khuyến nghị** | Pipeline theo chunk, mỗi tầng chạy trên 1 luồng/tiến trình riêng, giao tiếp qua hàng đợi (queue) — ASR xử lý đoạn n+1 cùng lúc MT đang dịch đoạn n, cùng lúc TTS đang đọc đoạn n−1 | Đúng như blueprint đã được kiểm chứng, tận dụng NPU+CPU chạy song song |
| C | Kiến trúc hướng sự kiện (event-driven) — mỗi tầng phát ra sự kiện khi có kết quả partial mới, tầng sau lắng nghe | Linh hoạt hơn B nhưng phức tạp hơn khi debug |

### Paper cần đọc
1. **On-Device Speech Translation (AliBaStr-MT)** — arXiv 2508.13358 — đọc kỹ phần họ mô tả **"context flushing"** (giải phóng bớt bộ nhớ khi gần chạm giới hạn) và cách các tầng chồng lấn nhau — đây là paper DUY NHẤT trong danh sách bàn trực tiếp về vấn đề tích hợp/bộ nhớ trên thiết bị thật, rất đáng đọc kỹ thay vì đọc lướt.

### Quy trình thử nghiệm (từng bước)
1. Test độ trễ từng tầng riêng lẻ trước (đã đo ở mục 0–3) — đây là số liệu nền để so sánh.
2. Ghép 2 tầng một: audio → ASR → MT (hiện chữ dịch ra màn hình, CHƯA nối TTS) — đo độ trễ cộng dồn của 2 tầng.
3. Ghép full pipeline (cả 4 tầng) — đo **first-audio-out latency** (từ lúc bắt đầu nói tới lúc nghe tiếng dịch đầu tiên) và **full-utterance latency** (tới lúc dịch xong cả câu), với cả câu ngắn và câu dài.
4. **Stress test**: cho nói liên tục không ngừng nghỉ trong nhiều phút — kiểm tra pipeline có bị "trễ dồn" tăng dần theo thời gian không (dấu hiệu hàng đợi bị tràn, xử lý không kịp).
5. Test các trường hợp biên (edge case): nói rất nhanh, nói rất chậm, im lặng giữa chừng rồi nói tiếp, 2 người nói chồng lên nhau.
6. Đo mức dùng RAM/CPU/NPU khi chạy pipeline liên tục 30 phút — kiểm tra có bị rò rỉ bộ nhớ (memory leak, RAM tăng dần không giảm) không.

**Tiêu chí hoàn thành (DoD):** first-audio-out < 1.5 giây; không bị trễ dồn tăng dần theo thời gian khi nói liên tục; RAM ổn định, không tăng dần không kiểm soát.

---

## 5. Data & Fine-tuning

**Nhiệm vụ là gì:** Thu thập, làm sạch, tăng cường (augment) dữ liệu huấn luyện; fine-tune từng model (ASR ở mục 1, MT ở mục 2); đánh giá chất lượng model sau khi train xong, trước khi giao cho người làm Integration ráp vào hệ thống.

**Vì sao quan trọng:** Model tốt đến đâu cũng thua nếu dữ liệu train tệ hoặc thiếu đúng loại cần thiết (giọng địa phương, noise nhà máy, thuật ngữ chuyên ngành) — đây là nguyên tắc "garbage in, garbage out".

### Việc cụ thể
- Tải các bộ dữ liệu công khai: ViMD, Bud500, VLSP, FLORES-200, PhoMT, OPUS (OpenSubtitles, CCMatrix...).
- Liên hệ/xin quyền truy cập corpus Việt-Hàn từ AI Hub Hàn Quốc (có thể cần thời gian xử lý thủ tục, nên bắt đầu sớm).
- **Đi thực địa thu âm noise nhà máy thật** (5–10 giờ): tiếng máy dập, băng chuyền, xe nâng, tiếng loa thông báo, tiếng đông người — nên xin hỗ trợ kết nối nhà máy qua chính đơn vị tổ chức cuộc thi.
- Sinh dữ liệu tổng hợp: back-translation (dịch ngược để tạo thêm cặp câu), distill từ teacher model lớn (Hunyuan-MT-7B).
- **Làm bộ test riêng, giữ kín**: 500 câu hội thoại domain nhà máy mỗi cặp ngôn ngữ, dịch chuẩn bởi người — bộ này **không được đưa vào lúc train**, chỉ dùng để đánh giá cuối cùng.

### Paper cần đọc
Xem lại danh sách ở mục 1 (ViMD, Bud500) và mục 2 (Việt-Trung low-resource, VietMix, Hunyuan-MT) — người phụ trách Data nên đọc chung với người phụ trách ASR/MT vì công việc gắn chặt với nhau.

### Quy trình
1. Kiểm kê: liệt kê dữ liệu đã có sẵn vs dữ liệu còn thiếu (gap analysis) cho từng ngôn ngữ/cặp.
2. Làm sạch: loại câu trùng lặp, câu quá ngắn/quá dài, lọc sai ngôn ngữ.
3. **Chia tập train/dev/test rõ ràng — tập test niêm phong, không ai được nhìn vào lúc train** (đây là bài học từ kinh nghiệm thi trước: model có thể đạt điểm rất cao trên tập test nếu vô tình học lẫn dữ liệu test, nhưng sẽ sập khi gặp dữ liệu thật ngoài đời — không được lặp lại sai lầm này).
4. Tăng cường dữ liệu: trộn noise, back-translation, tạo câu code-switch.
5. Đặt tên phiên bản dữ liệu rõ ràng (v1, v2...) để có thể lặp lại thí nghiệm khi cần.

**Tiêu chí hoàn thành (DoD):** có bộ test niêm phong hoàn chỉnh cho cả 3 cặp ngôn ngữ trước khi ai đó bắt đầu train chính thức; augmentation áp dụng nhất quán giữa các lần train.

---

## 6. Deploy & Quản lý phần cứng

**Nhiệm vụ là gì:** Đưa các model đã train (mục 1, 2, 3) chạy thật trên chip Snapdragon, tối ưu tốc độ/bộ nhớ/điện năng tiêu thụ; lắp ráp và vận hành thiết bị vật lý.

**Vì sao quan trọng:** Model chạy mượt trên máy tính có GPU mạnh **chưa chắc chạy được** trên chip NPU của điện thoại/thiết bị nhúng — đây là nơi nhiều đội thi "vỡ trận" vì để việc này tới sát ngày mới làm.

### Việc cụ thể
- Convert model sang định dạng chạy trên NPU (QNN) thông qua **Qualcomm AI Hub**.
- Quantize (nén số học, ví dụ int8/int4) để giảm dung lượng và tăng tốc — sau đó phải kiểm tra độ chính xác không bị tụt quá nhiều.
- Lắp ráp thiết bị vật lý (nếu dùng board rời như Rubik Pi 3): gắn board, mic array, pin, vỏ.
- Viết script/driver quản lý luồng âm thanh vào/ra trên thiết bị thật.

### Paper/tài liệu cần đọc
1. **Qualcomm AI Hub Models** — https://github.com/qualcomm/ai-hub-models — đọc hướng dẫn convert model.
2. **Whisper-Base benchmark trên Snapdragon** — https://huggingface.co/qualcomm/Whisper-Base — đọc để có số liệu latency/RAM tham khảo (encoder ~26ms trên Snapdragon 8 Elite).
3. Tài liệu **Rubik Pi 3** (nếu dùng board này) — https://rubikpi.ai/ — đọc hướng dẫn cài đặt và GPIO/mic.

### Quy trình thử nghiệm
1. Convert từng model một, đo latency/RAM **trên thiết bị thật** (không phải giả lập trên máy tính) — số liệu giả lập thường sai lệch nhiều so với thực tế.
2. So sánh WER/BLEU **trước và sau quantize** — quy định trước ngưỡng chấp nhận được (ví dụ không tụt quá 1–2 điểm phần trăm), nếu tụt quá thì phải thử mức quantize nhẹ hơn.
3. Test chạy liên tục 4 giờ, đo mức pin hao hụt.
4. Test **airplane mode** — đảm bảo hệ thống chạy 100% offline thật, không lỡ gọi API nào ra ngoài (kiểm tra bằng cách theo dõi lưu lượng mạng khi chạy).
5. Test nhiệt độ máy khi chạy liên tục lâu — kiểm tra có bị giảm xung nhịp (throttle) làm chậm dần không.

**Tiêu chí hoàn thành (DoD):** mọi model chạy được trên NPU/CPU của thiết bị đích; pin trên 4 giờ sử dụng liên tục; không throttle nghiêm trọng; xác nhận 100% offline.

---

## 7. Testing & Đánh giá tổng

**Nhiệm vụ là gì:** Đóng vai "người ngoài cuộc" kiểm tra chéo toàn bộ hệ thống — bao gồm test từng phần riêng (unit test), test khi ghép lại (integration test), test ở môi trường thật (field test), và so sánh với đối thủ (Google Translate).

**Nguyên tắc quan trọng:** người test một module **không nên** là người trực tiếp code module đó — để tránh việc tự đánh giá thiên vị (dễ bỏ qua lỗi của chính mình).

### Việc cụ thể
- Tổng hợp toàn bộ kết quả test đã mô tả rải rác ở các mục 0–6 thành **một bảng/dashboard số liệu duy nhất**, cập nhật thường xuyên — dùng để nộp báo cáo kỹ thuật và theo dõi tiến độ.
- Tổ chức field test thực địa trước ngày ban tổ chức chấm điểm chính thức (đừng để lần đầu ra thực địa lại chính là ngày thi).
- So sánh trực tiếp (head-to-head) với Google Translate trên cùng bộ test domain, quay video làm bằng chứng.

**Tiêu chí hoàn thành (DoD):** có một bảng số liệu tổng hợp duy nhất, đầy đủ và cập nhật; đã chạy field test thực địa ít nhất 1 lần thành công trước ngày thi.

---

## 8. Business/Demo

**Nhiệm vụ là gì:** Kể câu chuyện sản phẩm ra bên ngoài — chuẩn bị kịch bản demo, slide thuyết trình, tính toán chi phí (BOM), và làm dashboard hiển thị số liệu (độ trễ, điện năng) trực tiếp khi trình diễn.

**Vì sao không được bỏ qua:** đây là 25% điểm số ("Business Impact") — nhiều đội kỹ thuật giỏi bị mất điểm ở đây vì chỉ tập trung vào model mà quên phần trình bày.

### Việc cụ thể
- Viết kịch bản demo cụ thể (ai nói gì, tình huống nào minh họa cho điểm mạnh nào).
- Chuẩn bị số liệu để trình bày: bảng so sánh với Google Translate, biểu đồ độ trễ, biểu đồ độ chính xác theo điều kiện nhiễu.
- Làm dashboard hiện số liệu độ trễ/điện năng ngay trên màn hình thiết bị lúc demo trực tiếp.
- Tính chi phí linh kiện (BOM) và câu chuyện thương mại (ai sẽ mua, vì sao offline quan trọng với họ).

**Tiêu chí hoàn thành (DoD):** kịch bản demo đã tập dượt ít nhất 3 lần trước ngày thi; có đầy đủ slide + số liệu dẫn chứng; dashboard số liệu chạy ổn định trên thiết bị thật.


Kính gửi Ban quản lý tài nguyên VLSP,

Em/tôi là [TÊN CỦA BẠN], hiện đang tham gia dự án nghiên cứu OneVoice — xây dựng thiết bị dịch giọng nói đa ngôn ngữ (Việt-Anh-Trung-Hàn) chạy offline trên chip Qualcomm Snapdragon, trong khuôn khổ cuộc thi [TÊN CUỘC THI nếu có].

Em/tôi xin phép được truy cập các bộ dữ liệu sau để phục vụ nghiên cứu:

VLSP 2020 ASR — Dữ liệu nhận dạng giọng nói tiếng Việt (~250 giờ)

Mục đích: Fine-tune model PhoWhisper cho nhận dạng giọng nói đa phương ngữ
VLSP 2022 Machine Translation (Vi-Zh) — Dữ liệu dịch máy Việt-Trung (>300K cặp câu)

Mục đích: Fine-tune model NLLB-600M cho dịch Việt↔Trung chuyên ngành công nghiệp
Thông tin người yêu cầu:

Họ tên: [TÊN]
Đơn vị: [TRƯỜNG/TỔ CHỨC]
Email: [EMAIL]
Số điện thoại: [SĐT]
Cam kết sử dụng:

Dữ liệu chỉ được sử dụng cho mục đích nghiên cứu phi thương mại
Không phân phối lại cho bên thứ ba
Tuân thủ đầy đủ các điều khoản sử dụng của VLSP
Em/tôi sẵn sàng ký Data User Agreement nếu cần.

Xin chân thành cảm ơn!

Trân trọng, [TÊN] [SĐT / EMAIL]