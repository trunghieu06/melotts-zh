# MeloTTS on Qualcomm NPU (Dragonwing IQ-9075 EVK)

Thư mục này tổng hợp toàn bộ mã nguồn, mô hình nhị phân, sổ tay huấn luyện, tệp âm thanh kiểm chứng và báo cáo kỹ thuật của phân hệ **MeloTTS trên Qualcomm Hexagon NPU (HTP v73, QAIRT 2.45.0)**.

---

## 📁 Cấu Trúc Thư Mục

```
melotts_npu/
├── README.md                      # Hướng dẫn tổng quan thư mục
├── docs/                          # Toàn bộ báo cáo kỹ thuật & đối soát
│   ├── bao_cao_toan_dien_loi_du_an.md    # [QUAN TRỌNG] Báo cáo mổ xẻ nguyên nhân lỗi toán học
│   ├── bao_cao_evaluation_melo_npu.md    # Báo cáo đánh giá đối soát vs FP32 Baseline
│   ├── bao_cao_problem1.md               # Báo cáo Problem 1 (Byte Tokenizer & LLM trên EVK)
│   ├── bao_cao_problem1_npu.md           # Số liệu đo đạc telemetry phần cứng Problem 1
│   └── bao_cao_melotts.md                # Báo cáo cấu hình gốc MeloTTS
│
├── scripts/                       # Mã nguồn thực thi & kiểm chứng phần cứng
│   ├── phase1_compile_profile_byte_encoder.py  # Compile & Profile ByteEncoder trên AI Hub
│   ├── phase2_pipeline_e2e_npu.py              # Ghép nối E2E pipeline trên NPU
│   ├── run_clean_e2e_npu_evk.py                # Chạy chuỗi Flow + Decoder trên NPU EVK
│   ├── evaluate_melo_npu_vs_baseline.py        # Đo đạc chỉ số Spectral Convergence & Log-Mel
│   ├── verify_decoder_cosine.py                # Kiểm chứng Cosine Similarity của Vocoder
│   ├── verify_problem1_npu.py                  # Kiểm chứng độ chính xác Problem 1
│   └── train_distill_byte_encoder.py           # Mã nguồn PyTorch chưng cất ByteEncoder
│
├── notebooks/                     # Sổ tay huấn luyện trên Kaggle / GPU
│   └── Train_Byte_Encoder_MeloTTS.ipynb        # Notebook gốc đã train 10 tiếng trên Kaggle
│
├── models/                        # Trọng số & tệp nhị phân NPU
│   ├── melo_byte_encoder_npu.onnx              # Graph ONNX tĩnh (25.45M tham số)
│   ├── melo_byte_encoder_npu.dlc               # Nhị phân QNN DLC biên dịch cho HTP v73
│   ├── byte_encoder_best.pth                   # Checkpoint PyTorch trọng số tốt nhất
│   └── melotts-zh/                             # Bộ nhị phân chuẩn Qualcomm AI Hub
│       ├── encoder.bin                         # QNN TextEncoder (18.4 MB)
│       ├── flow.bin                            # QNN Normalizing Flow (29.7 MB)
│       ├── decoder.bin                         # QNN HiFi-GAN Vocoder (19.3 MB)
│       ├── spk_emb_neutral.npy                 # Vector đặc trưng người nói chuẩn
│       └── metadata.json                       # Định nghĩa tensor I/O và tham số lượng tử
│
└── audio_eval/                    # Tệp âm thanh kiểm chứng thực tế
    ├── reference_fp32_melo.wav                 # Bản âm thanh chuẩn PyTorch FP32 CPU
    ├── melo_npu_evk_clean_final.wav            # Âm thanh NPU sạch từ chip EVK thật (0% bíp)
    ├── melo_optionA_snapdragon.wav             # Âm thanh lỗi bị tiếng bíp 430.7 Hz (đối chứng)
    └── melo_clean_vs_broken_comparison.json    # Dữ liệu JSON đo đạc phổ STFT & Log-Mel
```

---

## 📌 Tóm Tắt Kết Quả & Lưu Ý Kỹ Thuật

1. **Khối phần cứng Vocoder NPU (`decoder.bin`):** Hoạt động hoàn hảo trên chip **Dragonwing IQ-9075 EVK** với độ trễ chỉ **6.84 ms** (nhanh gấp 16 lần CPU). Khi nhận đúng ma trận $z$, âm thanh xuất ra trong trẻo, F0 = 249 Hz, không hề có tiếng bíp.
2. **Mô hình `melo_byte_encoder_npu`:** Hiện tại chưa thể dùng để tổng hợp giọng nói do lỗi cắt cụt token và lệch pha trong notebook huấn luyện Kaggle (xem chi tiết tại `docs/bao_cao_toan_dien_loi_du_an.md`).
3. **Giải pháp đề xuất cho OneVoice:** Sử dụng frontend âm vị chuẩn chạy nhẹ trên CPU (~2 ms), toàn bộ phần tính toán nặng (Encoder, Flow, Decoder) giao cho NPU xử lý qua các tệp nhị phân trong thư mục `models/melotts-zh/`.
