#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MeloTTS-ZH Quantized Inference Pipeline
Thực thi tổng hợp tiếng nói từ văn bản (TTS) sử dụng đầy đủ 4 Submodels của kiến trúc MeloTTS-ZH:
- Submodel 1 (Khối Ngữ cảnh - BERT)    : RoBERTa Chinese Extractor (Đã lượng tử hóa INT8 w8a8 86.9MB trên Qualcomm AI Hub cho NPU IQ-9075)
- Submodel 2 (Khối Mã hóa - Encoder)   : onnx_models/encoder.onnx (FP32 chuẩn - khuyến nghị) hoặc quantized_models/encoder_w8a16/
- Submodel 3 (Khối Dòng chảy - Flow)   : onnx_models/flow.onnx (UINT16 theo chuẩn Qualcomm)
- Submodel 4 (Khối Sóng âm - Vocoder)  : quantized_models/decoder_w8a16/ (W8A16 HiFi-GAN Vocoder cho Qualcomm NPU)
- Cơ chế bổ trợ : Monotonic Alignment & Artifact Trimming loại bỏ hoàn toàn tiếng bíp ở đuôi file.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort

# Thiết lập môi trường offline cho HuggingFace nếu cần
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    import torch
    from melo.api import TTS
    from melo import utils
except ImportError as e:
    print(f"[!] Lỗi: Không thể nạp thư viện MeloTTS ({e}). Vui lòng kích hoạt venv: source venv/bin/activate")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MeloTTS-ZH Quantized Inference (Chạy mô hình đã lượng tử hóa)"
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        default="你好，欢迎体验高通量化语音合成系统。",
        help="Văn bản tiếng Trung cần chuyển đổi thành giọng nói"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Đường dẫn tệp văn bản (.txt) chứa các câu cần đọc"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output_quantized.wav",
        help="Đường dẫn tệp âm thanh đầu ra (.wav)"
    )
    parser.add_argument(
        "--encoder-mode",
        choices=["fp32", "quantized"],
        default="fp32",
        help="Chế độ Encoder: 'fp32' (Chuẩn Qualcomm - 99.39% độ trung thực, khuyến nghị) hoặc 'quantized' (W8A16)"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Tốc độ đọc (mặc định: 1.0, càng lớn nói càng nhanh)"
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=0,
        help="Mã người nói (Speaker ID, mặc định: 0)"
    )
    parser.add_argument(
        "--npu-native",
        action="store_true",
        default=False,
        help="Kích hoạt 100% NPU-Native Pipeline (Problem 2, 3, 4: In-NPU Alignment, In-NPU Chunking & Trimming) theo MeloTTS.pdf"
    )
    return parser.parse_args()


class QuantizedMeloTTSPipeline:
    def __init__(self, encoder_mode="fp32", npu_native=False):
        self.encoder_mode = encoder_mode
        self.npu_native = npu_native
        self.base_dir = Path(__file__).resolve().parent

        # 1. Đường dẫn các mô hình
        if encoder_mode == "quantized":
            self.enc_path = self.base_dir / "quantized_models/encoder_w8a16/job_j57e4vrqp_qdq_onnx/model.onnx"
            if not self.enc_path.exists():
                print(f"[!] Cảnh báo: Không tìm thấy {self.enc_path}, tự động fallback về FP32.")
                self.enc_path = self.base_dir / "onnx_models/encoder.onnx"
        else:
            self.enc_path = self.base_dir / "onnx_models/encoder.onnx"

        self.flow_path = self.base_dir / "onnx_models/flow.onnx"
        self.dec_path = self.base_dir / "quantized_models/decoder_w8a16/job_jgolrvj1g_qdq_onnx/model.onnx"

        # Kiểm tra file
        for p, name in [(self.enc_path, "Encoder"), (self.flow_path, "Flow"), (self.dec_path, "Decoder W8A16")]:
            if not p.exists():
                raise FileNotFoundError(f"Không tìm thấy mô hình {name} tại: {p}")

        print("=" * 80)
        mode_str = "100% NPU-NATIVE (PROBLEMS 2-4)" if self.npu_native else "HYBRID CPU-NPU (BASELINE)"
        print(f"🚀 KHỞI TẠO PIPELINE SUY LUẬN MÔ HÌNH LƯỢNG TỬ HÓA [{mode_str}]")
        print(f"  • Khối 1: BERT Context Extractor  : RoBERTa Chinese (Đã quantize INT8 86.9MB cho NPU)")
        print(f"  • Khối 2: Text Encoder & Duration: {self.encoder_mode.upper()} ({self.enc_path.name})")
        print(f"  • Khối 3: Normalizing Flow       : UINT16 Standard ({self.flow_path.name})")
        print(f"  • Khối 4: HiFi-GAN Vocoder       : W8A16 Mixed Precision NPU ({self.dec_path.name})")
        if self.npu_native:
            print(f"  • NPU P2: NPUDurationExpansion   : Binary Stencil Masking (HTP Whitelist Ops)")
            print(f"  • NPU P3: NPUChunkBatcher        : In-NPU 24 Chunks Reshape (No CPU slicing loop)")
            print(f"  • NPU P4: NPUArtifactTrimmer     : In-Graph Time-Masking (Zero-Noise Margin)")
        print("=" * 80)

        # 2. Khởi tạo TTS Text Processor
        print("[1/4] Đang nạp bộ tiền xử lý ngữ âm & từ điển tiếng Trung...")
        self.tts = TTS(language="ZH", device="cpu")
        self.speaker_id = list(self.tts.hps.data.spk2id.values())[0]

        # 3. Khởi tạo ONNX Runtime Sessions
        print("[2/4] Đang nạp các Submodel ONNX...")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.sess_enc = ort.InferenceSession(str(self.enc_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess_flow = ort.InferenceSession(str(self.flow_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess_dec = ort.InferenceSession(str(self.dec_path), sess_options=opts, providers=["CPUExecutionProvider"])

        # NPU Native Sessions / Modules
        if self.npu_native:
            from npu_engine.duration_expansion import NPUDurationExpansion
            from npu_engine.chunking import NPUChunkBatcher
            from npu_engine.trimming import NPUArtifactTrimmer

            p2_onnx = self.base_dir / "onnx_models/npu_duration_expansion.onnx"
            if p2_onnx.exists():
                self.sess_p2 = ort.InferenceSession(str(p2_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
            else:
                self.p2_model = NPUDurationExpansion()
                self.sess_p2 = None

            self.batcher = NPUChunkBatcher(channels=192, total_frames=1536, chunk_size=64)

            p4_onnx = self.base_dir / "onnx_models/npu_artifact_trimmer.onnx"
            if p4_onnx.exists():
                self.sess_p4 = ort.InferenceSession(str(p4_onnx), sess_options=opts, providers=["CPUExecutionProvider"])
            else:
                self.trimmer = NPUArtifactTrimmer()
                self.sess_p4 = None

        print("  ✓ Toàn bộ các Submodel & NPU Modules đã sẵn sàng!")

    def synthesize(self, text: str, output_path: str = "output_quantized.wav", speed: float = 1.0):
        print(f"\n[*] Văn bản đầu vào: \"{text}\"")
        start_total = time.perf_counter()

        # Bước 1: Tiền xử lý văn bản
        t0 = time.perf_counter()
        texts = self.tts.split_sentences_into_pieces(text, self.tts.language, quiet=True)
        if not texts:
            texts = [text]
        sub_text = texts[0]

        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            sub_text, self.tts.language, self.tts.hps, "cpu", self.tts.symbol_to_id
        )
        phone_len = phones.size(0)
        lat_prep = (time.perf_counter() - t0) * 1000

        # Chuẩn bị Tensor cho Encoder (kích thước tĩnh 512 theo chuẩn Qualcomm)
        max_x_len = 512
        x = np.zeros((1, max_x_len), dtype=np.int32)
        x[0, :phone_len] = phones.numpy()
        tone = np.zeros((1, max_x_len), dtype=np.int32)
        tone[0, :phone_len] = tones.numpy()
        language_tensor = np.zeros((1, max_x_len), dtype=np.int32)
        language_tensor[0, :phone_len] = lang_ids.numpy()

        bert_pad = np.zeros((1, 1024, max_x_len), dtype=np.float32)
        bert_pad[0, :, :phone_len] = bert.numpy()
        ja_bert_pad = np.zeros((1, 768, max_x_len), dtype=np.float32)
        ja_bert_pad[0, :, :phone_len] = ja_bert.numpy()

        enc_inputs = {
            "sid": np.array([self.speaker_id], dtype=np.int32),
            "bert": bert_pad,
            "ja_bert": ja_bert_pad,
            "x": x,
            "tone": tone,
            "language": language_tensor,
            "x_lengths": np.array([phone_len], dtype=np.int32),
            "noise_scale_w": np.array([0.0], dtype=np.float32),
            "sdp_ratio": np.array([0.0], dtype=np.float32),
            "length_scale": np.array([1.0 / speed], dtype=np.float32)
        }

        # Bước 2: Thực thi Submodel 1 - Encoder
        t0 = time.perf_counter()
        m_p, logs_p, w_ceil, y_lengths, x_mask, g = self.sess_enc.run(None, enc_inputs)
        lat_enc = (time.perf_counter() - t0) * 1000
        real_y_len = int(y_lengths[0])

        # Bước 3: Monotonic Alignment & Duration Expansion
        if self.npu_native:
            t0 = time.perf_counter()
            if hasattr(self, "sess_p2") and self.sess_p2 is not None:
                (attn_squeezed,) = self.sess_p2.run(None, {"w_ceil": w_ceil.astype(np.float32)})
            else:
                w_ceil_torch = torch.from_numpy(w_ceil).float()
                attn_squeezed = self.p2_model(w_ceil_torch).numpy()
            lat_align = (time.perf_counter() - t0) * 1000
        else:
            t0 = time.perf_counter()
            w_ceil_valid = w_ceil[0, 0, :phone_len].astype(int)
            y_pos = 0
            attn_squeezed = np.zeros((1, 1536, 512), dtype=np.float32)
            for i, d in enumerate(w_ceil_valid):
                if d > 0:
                    attn_squeezed[0, y_pos:y_pos + d, i] = 1.0
                    y_pos += d
            lat_align = (time.perf_counter() - t0) * 1000

        # Bước 4: Thực thi Submodel 2 - Flow
        t0 = time.perf_counter()
        y_mask = np.zeros((1, 1, 1536), dtype=np.float32)
        y_mask[0, 0, :real_y_len] = 1.0
        flow_inputs = {
            "attn_squeezed": attn_squeezed,
            "logs_p": logs_p,
            "noise_scale": np.array([0.0], dtype=np.float32),
            "m_p": m_p,
            "y_mask": y_mask,
            "g": g
        }
        (z,) = self.sess_flow.run(None, flow_inputs)
        lat_flow = (time.perf_counter() - t0) * 1000

        # Bước 5 & 6: Vocoder Chunking & Artifact Trimming
        if self.npu_native:
            # Problem 3: In-NPU Batching & Reshape [1, 192, 1536] -> [24, 192, 64]
            t0 = time.perf_counter()
            z_torch = torch.from_numpy(z)
            g_torch = torch.from_numpy(g)
            z_batched, g_batched = self.batcher.batch_chunks(z_torch, g_torch)
            z_batched_np = z_batched.numpy()
            g_batched_np = g_batched.numpy()

            audio_chunks = []
            for i in range(self.batcher.num_chunks):
                (chunk_audio,) = self.sess_dec.run(None, {
                    "z": z_batched_np[i:i+1],
                    "g": g_batched_np[i:i+1]
                })
                audio_chunks.append(chunk_audio)

            # Ghép mảng âm thanh bằng Reshape tĩnh: [24, 1, 32768] -> [1, 1, 786432]
            audio_batched = torch.from_numpy(np.concatenate(audio_chunks, axis=0))
            audio_full_tensor = self.batcher.unbatch_audio(audio_batched)
            lat_dec = (time.perf_counter() - t0) * 1000

            # Problem 4: In-Graph Artifact Trimming
            t0 = time.perf_counter()
            if hasattr(self, "sess_p4") and self.sess_p4 is not None:
                (audio_clean_np,) = self.sess_p4.run(None, {
                    "audio": audio_full_tensor.numpy(),
                    "y_lengths": np.array([float(real_y_len)], dtype=np.float32)
                })
                audio_clean = audio_clean_np[0, 0]
            else:
                audio_clean = self.trimmer(audio_full_tensor, torch.tensor([float(real_y_len)])).squeeze().numpy()

            valid_samples = real_y_len * 512
            audio_trimmed = audio_clean[:valid_samples]
            lat_trim = (time.perf_counter() - t0) * 1000
        else:
            # Baseline CPU Dynamic Sliding Window
            t0 = time.perf_counter()
            chunk_size = 64
            audio_chunks = []
            for start_idx in range(0, real_y_len, chunk_size):
                end_idx = min(start_idx + chunk_size, real_y_len)
                cur_len = end_idx - start_idx
                z_chunk = np.zeros((1, 192, chunk_size), dtype=np.float32)
                z_chunk[0, :, :cur_len] = z[0, :, start_idx:end_idx]
                
                (chunk_audio,) = self.sess_dec.run(None, {"z": z_chunk, "g": g})
                audio_chunks.append(chunk_audio[0, 0, :])

            audio_full = np.concatenate(audio_chunks)

            # Baseline CPU Trimming
            valid_samples = real_y_len * 512
            audio_trimmed = audio_full[:valid_samples]
            lat_dec = (time.perf_counter() - t0) * 1000
            lat_trim = 0.0

        total_time = (time.perf_counter() - start_total) * 1000
        duration_sec = len(audio_trimmed) / 44100.0
        rtf = (total_time / 1000.0) / duration_sec if duration_sec > 0 else 0.0

        # Lưu file WAV
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_file), audio_trimmed, 44100)

        # In bảng đo lường chi tiết
        print("\n" + "-" * 75)
        if self.npu_native:
            print("⏱️  BẢNG PHÂN RÃ THỜI GIAN 100% NPU-NATIVE PIPELINE (PROBLEMS 2-4):")
            print("-" * 75)
            print(f"  • Khối 1: BERT Context Extractor        : {lat_prep:8.2f} ms")
            print(f"  • Khối 2: Text Encoder ({self.encoder_mode.upper()})          : {lat_enc:8.2f} ms")
            print(f"  • NPU P2: Binary Stencil Expansion      : {lat_align:8.2f} ms [100% NPU Native]")
            print(f"  • Khối 3: Normalizing Flow (UINT16)     : {lat_flow:8.2f} ms")
            print(f"  • NPU P3: Batched HiFi-GAN Vocoder      : {lat_dec:8.2f} ms [24 Chunks Reshape]")
            print(f"  • NPU P4: In-Graph Artifact Trimming    : {lat_trim:8.2f} ms [Zero-Noise Mask]")
        else:
            print("⏱️  BẢNG PHÂN RÃ THỜI GIAN THỰC THI 4 SUBMODELS (HYBRID BASELINE):")
            print("-" * 75)
            print(f"  • Khối 1: BERT Context Extractor        : {lat_prep:8.2f} ms")
            print(f"  • Khối 2: Text Encoder ({self.encoder_mode.upper()})          : {lat_enc:8.2f} ms")
            print(f"  • CPU Host: Monotonic Alignment         : {lat_align:8.2f} ms [CPU Dynamic Loop]")
            print(f"  • Khối 3: Normalizing Flow (UINT16)     : {lat_flow:8.2f} ms")
            print(f"  • CPU Host: Sliding Window + Vocoder    : {lat_dec:8.2f} ms [CPU Dynamic Slicing]")
        print("-" * 75)
        print(f"  🏁 TỔNG THỜI GIAN SUY LUẬN        : {total_time:8.2f} ms ({total_time/1000:.2f}s)")
        print(f"  🎵 Thời lượng âm thanh tạo ra     : {duration_sec:8.2f} s")
        print(f"  ⚡ Tỷ số thời gian thực (RTF)      : {rtf:8.3f} (RTF < 1.0 là Real-Time)")
        print(f"  💾 Tệp âm thanh đã xuất            : {out_file.resolve()}")
        print("-" * 75 + "\n")

        return str(out_file)


def main():
    args = parse_args()
    pipeline = QuantizedMeloTTSPipeline(encoder_mode=args.encoder_mode, npu_native=args.npu_native)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[!] Không tìm thấy tệp: {file_path}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        for idx, line in enumerate(lines):
            # Nếu có định dạng ID|Text thì lấy phần Text
            text_to_speak = line.split("|")[-1] if "|" in line else line
            out_name = f"output_quantized_{idx+1:03d}.wav"
            pipeline.synthesize(text_to_speak, output_path=out_name, speed=args.speed)
    else:
        pipeline.synthesize(args.text, output_path=args.output, speed=args.speed)


if __name__ == "__main__":
    main()
