#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Benchmark: Stitched Submodels vs Original MeloTTS FP32 Baseline
Using eval_dataset/baker_500_eval.txt
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import onnxruntime as ort

from melo.api import TTS
from melo import utils, commons

def run_evaluation(num_samples=10):
    print("=" * 80)
    print(f"🚀 BENCHMARK: GHÉP NỐI 4 SUBMODEL vs MÔ HÌNH FP32 GỐC ({num_samples} CÂU BAKER CSMSC)")
    print("=" * 80)

    dataset_path = Path("eval_dataset/baker_500_eval.txt")
    if not dataset_path.exists():
        print(f"[!] Error: {dataset_path} not found.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split("|") for line in f if "|" in line]

    test_cases = lines[:num_samples]

    print("[*] Loading MeloTTS FP32 reference model...")
    tts = TTS(language="ZH", device="cpu")
    speaker_id = list(tts.hps.data.spk2id.values())[0]

    print("[*] Loading ONNX / Quantized submodels...")
    sess_enc_fp32 = ort.InferenceSession("onnx_models/encoder.onnx", providers=["CPUExecutionProvider"])
    sess_enc_quant = ort.InferenceSession("quantized_models/encoder_w8a16/job_j57e4vrqp_qdq_onnx/model.onnx", providers=["CPUExecutionProvider"])
    sess_flow = ort.InferenceSession("onnx_models/flow.onnx", providers=["CPUExecutionProvider"])
    sess_dec_quant = ort.InferenceSession("quantized_models/decoder_w8a16/job_jgolrvj1g_qdq_onnx/model.onnx", providers=["CPUExecutionProvider"])

    out_dir = Path("output_eval/benchmark_stitched")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_x_len = 512
    max_y_len = 1536
    chunk_size = 64

    results_config_a = [] # Config A: Encoder FP32 + Flow + Decoder W8A16 (Standard Qualcomm setup)
    results_config_b = [] # Config B: Encoder W8A16 + Flow + Decoder W8A16 (All-Quantized setup)

    print("\n" + "=" * 95)
    print(f"{'ID':<6} | {'Text':<24} | {'Config A Cos Sim':<16} | {'Config A MCD':<12} | {'Config B Cos Sim':<16} | {'Config B MCD':<12}")
    print("=" * 95)

    for item_id, text in test_cases:
        # 1. FP32 Baseline
        texts = tts.split_sentences_into_pieces(text, tts.language, quiet=True)
        sub_text = texts[0]

        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            sub_text, tts.language, tts.hps, "cpu", tts.symbol_to_id
        )
        phone_len = phones.size(0)

        with torch.no_grad():
            x_tst = phones.unsqueeze(0)
            tones_tst = tones.unsqueeze(0)
            lang_ids_tst = lang_ids.unsqueeze(0)
            bert_tst = bert.unsqueeze(0)
            ja_bert_tst = ja_bert.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phones.size(0)])
            speakers = torch.LongTensor([speaker_id])

            t0 = time.perf_counter()
            audio_pt = tts.model.infer(
                x_tst, x_tst_lengths, speakers, tones_tst, lang_ids_tst, bert_tst, ja_bert_tst,
                sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.0, length_scale=1.0
            )[0][0, 0].cpu().float().numpy()
            lat_fp32 = (time.perf_counter() - t0) * 1000

        # Input tensors for ONNX Encoders
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

        sid = np.array([speaker_id], dtype=np.int32)
        x_lengths = np.array([phone_len], dtype=np.int32)
        noise_scale_w = np.array([0.0], dtype=np.float32)
        sdp_ratio = np.array([0.0], dtype=np.float32)
        length_scale = np.array([1.0], dtype=np.float32)

        enc_inputs = {
            "sid": sid, "bert": bert_pad, "ja_bert": ja_bert_pad,
            "x": x, "tone": tone, "language": language_tensor,
            "x_lengths": x_lengths, "noise_scale_w": noise_scale_w,
            "sdp_ratio": sdp_ratio, "length_scale": length_scale
        }

        # Helper pipeline runner
        def run_stitched_pipeline(sess_encoder):
            t_start = time.perf_counter()
            m_p, logs_p, w_ceil, y_lengths, x_mask, g = sess_encoder.run(None, enc_inputs)
            real_y_len = int(y_lengths[0])

            y_lengths_tensor = torch.from_numpy(y_lengths).long()
            x_mask_tensor = torch.from_numpy(x_mask)
            w_ceil_tensor = torch.from_numpy(w_ceil)

            y_mask_tensor = commons.sequence_mask(y_lengths_tensor, max_y_len).unsqueeze(1).to(x_mask_tensor.dtype)
            attn_mask = x_mask_tensor.unsqueeze(2) * y_mask_tensor.unsqueeze(-1)
            attn = commons.generate_path(w_ceil_tensor, attn_mask)
            attn_squeezed = attn.squeeze(1).numpy()

            noise_scale = np.array([0.0], dtype=np.float32)
            flow_inputs = {
                "attn_squeezed": attn_squeezed, "logs_p": logs_p,
                "noise_scale": noise_scale, "m_p": m_p,
                "y_mask": y_mask_tensor.numpy(), "g": g
            }
            z = sess_flow.run(None, flow_inputs)[0]

            z_valid = z[:, :, :real_y_len]
            num_chunks = int(np.ceil(real_y_len / chunk_size))
            audio_chunks = []

            for c in range(num_chunks):
                start = c * chunk_size
                end = min(start + chunk_size, real_y_len)
                chunk = np.zeros((1, 192, chunk_size), dtype=np.float32)
                chunk[:, :, :end - start] = z_valid[:, :, start:end]
                chunk_audio = sess_dec_quant.run(None, {"z": chunk, "g": g})[0]
                audio_chunks.append(chunk_audio.squeeze())

            full_audio = np.concatenate(audio_chunks)
            valid_samples = real_y_len * 512
            final_audio = full_audio[:valid_samples]
            lat_ms = (time.perf_counter() - t_start) * 1000
            return final_audio, lat_ms

        # Run Config A (Encoder FP32 + Decoder W8A16)
        audio_a, lat_a = run_stitched_pipeline(sess_enc_fp32)
        min_len_a = min(len(audio_pt), len(audio_a))
        mel_pt = librosa.feature.melspectrogram(y=audio_pt[:min_len_a], sr=24000).flatten()
        mel_a = librosa.feature.melspectrogram(y=audio_a[:min_len_a], sr=24000).flatten()
        cos_a = float(np.dot(mel_pt, mel_a) / (np.linalg.norm(mel_pt) * np.linalg.norm(mel_a) + 1e-12))

        # MCD Config A
        mfcc_pt = librosa.feature.mfcc(y=audio_pt, sr=24000, n_mfcc=16).T
        mfcc_a = librosa.feature.mfcc(y=audio_a, sr=24000, n_mfcc=16).T
        _, path_a = fastdtw(mfcc_pt, mfcc_a, dist=euclidean)
        cost_a = sum(np.sqrt(2.0 * np.sum((mfcc_pt[i] - mfcc_a[j]) ** 2)) for i, j in path_a)
        mcd_a = float((cost_a / len(path_a)) * (10.0 / np.log(10.0)))

        # Run Config B (Encoder W8A16 + Decoder W8A16)
        audio_b, lat_b = run_stitched_pipeline(sess_enc_quant)
        min_len_b = min(len(audio_pt), len(audio_b))
        mel_b = librosa.feature.melspectrogram(y=audio_b[:min_len_b], sr=24000).flatten()
        cos_b = float(np.dot(mel_pt[:len(mel_b)], mel_b) / (np.linalg.norm(mel_pt[:len(mel_b)]) * np.linalg.norm(mel_b) + 1e-12))

        # MCD Config B
        mfcc_b = librosa.feature.mfcc(y=audio_b, sr=24000, n_mfcc=16).T
        _, path_b = fastdtw(mfcc_pt, mfcc_b, dist=euclidean)
        cost_b = sum(np.sqrt(2.0 * np.sum((mfcc_pt[i] - mfcc_b[j]) ** 2)) for i, j in path_b)
        mcd_b = float((cost_b / len(path_b)) * (10.0 / np.log(10.0)))

        results_config_a.append({
            "id": item_id, "cos_sim": cos_a, "mcd": mcd_a, "lat_ms": lat_a
        })
        results_config_b.append({
            "id": item_id, "cos_sim": cos_b, "mcd": mcd_b, "lat_ms": lat_b
        })

        short_text = text[:15] + "..." if len(text) > 15 else text
        print(f"{item_id:<6} | {short_text:<24} | {cos_a*100:6.2f}% ({cos_a:.4f})  | {mcd_a:8.2f} dB   | {cos_b*100:6.2f}% ({cos_b:.4f})  | {mcd_b:8.2f} dB")

    # Summary
    avg_cos_a = np.mean([r["cos_sim"] for r in results_config_a])
    avg_mcd_a = np.mean([r["mcd"] for r in results_config_a])
    avg_lat_a = np.mean([r["lat_ms"] for r in results_config_a])

    avg_cos_b = np.mean([r["cos_sim"] for r in results_config_b])
    avg_mcd_b = np.mean([r["mcd"] for r in results_config_b])
    avg_lat_b = np.mean([r["lat_ms"] for r in results_config_b])

    print("=" * 95)
    print("\n" + "=" * 70)
    print("🏆 BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG & ĐỘ TRUNG THỰC (% MODEL GỐC):")
    print("=" * 70)
    print(f"CẤU HÌNH A (CHUẨN QUALCOMM METADATA: Encoder FP32 + Flow + Vocoder W8A16):")
    print(f"  • Độ tương đồng phổ Mel (Cosine Similarity) : {avg_cos_a * 100:.2f}% (Đạt {avg_cos_a:.5f})")
    print(f"  • Sai lệch âm học trung bình (MCD)         : {avg_mcd_a:.2f} dB")
    print(f"  • Thời gian suy luận trung bình             : {avg_lat_a:.1f} ms")
    print("-" * 70)
    print(f"CẤU HÌNH B (LƯỢNG TỬ HÓA TOÀN PHẦN: Encoder W8A16 + Flow + Vocoder W8A16):")
    print(f"  • Độ tương đồng phổ Mel (Cosine Similarity) : {avg_cos_b * 100:.2f}% (Đạt {avg_cos_b:.5f})")
    print(f"  • Sai lệch âm học trung bình (MCD)         : {avg_mcd_b:.2f} dB")
    print(f"  • Thời gian suy luận trung bình             : {avg_lat_b:.1f} ms")
    print("=" * 70)

    # Save report
    report_file = out_dir / "stitched_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "num_evaluated": len(test_cases),
            "config_a_standard_qualcomm": {
                "avg_cosine_similarity_pct": float(avg_cos_a * 100),
                "avg_mcd_db": float(avg_mcd_a),
                "avg_latency_ms": float(avg_lat_a)
            },
            "config_b_fully_quantized": {
                "avg_cosine_similarity_pct": float(avg_cos_b * 100),
                "avg_mcd_db": float(avg_mcd_b),
                "avg_latency_ms": float(avg_lat_b)
            },
            "detailed_results_config_a": results_config_a,
            "detailed_results_config_b": results_config_b
        }, f, indent=2, ensure_ascii=False)
    print(f"[✓] Đã lưu báo cáo chi tiết vào: {report_file}")

if __name__ == "__main__":
    run_evaluation(num_samples=10)
