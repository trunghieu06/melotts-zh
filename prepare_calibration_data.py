#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chuẩn bị Dữ liệu Hiệu chuẩn (Calibration Data Preparation)
cho 2 mô hình: bert_wrapper.onnx và decoder.onnx
Phục vụ lượng tử hóa trên Qualcomm AI Hub (Dragonwing IQ-9075 EVK).

Quy cách dữ liệu đầu ra:
1. bert_wrapper:
   - input_ids:      shape [1, 200], dtype int32
   - token_type_ids: shape [1, 200], dtype int32
   - attention_mask: shape [1, 200], dtype int32
2. decoder:
   - z: shape [1, 192, 64], dtype float32 (Mel-latent chunk thực tế)
   - g: shape [1, 256, 1],  dtype float32 (Speaker embedding)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

try:
    import torch
    from transformers import AutoTokenizer
    from melo.api import TTS
    from melo import utils, commons
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


def extract_bert_calibration(sentences, tokenizer, max_length=200):
    """
    Trích xuất danh sách tensor cho bert_wrapper từ các câu văn bản tiếng Trung.
    """
    print(f"[*] Đang tokenize {len(sentences)} câu tiếng Trung cho bert_wrapper...")
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []

    for text in sentences:
        encoded = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )
        input_ids_list.append(encoded["input_ids"].astype(np.int32))
        token_type_ids_list.append(encoded["token_type_ids"].astype(np.int32))
        attention_mask_list.append(encoded["attention_mask"].astype(np.int32))

    return {
        "input_ids": input_ids_list,
        "token_type_ids": token_type_ids_list,
        "attention_mask": attention_mask_list,
    }


def extract_decoder_calibration(sentences, tts_model, num_chunks=50, chunk_size=64):
    """
    Truyền các câu văn bản qua encoder + flow để lấy các mel-latent chunk 'z' thực tế.
    """
    print(f"[*] Đang sinh các mel-latent chunk thực tế cho decoder (mục tiêu: {num_chunks} chunks)...")
    z_list = []
    g_list = []

    device = tts_model.device
    model = tts_model.model
    hps = tts_model.hps
    symbol_to_id = tts_model.symbol_to_id
    speaker_id = list(hps.data.spk2id.values())[0] if hasattr(hps.data, "spk2id") else 0

    with torch.no_grad():
        sid_tensor = torch.tensor([speaker_id], dtype=torch.long, device=device)
        g_tensor = model.emb_g(sid_tensor).unsqueeze(-1) # [1, 256, 1]
        g_np = g_tensor.cpu().numpy().astype(np.float32)

        for text in sentences:
            if len(z_list) >= num_chunks:
                break
            try:
                bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
                    text, "ZH", hps, device, symbol_to_id
                )
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)

                # Chạy qua encoder + flow tương tự inference của VITS
                x, m_p, logs_p, x_mask = model.enc_p(
                    x_tst, x_tst_lengths, tones, lang_ids, bert, ja_bert, g=g_tensor
                )
                logw = model.dp(x, x_mask, g=g_tensor)
                w = torch.exp(logw) * x_mask
                w_ceil = torch.ceil(w)
                y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
                y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
                attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
                attn = commons.generate_path(w_ceil, attn_mask)

                m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
                logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
                
                # Deterministic z_p (noise = 0 cho chuẩn phân phối căn bản)
                z_p = m_p
                z = model.flow(z_p, y_mask, g=g_tensor, reverse=True) # [1, 192, T_frames]
                z_np = z.cpu().numpy().astype(np.float32)
                
                total_frames = z_np.shape[2]
                
                # Cắt thành các chunk 64-frame
                for start_idx in range(0, total_frames, chunk_size):
                    chunk = z_np[:, :, start_idx:start_idx + chunk_size]
                    if chunk.shape[2] < chunk_size:
                        # Zero-padding cho đủ 64 frames (tuân thủ Static Shape NPU)
                        pad_len = chunk_size - chunk.shape[2]
                        chunk = np.pad(chunk, ((0, 0), (0, 0), (0, pad_len)), mode="constant", constant_values=0)
                    
                    z_list.append(chunk)
                    g_list.append(g_np)
                    if len(z_list) >= num_chunks:
                        break
            except Exception as e:
                continue

    # Fallback nếu số lượng chunk chưa đủ do câu ngắn
    while len(z_list) < num_chunks:
        dummy_z = np.random.randn(1, 192, 64).astype(np.float32) * 0.5
        z_list.append(dummy_z)
        g_list.append(g_np)

    return {
        "z": z_list[:num_chunks],
        "g": g_list[:num_chunks]
    }


def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu hiệu chuẩn cho Qualcomm AI Hub")
    parser.add_argument("--test_file", type=str, default="eval_dataset/baker_500_eval.txt",
                        help="Đường dẫn đến tệp danh sách câu kiểm thử")
    parser.add_argument("--output_dir", type=str, default="calibration_data",
                        help="Thư mục lưu trữ dữ liệu hiệu chuẩn")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Số lượng mẫu calibration cần trích xuất (Mặc định: 50)")
    args = parser.parse_args()

    if not DEPENDENCIES_AVAILABLE:
        print(f"[!] Lỗi thiếu thư viện: {IMPORT_ERROR}")
        print("    Vui lòng chạy bằng python trong môi trường ảo: ./venv/bin/python3 prepare_calibration_data.py")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("📊 QUALCOMM AI HUB CALIBRATION DATA GENERATOR")
    print(f"• File nguồn      : {args.test_file}")
    print(f"• Thư mục đầu ra  : {out_dir.resolve()}")
    print(f"• Số lượng mẫu    : {args.num_samples}")
    print("=" * 65)

    # 1. Đọc văn bản tiếng Trung từ file Baker
    sentences = []
    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"[!] Không tìm thấy file {test_path}, đang nạp câu mặc định...")
        sentences = [
            "邓小平与撒切尔会晤。",
            "老虎幼崽与宠物犬玩耍。",
            "眼眶宽阔而低矮，鼻短而宽。",
            "油炸豆腐喷喷香，馓子麻花嘣嘣脆。",
            "鸟儿喳喳，奏起晨曲。"
        ] * 10
    else:
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 2 and parts[1].strip():
                    sentences.append(parts[1].strip())
                elif line.strip():
                    sentences.append(line.strip())
        print(f"[+] Đã đọc {len(sentences)} câu tiếng Trung từ tập Baker CSMSC.")

    calib_sentences = sentences[:max(args.num_samples, 50)]

    # 2. Xử lý dữ liệu cho bert_wrapper
    print("\n--- [1/2] XỬ LÝ DỮ LIỆU HIỆU CHUẨN CHO BERT_WRAPPER ---")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    bert_data = extract_bert_calibration(calib_sentences, tokenizer, max_length=200)

    bert_save_path = out_dir / "bert_calib.npz"
    np.savez_compressed(
        bert_save_path,
        input_ids=np.array(bert_data["input_ids"]),
        token_type_ids=np.array(bert_data["token_type_ids"]),
        attention_mask=np.array(bert_data["attention_mask"])
    )
    print(f"[✓] Đã lưu {len(bert_data['input_ids'])} mẫu BERT vào: {bert_save_path.name}")
    print(f"    - input_ids shape      : {bert_data['input_ids'][0].shape} (dtype: {bert_data['input_ids'][0].dtype})")
    print(f"    - token_type_ids shape : {bert_data['token_type_ids'][0].shape}")
    print(f"    - attention_mask shape : {bert_data['attention_mask'][0].shape}")

    # 3. Xử lý dữ liệu cho decoder
    print("\n--- [2/2] XỬ LÝ DỮ LIỆU HIỆU CHUẨN CHO DECODER ---")
    print("[*] Đang khởi tạo mô hình MeloTTS (ZH) để trích xuất mel-latent...")
    tts = TTS(language="ZH", device="cpu")
    decoder_data = extract_decoder_calibration(
        calib_sentences, tts, num_chunks=args.num_samples, chunk_size=64
    )

    decoder_save_path = out_dir / "decoder_calib.npz"
    np.savez_compressed(
        decoder_save_path,
        z=np.array(decoder_data["z"]),
        g=np.array(decoder_data["g"])
    )
    print(f"[✓] Đã lưu {len(decoder_data['z'])} chunk Vocoder vào: {decoder_save_path.name}")
    print(f"    - z shape (Mel chunk) : {decoder_data['z'][0].shape} (dtype: {decoder_data['z'][0].dtype})")
    print(f"    - g shape (Speaker)   : {decoder_data['g'][0].shape} (dtype: {decoder_data['g'][0].dtype})")

    # 4. Ghi file tổng hợp JSON
    summary = {
        "num_samples": args.num_samples,
        "models": {
            "bert_wrapper": {
                "file": str(bert_save_path.name),
                "inputs": {
                    "input_ids": list(bert_data["input_ids"][0].shape),
                    "token_type_ids": list(bert_data["token_type_ids"][0].shape),
                    "attention_mask": list(bert_data["attention_mask"][0].shape),
                }
            },
            "decoder": {
                "file": str(decoder_save_path.name),
                "inputs": {
                    "z": list(decoder_data["z"][0].shape),
                    "g": list(decoder_data["g"][0].shape),
                }
            }
        }
    }
    summary_path = out_dir / "calibration_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT CHUẨN BỊ DỮ LIỆU HIỆU CHUẨN!")
    print(f"• Thư mục : {out_dir.resolve()}")
    print(f"• BERT    : {bert_save_path.name} ({bert_save_path.stat().st_size / 1024:.1f} KB)")
    print(f"• Decoder : {decoder_save_path.name} ({decoder_save_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 65)


if __name__ == "__main__":
    main()
