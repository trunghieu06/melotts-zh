#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare calibration data for encoder.onnx (MeloTTS-ZH)
Extracts 30 genuine Chinese speech samples from Baker CSMSC dataset.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from melo.api import TTS
from melo import utils

def main():
    print("=" * 60)
    print("📦 PREPARING CALIBRATION DATA FOR ENCODER.ONNX")
    print("=" * 60)

    dataset_path = Path("eval_dataset/baker_500_eval.txt")
    if not dataset_path.exists():
        print(f"[!] Error: {dataset_path} does not exist.")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        lines = [line.strip().split("|") for line in f if "|" in line]

    num_samples = 30
    selected_lines = lines[:num_samples]
    print(f"[+] Selected {len(selected_lines)} sentences from Baker CSMSC.")

    print("[*] Initializing MeloTTS (ZH)...")
    tts = TTS(language="ZH", device="cpu")
    speaker_id = list(tts.hps.data.spk2id.values())[0]

    max_x_len = 512

    sid_list = []
    bert_list = []
    ja_bert_list = []
    x_list = []
    tone_list = []
    language_list = []
    x_lengths_list = []
    noise_scale_w_list = []
    sdp_ratio_list = []
    length_scale_list = []

    print("[*] Extracting phonemes, tones, and BERT representations...")
    for idx, (item_id, text) in enumerate(selected_lines):
        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            text, tts.language, tts.hps, "cpu", tts.symbol_to_id
        )
        phone_len = min(phones.size(0), max_x_len)

        # 1. Phonemes
        x_arr = np.zeros((1, max_x_len), dtype=np.int32)
        x_arr[0, :phone_len] = phones[:phone_len].numpy().astype(np.int32)

        # 2. Tones
        tone_arr = np.zeros((1, max_x_len), dtype=np.int32)
        tone_arr[0, :phone_len] = tones[:phone_len].numpy().astype(np.int32)

        # 3. Language IDs
        lang_arr = np.zeros((1, max_x_len), dtype=np.int32)
        lang_arr[0, :phone_len] = lang_ids[:phone_len].numpy().astype(np.int32)

        # 4. BERT (1, 1024, 512)
        bert_arr = np.zeros((1, 1024, max_x_len), dtype=np.float32)
        bert_arr[0, :, :phone_len] = bert[:, :phone_len].numpy().astype(np.float32)

        # 5. JA BERT (1, 768, 512)
        ja_bert_arr = np.zeros((1, 768, max_x_len), dtype=np.float32)
        ja_bert_arr[0, :, :phone_len] = ja_bert[:, :phone_len].numpy().astype(np.float32)

        # Scalar/Vector inputs
        sid_arr = np.array([speaker_id], dtype=np.int32)
        x_len_arr = np.array([phone_len], dtype=np.int32)
        noise_w_arr = np.array([0.8], dtype=np.float32)
        sdp_arr = np.array([0.2], dtype=np.float32)
        len_scale_arr = np.array([1.0], dtype=np.float32)

        sid_list.append(sid_arr)
        bert_list.append(bert_arr)
        ja_bert_list.append(ja_bert_arr)
        x_list.append(x_arr)
        tone_list.append(tone_arr)
        language_list.append(lang_arr)
        x_lengths_list.append(x_len_arr)
        noise_scale_w_list.append(noise_w_arr)
        sdp_ratio_list.append(sdp_arr)
        length_scale_list.append(len_scale_arr)

        if (idx + 1) % 10 == 0 or idx == len(selected_lines) - 1:
            print(f"  Processed {idx + 1}/{len(selected_lines)} samples...")

    out_file = Path("calibration_data/encoder_calib.npz")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(out_file),
        sid=np.array(sid_list),
        bert=np.array(bert_list),
        ja_bert=np.array(ja_bert_list),
        x=np.array(x_list),
        tone=np.array(tone_list),
        language=np.array(language_list),
        x_lengths=np.array(x_lengths_list),
        noise_scale_w=np.array(noise_scale_w_list),
        sdp_ratio=np.array(sdp_ratio_list),
        length_scale=np.array(length_scale_list)
    )

    print(f"\n[✓] Successfully saved {len(sid_list)} calibration samples to {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
