"""Step 2 test -- NLLB-200-distilled-600M (baseline MT candidate).
Loads the official pretrained checkpoint -- zero fine-tuning, matching how
step1.md tested ASR candidates zero-shot before deciding what to fine-tune.
Measures sentence BLEU + RTF on all 6 directions (Vi<->En, Vi<->Zh, Vi<->Ko)
using the FLORES-200 sample from fetch_mt_data.py.
Maps to Technical Proposal SS4.2/SS4.3 (MT row).
"""
import os
import sys
import csv
import json
import time

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/ (for common.py)
from common import get_device, bleu

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "mt", "manifest.json")
RESULTS_CSV = os.path.join(ROOT, "outputs", "mt_nllb_results.csv")

MODEL_ID = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {"vi": "vie_Latn", "en": "eng_Latn", "zh": "zho_Hans", "ko": "kor_Hang"}
DIRECTIONS = [("vi", "en"), ("en", "vi"), ("vi", "zh"), ("zh", "vi"), ("vi", "ko"), ("ko", "vi")]


def translate(model, tokenizer, text, src, tgt, device):
    tokenizer.src_lang = NLLB_CODE[src]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tgt_id = tokenizer.convert_tokens_to_ids(NLLB_CODE[tgt])
    out = model.generate(**inputs, forced_bos_token_id=tgt_id, max_new_tokens=200)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def main():
    device = get_device()
    print(f"[test_mt_nllb] device = {device}, model = {MODEL_ID}")

    if not os.path.exists(MANIFEST):
        print("[test_mt_nllb] No MT data found. Run fetch_mt_data.py first.")
        return

    with open(MANIFEST, "r", encoding="utf-8") as f:
        rows = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device).eval()

    results = []
    for src, tgt in DIRECTIONS:
        scores, rtfs = [], []
        for row in rows:
            src_text, ref = row[src], row[tgt]
            t0 = time.perf_counter()
            hyp = translate(model, tokenizer, src_text, src, tgt, device)
            elapsed = time.perf_counter() - t0
            score = bleu(hyp, ref, tgt)
            scores.append(score)
            rtfs.append(elapsed)
        avg_bleu = sum(scores) / len(scores)
        avg_time = sum(rtfs) / len(rtfs)
        results.append({"direction": f"{src}->{tgt}", "bleu": round(avg_bleu, 2),
                         "avg_sec_per_sentence": round(avg_time, 4), "n": len(rows)})
        print(f"[test_mt_nllb] {src}->{tgt}  BLEU={avg_bleu:.2f}  "
              f"avg_sec/sentence={avg_time:.4f}")

    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[test_mt_nllb] wrote {RESULTS_CSV}")


if __name__ == "__main__":
    main()
