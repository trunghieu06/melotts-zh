# Step 2 — MT (Machine Translation)

Full analysis, FLORES-200 BLEU numbers for all 3 candidates: [`../../step2.md`](../../step2.md)

**Pick:** NLLB-200-distilled-600M — wins 5/6 directions (Vi↔En/Zh/Ko) and is 3-15x faster than either Qwen3 size tested. Only loses vi→zh to Qwen3-1.7B.

## Setup

```bash
pip install -r ../../requirements.txt
```

## Run

```bash
python fetch_mt_data.py   # downloads FLORES-200 devtest directly from Meta's public tarball
                           # (the HF mirrors are either gated or use an unsupported loading script)
python test_mt_nllb.py    # NLLB-200-distilled-600M -- current pick
python test_mt_qwen3.py   # Qwen3-0.6B and Qwen3-1.7B -- both tested, rejected (see step2.md)
```

Results -> `outputs/mt_*_results.csv` (BLEU per direction: Vi<->En/Zh/Ko).
