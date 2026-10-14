# Step 1 — ASR (Speech Recognition)

Full analysis, all 5 candidates tested with real numbers, license caveats: [`../../step1.md`](../../step1.md)

**Picks:** Zipformer-30M-RNNT-6000h (Vietnamese) + SenseVoice-Small (English/Chinese/Korean). PhoWhisper, Moonshine, and Qwen3-ASR-0.6B were all tested and rejected — see step1.md §12-15 for why.

## Setup

```bash
pip install -r ../../requirements.txt
```

`test_asr_qwen.py` needs `transformers>=5.13.0`, which can conflict with funasr's own pin — install that one in an isolated venv/conda env if a plain `pip install -U transformers` breaks `test_asr_vi.py` / `test_asr_multi.py`.

## Run

```bash
python fetch_asr_data.py      # -> data/asr/<lang>/*.wav + manifest.json (FLEURS-based)
python mix_asr_noise.py       # -> data/asr_mixed/<lang>/*_snrN.wav (SNR-robustness test set)

python test_asr_vi.py         # PhoWhisper (Vi) -- rejected candidate, kept for comparison
python test_asr_multi.py      # SenseVoice-Small (En/Zh/Ko) -- current pick
python test_asr_zipformer.py  # Zipformer-30M (Vi) -- current pick
python test_asr_moonshine.py  # Moonshine (Vi/En/Zh/Ko) -- rejected candidate
python test_asr_qwen.py       # Qwen3-ASR-0.6B -- rejected (10x slower than SenseVoice)

# or run the current-pick pair + all alt candidates in order:
python run_all_asr.py
```

Results -> `outputs/asr_*_results.csv` (WER/CER/RTF per language, per SNR level).
