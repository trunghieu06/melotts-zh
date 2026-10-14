# Step 3 — TTS (Speech Synthesis)

Full analysis, all 5 candidates tested with real RTF + round-trip WER/CER: [`../../step3.md`](../../step3.md)

**Picks:** Piper or VieNeu-TTS (Vietnamese — see step3.md §2.3 for the tradeoff) + Supertonic (Korean+English) + MeloTTS-ZH (Chinese). Supertonic's Vietnamese output and Confucius4-TTS were both tested and rejected — see step3.md for why.

## Setup

Each engine has its own install quirk — see `../../requirements.txt` for the full list. The two that need special handling:

```bash
# Piper: pip install piper-tts, then download each voice once before first run
python -m piper.download_voices vi_VN-vais1000-medium
python -m piper.download_voices en_US-amy-medium

# VieNeu-TTS: mode="standard" needs neucodec + a torchao version that matches your torch build
pip install vieneu neucodec "torchao==0.9.0"   # torchao>=0.10 needs torch>=2.7 (register_constant API)
# on Windows, llama-cpp-python's default pip wheel often fails to load its native DLL -- if so:
pip install llama-cpp-python --prefer-binary --index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# MeloTTS is not on PyPI:
git clone https://github.com/myshell-ai/MeloTTS && cd MeloTTS && pip install -e .
# needs Python<=3.11 (tokenizers has no 3.13 wheel and needs Rust to build from source)
```

## Run

```bash
python test_tts_supertonic.py    # Vi+Ko+En -- current pick for Ko+En only (Vi has repetition artifacts)
python test_tts_melotts.py       # Zh+En -- current pick for Zh
python test_tts_piper.py         # Vi+En -- current pick for Vi (recommended: fastest + smallest)
python test_tts_vieneu.py        # Vi -- alternative pick for Vi (slightly better WER, 3x slower, 8x larger)
python test_tts_confucius.py     # rejected -- >2.4GB for the speaker-encoder alone, testing abandoned

# round-trip quality eval: re-transcribes every outputs/tts_*_results.csv WAV through
# Step 1's Zipformer (vi) / SenseVoice (en/zh/ko) and scores WER/CER against the source text
python test_tts_eval_quality.py
```

Results -> `outputs/tts_*_results.csv` (RTF) and `outputs/tts_quality_results.csv` (round-trip WER/CER).
