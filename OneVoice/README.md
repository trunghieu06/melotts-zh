# OneVoice — Offline Edge Speech-Translation Device

Streaming ASR → MT → TTS cascade for Vietnamese ↔ English/Chinese/Korean, designed for offline, on-device
operation on Qualcomm Snapdragon hardware (noisy factory/construction/logistics environments).
Built for the **OneVoice AI Challenge** (Saigon AI Hub × Qualcomm).

Every model choice below was **code-tested with real numbers** (WER/CER/RTF/model size measured directly,
not taken from vendor claims) — the full reasoning, comparison tables, and every rejected candidate's
numbers live in each module's `stepN.md`.

## Architecture at a glance

| Step | Module | Pick | Doc | Code |
|---|---|---|---|---|
| 0 | Audio Front-end (VAD + denoise + beamform) | Silero VAD + GTCRN + MVDR/GSC | [step0.md](step0.md) | [src/step0_frontend/](src/step0_frontend/) |
| 1 | ASR (Speech Recognition) | Zipformer-30M (Vi) + SenseVoice-Small (En/Zh/Ko) | [step1.md](step1.md) | [src/step1_asr/](src/step1_asr/) |
| 2 | MT (Machine Translation) | NLLB-200-distilled-600M | [step2.md](step2.md) | [src/step2_mt/](src/step2_mt/) |
| 3 | TTS (Speech Synthesis) | Piper (Vi) + Supertonic (Ko/En) + MeloTTS-ZH (Zh) | [step3.md](step3.md) | [src/step3_tts/](src/step3_tts/) |
| 4 | Hardware & Quantization | Rubik Pi 3 (QCS6490), fallback Snapdragon 8 Elite phone | [step4.md](step4.md) | [src/step1_asr/verify_sensevoice_int8.py](src/step1_asr/verify_sensevoice_int8.py), [src/step2_mt/verify_nllb_int8.py](src/step2_mt/verify_nllb_int8.py), [src/step3_tts/quantize_supertonic.py](src/step3_tts/quantize_supertonic.py) |
| 5 | End-to-end pipeline (ASR→MT→TTS chained) | Same picks as Step 1-4, CPU/GPU comparison | — (no doc yet, see script docstrings) | [src/step5_pipeline/](src/step5_pipeline/) |

**Known open gap (flagged in every stepN.md):** all RTF numbers above except MeloTTS-ZH are measured on a
dev-machine GPU/CPU, not real Snapdragon hardware. MeloTTS-ZH is the only module profiled on an actual
Snapdragon 8 Elite Gen 5 via Qualcomm AI Hub. Next step for every other module: `qai-hub` remote-profiling.

**Total model footprint (real, on-disk, measured + quantized in step4.md):** 3.86GB unquantized → **~1.38GB
achieved** after quantizing NLLB-600M (CTranslate2 int8, verified safe via BLEU) and Supertonic (int8 with
its vocoder kept fp32 — full int8 broke the model, root-caused via bisection, see step4.md §3a).
SenseVoice-Small int8 works but costs real zh/ko quality (step4.md §3b) — not yet a final call.

## Repo layout

```
step0.md ... step4.md     full per-module analysis: every candidate tested, why it was chosen/rejected, real numbers
src/
  common.py                shared utilities (device selection, WAV I/O, WER/CER normalization, RTF calc)
  step0_frontend/           Step 0 code + README
  step1_asr/                 Step 1 code + README
  step2_mt/                  Step 2 code + README
  step3_tts/                  Step 3 code + README
  step5_pipeline/             Step 5 code + README (chains Step 1-4 into one CPU/GPU-comparable pipeline)
data/                       test sentences/audio (gitignored: raw audio + large downloaded corpora)
outputs/                    generated results -- CSVs + WAVs (gitignored, regenerate by running the scripts)
third_party_gtcrn/          vendored GTCRN model code + checkpoint (MIT, used verbatim by Step 0)
notebooks/                  Kaggle notebook (GPU-bound experiments)
```

## Quick start

Each step is self-contained — see that step's own README for exact commands and any engine-specific setup
quirks (a few TTS engines in particular need special install steps, see [src/step3_tts/README.md](src/step3_tts/README.md)):

```bash
pip install -r requirements.txt

cd src/step0_frontend && python run_all.py       # audio front-end
cd src/step1_asr && python run_all_asr.py         # ASR
cd src/step2_mt && python test_mt_nllb.py          # MT
cd src/step3_tts && python test_tts_piper.py       # TTS (Vietnamese)
cd src/step5_pipeline && python pipeline_s2s.py --device cpu   # full ASR->MT->TTS pipeline
```

Every script writes its own `outputs/<name>_results.csv` at the repo root regardless of which step folder
you run it from, and reads shared test data from `data/` the same way — the folder split only reorganizes
where the *code* lives, not where data/results go.

Common module import: `from common import ...` — every script auto-adds `src/` to `sys.path` at the top, so
running from inside a `stepN_*/` folder works without any manual `PYTHONPATH` setup.
