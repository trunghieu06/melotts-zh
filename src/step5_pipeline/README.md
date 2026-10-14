# Step 5 — End-to-end Speech-to-Speech Pipeline

Chains Step 1-4's picks into one measurable ASR→MT→TTS pipeline, runnable on CPU or GPU for
comparison. No separate `step5.md` yet — findings below and in the script's own docstrings/comments.

**Pipeline:** `audio in → ASR (Zipformer-30M int8 vi / SenseVoice-Small en·zh·ko) → MT (NLLB-600M
CTranslate2 int8) → TTS (Piper vi / Supertonic en·ko / MeloTTS zh) → audio out`

## Setup

**Needs its own venv** — MeloTTS requires Python<=3.11 (no 3.13 wheel for `tokenizers`, needs Rust to
build from source), which conflicts with the rest of this repo's usual Python 3.13 base env. Run this
folder's scripts with a dedicated Python 3.11 venv (`venv_pipeline/` in this repo, gitignored — rebuild
with `python3.11 -m venv venv_pipeline` if missing, then `pip install -r ../../requirements.txt`).

Also needs everything from Step 1-4 already built:
- `outputs/nllb-ct2-int8/` — run `python src/step2_mt/verify_nllb_int8.py` once first if missing (or any script that builds it)
- `outputs/supertonic-deploy/` — the int8-mixed Supertonic dir (3 submodels int8, `vocoder.onnx` fp32 — see step4.md §3a for why vocoder must stay fp32)
- `third_party_zipformer/.../tokens.generated.txt` — auto-created by `src/step1_asr/test_asr_zipformer.py` on first run
- `src/step3_tts/vi_VN-vais1000-medium.onnx` — download via `python -m piper.download_voices vi_VN-vais1000-medium` from that folder

```bash
pip install -r ../../requirements.txt
```

## Run

```bash
../../venv_pipeline/Scripts/python.exe pipeline_s2s.py --device cpu
../../venv_pipeline/Scripts/python.exe pipeline_s2s.py --device cuda
```

Results → `outputs/pipeline_results_<device>.csv` (per-stage + end-to-end latency, input-ASR score,
round-trip TTS-quality score) and `outputs/pipeline/<device>/*.wav` (every synthesized clip).

## Known findings

- **Zipformer outputs ALL-CAPS Vietnamese** — feeding that verbatim into NLLB produces garbage
  translations. Fixed: lowercase ASR output before MT (NLLB re-cases the target language on its own).
- **Supertonic Korean has a real reliability issue**, not a steps-count issue: its flow-matching
  sampler is stochastic and produces a repeated-syllable artifact on ~50% of single-attempt runs,
  confirmed at **both** `total_steps=5` and `total_steps=8` (3/6 failures each, see
  `outputs/ko_reliability.log`) — so raising the step count does not fix it. Fixed instead with a
  quality-gated retry: synthesize, round-trip through the already-loaded SenseVoice ASR, retry (up to
  5x) if CER > 5%, keep the best attempt.
- **CPU vs GPU**: GPU is decisively faster end-to-end (NLLB + SenseVoice run on CUDA; Zipformer/Piper/
  Supertonic stay CPU-only regardless of `--device` — sherpa-onnx's PyPI wheel and the local ONNX
  Runtime build are both CPU-only on Windows here). Read the actual CSV for current numbers rather
  than trusting a stale chat summary — re-run if in doubt, this pipeline is cheap to re-run.
- MeloTTS's BERT-based prosody model adds a hidden ~1.35GB runtime download not counted in step4.md's
  199MB MeloTTS-ZH figure; disabled by default here (`disable_bert=True`) for both speed and to avoid
  the surprise download. Naturalness tradeoff not yet verified by ear — flag before a real demo.
