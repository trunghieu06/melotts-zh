# Step 0 — Audio Front-end (VAD + Denoise + Beamforming)

Full analysis, candidate comparison, and reasoning: [`../../step0.md`](../../step0.md)

**Picks:** Silero VAD + GTCRN denoiser + MVDR/GSC beamforming (all near-zero-train, discriminative/mask-based — not generative/diffusion, which is language-sensitive per the URGENT 2025 Challenge finding).

## Setup

```bash
pip install -r ../../requirements.txt
```

GTCRN's official checkpoint + model code is vendored verbatim (MIT licensed) at [`../../third_party_gtcrn/`](../../third_party_gtcrn/) — no extra install step needed, `test_denoise.py` imports it directly.

## Run

```bash
# put clean speech under data/clean/<lang>/ and noise clips under data/noise/
# (falls back to a synthetic factory-noise generator if data/noise/ is empty)
python mix_noise.py       # -> data/mixed/<lang>/*.wav + manifest.json
python test_vad.py        # -> outputs/vad_results.csv
python test_denoise.py    # -> outputs/denoise_results.csv (PESQ/STOI/RTF)
python test_beamform.py   # -> outputs/beamform_results.csv

# or run the whole suite in order:
python run_all.py
```

Results (CSV numbers + listenable WAVs under `outputs/`) map to Technical Proposal §4.2/§4.4.
