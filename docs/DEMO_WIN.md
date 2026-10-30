# ChronoFlux — Jury Demo Cheatsheet (Win Mode)

## Start

1. Double-click `START_CHRONOFLUX.bat`
2. It frees port 8000, starts the API, and opens **Chrome** at http://127.0.0.1:8000
3. Prefer **Google Chrome** for demos (Cursor preview / other browsers can look different because of font/cache quirks)
4. Open **Simulation** → **Play**, then **Run demo** (no Ubuntu sensor required)

Hard refresh if the UI looks stale: `Ctrl+F5`

## 90-second story

1. **Simulation** — Play animated pipeline (capture → detect → LSTM → T5–T7 → evidence)  
2. **Overview** — hybrid detection + LSTM world-model forecast  
3. **Forecasting** — T1→T4 → T5/T6/T7, infiltration %, driving features, LR baseline comparison  
4. **Offline Analysis** — upload CSV (**ChronoFlux / CIC-IDS-2018 / CTU-13**) or PCAP (world-model rollout)  
   - Try `data/samples/cicids2018_sample.csv` or `data/samples/ctu13_sample.csv`
5. **Evidence Chain** — SHA-256 tamper-evident ledger  

Quick start: open **Simulation** → **Play** → **Run live demo**.

## One-line answers

**What is novel?**  
LSTM world model learns `P(S_t+1|history)` from dual flow/packet features, rolls out K steps before compromise, maps MITRE stages, and explains with attention attribution — not a static packet classifier.

**World model vs LR?**  
Same stratified holdout; world model uses temporal LSTM encodings. Current lab holdout: world-model macro F1 ≈ **0.53** vs LR ≈ **0.38** (+0.15). Exact numbers are in `models/temporal/world_model_meta.json`.

**Proof of forecasting?**  
Issued T5 predictions are verified against later observed stages with lead-time measurement on the Forecasting page.

## Pre-flight command

```powershell
.\venv\Scripts\python.exe tests\win_suite.py
```

All checks must print PASS.
