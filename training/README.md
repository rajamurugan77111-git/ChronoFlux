# ChronoFlux model training — v7.1

## Current-state detection

```powershell
.\venv\Scripts\python.exe data_tools\prepare_real_unsw_dataset.py
.\venv\Scripts\python.exe -m training.train_all
```

## LSTM world model + logistic regression baseline

```powershell
.\venv\Scripts\python.exe -m training.train_world_model
```

This trains:

1. LSTM world model learning P(S_t+1|history) with K-step rollout
2. Logistic regression baseline on the same flattened T1–T4 features
3. Writes metrics and F1 comparison under `models/temporal/`

Curated lab attack-progression sessions are generated automatically. Live gateway sessions are merged when present.

## Sequence RF ablation (optional)

```powershell
.\venv\Scripts\python.exe -m training.train_temporal
```
