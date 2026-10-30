# Sequence / world-model directory

ChronoFlux writes:

```text
world_model.joblib          # LSTM world model weights
world_model_meta.json       # metrics + LR comparison
logistic_baseline.joblib    # logistic regression baseline
baseline_meta.json
sequence_rf.joblib          # optional RF ablation
temporal_meta.json
```

Train:

```powershell
.\venv\Scripts\python.exe -m training.train_world_model
```

Or use **Train / refresh world model** in the Forecasting page.
