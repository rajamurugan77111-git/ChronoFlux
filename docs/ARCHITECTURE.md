# ChronoFlux v7.1 Architecture — LSTM World Model

```text
AUTHORIZED LAB DEMO / PCAP / CSV
                    ↓
        Dual feature extraction
   flow-level + packet-level state S_t
                    ↓
             Detection Engine
       ┌────────────┼────────────┐
       ↓            ↓            ↓
 Supervised ML  Isolation IF   Signatures
       └────────────┼────────────┘
                    ↓
             Weighted Fusion
                    ↓
          Current Behaviour State
                    ↓
              T1 → T2 → T3 → T4
                    ↓
         LSTM World Model
      learns P(S_t+1 | history)
                    ↓
         K-step forward rollout
             ↙       ↓       ↘
           T5       T6       T7
                    ↓
  Infiltration probability timeline
  MITRE ATT&CK stage mapping
  Attention / SHAP-style drivers
                    ↓
   Baselines: Logistic Regression + Sequence RF
                    ↓
      SHA-256 tamper-evident evidence chain
```

## Research separation

- **UNSW-NB15:** benchmark current-state Normal/Attack + attack-category classification.
- **World model:** chronological lab progressions + live gateway sessions teach transition dynamics.
- **Behaviour interpretation:** category → ChronoFlux stage / MITRE-style analyst context.

## Why this is a world model

The LSTM does not only classify the current window. It predicts the next network-state vector and stage distribution, then feeds predicted states back for K-step simulation before compromise completes.

## Explainability

Each forecast returns top driving features from temporal attention and gated feature attribution (SHAP-style), covering flags, ports, IAT, TTL, window size and scan scores.

## Baselines

Logistic regression is trained on the same flattened T1–T4 features. Sequence Random Forest remains as a non-recurrent ablation. Macro F1 / accuracy deltas are reported in `models/temporal/world_model_meta.json`.

Typical lab holdout (regenerate with `python -m training.train_world_model`):

- World model accuracy / macro F1 > Logistic Regression on the same split
- Deployment weights are refit on all labelled transitions after metrics are locked

## Demo without a live gateway

Use **Run demo** in the web console (`POST /api/demo/seed`) to inject a curated attack-progression session and show an active LSTM forecast offline. CSV/PCAP offline analysis is also supported. Live Ubuntu/Scapy gateway sensing is deferred in this build.

## External datasets (NTRO-aligned)

- **CIC-IDS-2018** and **CTU-13** are first-class inputs via `core/dataset_adapters.py`.
- Offline Analysis auto-detects schema; converters live in `data_tools/`.
- Details: `data_tools/EXTERNAL_DATASETS.md`.
