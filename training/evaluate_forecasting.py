"""Rigorous evaluation of ChronoFlux attack forecasting  P(stage_{t+1} | T1..T4).

    python -m training.evaluate_forecasting

Two protocols, both split by WHOLE recording session (no overlapping windows
leak between train and test):

A. Unseen sessions    - every attack progression pattern appears in training,
                        but the test sessions themselves were never seen
                        (4 folds; each fold holds out one session per pattern)
B. Unseen progressions- leave-one-pattern-out: the test kill-chain pattern is
                        absent from training (8 folds). This is the hardest
                        setting (genuinely new attack campaigns).

Forecasters compared on identical folds:
  * LSTM world model      (ChronoFlux v7)
  * Logistic regression   (flattened T1-T4 baseline)
  * Kill-chain prior      (Markov P(next | current) + MITRE ordering)
  * Hybrid  (v8)          confidence-gated LSTM ⊗ kill-chain prior; gate chosen
                          by inner cross-validation on the TRAINING folds only

Writes models/temporal/forecast_evaluation_v8.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.killchain import KillChainPrior, blend, gated_blend
from core.world_model import LSTMWorldModel, build_transition_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "world_model_sessions.jsonl"
OUT = ROOT / "models" / "temporal" / "forecast_evaluation_v8.json"
ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8]
GATE_GRID = [(hi, lo, tau) for hi in (0.65, 0.8, 0.9, 1.0) for lo in (0.0, 0.2, 0.35, 0.5)
             for tau in (0.4, 0.5, 0.6, 0.7, 0.8) if lo <= hi]
EPOCHS = 30


def load_samples():
    rows = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    built = build_transition_dataset(rows)
    stages_by_session = defaultdict(list)
    for r in rows:
        stages_by_session[r["session_id"]].append((r.get("prediction") or {}).get("current_stage"))
    pattern_of = {sid: tuple(st) for sid, st in stages_by_session.items()}
    patterns = sorted(set(pattern_of.values()))
    pid = {sid: patterns.index(p) for sid, p in pattern_of.items()}
    for s in built["samples"]:
        s["pattern"] = pid[s["session_id"]]
    # index of each session within its pattern (for protocol A folds)
    within = {}
    for p in range(len(patterns)):
        sids = sorted(sid for sid, v in pid.items() if v == p)
        within.update({sid: i for i, sid in enumerate(sids)})
    for s in built["samples"]:
        s["within"] = within[s["session_id"]]
    return built["samples"], len(patterns)


def fit_models(train, classes, seed=7):
    lstm = LSTMWorldModel(seed=seed)
    lstm.fit(np.stack([s["seq"] for s in train]), np.stack([s["next_state"] for s in train]),
             [s["stage"] for s in train], epochs=EPOCHS)
    lr = Pipeline([("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    lr.fit(np.stack([s["flat"] for s in train]), [s["stage"] for s in train])
    prior = KillChainPrior().fit((s["current_stage"], s["stage"]) for s in train)
    return lstm, lr, prior


def lstm_proba(lstm, s, classes):
    probs = lstm.predict_next(s["seq"])["stage_probs"]
    p = np.array([probs.get(c, 0.0) for c in classes]) + 1e-6
    return p / p.sum()


def lr_proba(lr, s, classes):
    raw = lr.predict_proba(s["flat"].reshape(1, -1))[0]
    p = np.full(len(classes), 1e-6)
    for c, v in zip(lr.classes_, raw):
        p[classes.index(c)] += v
    return p / p.sum()


def choose_gate(train, classes):
    """Tune the gated hybrid on TRAINING data only, for robustness in both regimes:
    inner folds that hold out sessions AND inner folds that hold out whole attack
    progressions. The objective is the mean macro-F1 over the two regimes."""
    sids = sorted({s["session_id"] for s in train})
    pats = sorted({s["pattern"] for s in train})
    inner = {"sessions": [(lambda h: (lambda s: s["session_id"] in h))(set(sids[i::3])) for i in range(3)],
             "patterns": [(lambda k: (lambda s: s["pattern"] == k))(k) for k in pats]}
    cache = {}
    for regime, folds in inner.items():
        rows = []
        for hold in folds:
            tr = [s for s in train if not hold(s)]
            va = [s for s in train if hold(s)]
            if not tr or not va:
                continue
            lstm, _, prior = fit_models(tr, classes, seed=11)
            rows += [(s["stage"], lstm_proba(lstm, s, classes), prior.proba(s["current_stage"], classes))
                     for s in va]
        cache[regime] = rows

    def score(params):
        out = []
        for rows in cache.values():
            if rows:
                y = [r[0] for r in rows]
                pred = [classes[int(np.argmax(gated_blend(r[1], r[2], *params)))] for r in rows]
                out.append(f1_score(y, pred, average="macro", zero_division=0))
        return np.mean(out) if out else 0

    return max(GATE_GRID, key=score)


def metrics(y, probs, classes):
    pred = [classes[int(np.argmax(p))] for p in probs]
    top2 = [y_i in {classes[j] for j in np.argsort(p)[-2:]} for y_i, p in zip(y, probs)]
    return {"accuracy": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "top2_accuracy": round(float(np.mean(top2)), 4),
            "samples": len(y)}


def run_protocol(samples, folds, classes):
    pooled = defaultdict(lambda: {"y": [], "p": []})
    alphas = []
    for test_key in folds:
        test = [s for s in samples if test_key(s)]
        train = [s for s in samples if not test_key(s)]
        if not test or not train:
            continue
        lstm, lr, prior = fit_models(train, classes)
        gate = choose_gate(train, classes)
        alphas.append(list(gate))
        for s in test:
            pl = lstm_proba(lstm, s, classes)
            pp = prior.proba(s["current_stage"], classes)
            for name, p in (("lstm_world_model_v7", pl),
                            ("logistic_regression", lr_proba(lr, s, classes)),
                            ("kill_chain_prior", pp),
                            ("hybrid_v8", gated_blend(pl, pp, *gate))):
                pooled[name]["y"].append(s["stage"])
                pooled[name]["p"].append(p)
    return ({k: metrics(v["y"], v["p"], classes) for k, v in pooled.items()},
            {"gate_per_fold (alpha_confident, alpha_uncertain, tau)": alphas})


def main():
    samples, n_patterns = load_samples()
    classes = sorted({s["stage"] for s in samples})
    print(f"{len(samples)} transition samples, {len({s['session_id'] for s in samples})} sessions, "
          f"{n_patterns} attack progressions, {len(classes)} stages")

    n_within = max(s["within"] for s in samples) + 1
    proto_a = [(lambda k: (lambda s: s["within"] == k))(k) for k in range(n_within)]
    proto_b = [(lambda k: (lambda s: s["pattern"] == k))(k) for k in range(n_patterns)]

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "data": str(DATA.relative_to(ROOT)), "classes": classes, "protocols": {}}
    for name, folds, desc in (
        ("A_unseen_sessions", proto_a, "Known attack progressions, unseen sessions (grouped by session)"),
        ("B_unseen_progressions", proto_b, "Leave-one-attack-progression-out (new campaigns)"),
    ):
        res, extra = run_protocol(samples, folds, classes)
        report["protocols"][name] = {"description": desc, "folds": len(folds), "results": res, **extra}
        print(f"\n{name}: {desc}")
        for model, m in res.items():
            print(f"  {model:<22} acc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}  "
                  f"top2={m['top2_accuracy']:.3f}")
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {OUT.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main()
