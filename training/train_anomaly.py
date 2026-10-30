from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from core.schema import NUMERIC_FEATURES, TARGET_COLUMN
from training.common import load_clean

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "training.csv"
OUT = ROOT / "models" / "anomaly"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_clean(TRAIN)
    normal = df[df[TARGET_COLUMN] == 0].copy()
    X = normal[NUMERIC_FEATURES]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("classifier", IsolationForest(
            n_estimators=160, contamination=.03, random_state=42, n_jobs=-1
        )),
    ])
    model.fit(X)
    scores = model.decision_function(X)
    meta = {
        "normal_rows": int(len(X)),
        "normal_q01": float(np.quantile(scores, .01)),
        "normal_q05": float(np.quantile(scores, .05)),
        "normal_q50": float(np.quantile(scores, .50)),
        "normal_q95": float(np.quantile(scores, .95)),
        "normal_q99": float(np.quantile(scores, .99)),
        "note": "Isolation Forest trained only on Normal traffic for baseline anomaly scoring.",
    }
    joblib.dump(model, OUT / "anomaly_model.joblib", compress=3)
    (OUT / "anomaly_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()
