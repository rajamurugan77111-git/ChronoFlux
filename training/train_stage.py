from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from core.schema import FEATURE_COLUMNS, ATTACK_CATEGORY_COLUMN
from training.common import load_clean, preprocessor

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "training.csv"
OUT = ROOT / "models" / "current_stage"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_clean(TRAIN)
    df = df[df[ATTACK_CATEGORY_COLUMN].str.lower().ne("unknown")].copy()
    X = df[FEATURE_COLUMNS]
    y = df[ATTACK_CATEGORY_COLUMN]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
    model = Pipeline([
        ("preprocess", preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=170, max_depth=26, random_state=42,
            n_jobs=-1, class_weight="balanced_subsample"
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    classes = list(model.named_steps["classifier"].classes_)
    metrics = {
        "accuracy": float(accuracy_score(y_val, pred)),
        "macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
        "train_rows": int(len(X_train)), "validation_rows": int(len(X_val)), "classes": classes,
    }
    joblib.dump(model, OUT / "stage_model.joblib", compress=3)
    (OUT / "stage_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "stage_classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
