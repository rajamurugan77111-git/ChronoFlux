from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from core.schema import FEATURE_COLUMNS, TARGET_COLUMN
from training.common import load_clean, preprocessor

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "training.csv"
TEST = ROOT / "data" / "testing.csv"
OUT = ROOT / "models" / "current_stage"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = load_clean(TRAIN)
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET_COLUMN]

    test_note = "external testing split"
    if TEST.exists():
        test = load_clean(TEST)
        if test[TARGET_COLUMN].nunique() >= 2:
            X_test, y_test = test[FEATURE_COLUMNS], test[TARGET_COLUMN]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=.20, random_state=42, stratify=y_train
            )
            test_note = "stratified holdout because external test lacks both classes"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=.20, random_state=42, stratify=y_train
        )
        test_note = "stratified holdout"

    model = Pipeline([
        ("preprocess", preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=140, max_depth=24, min_samples_leaf=1,
            random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "train_rows": int(len(X_train)), "evaluation_rows": int(len(X_test)),
        "evaluation_note": test_note,
    }
    joblib.dump(model, OUT / "detection_model.joblib", compress=3)
    (OUT / "detection_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
