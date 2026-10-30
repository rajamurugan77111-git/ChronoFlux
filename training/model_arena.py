"""ChronoFlux Model Arena - automated model selection for the current-state detectors.

Trains several candidate algorithms on the *same* stratified split, ranks them,
deploys the winner for each task and writes a leaderboard with per-class F1 and
permutation feature importance (explainability).

Tasks
-----
* detection : binary Normal vs Attack            (ranked by F1 on the attack class)
* stage     : attack-category classification     (ranked by macro-F1, which rewards
              getting rare classes such as Worms / Backdoor right, not just the
              majority classes)

Candidates
----------
* random_forest      - the original ChronoFlux v7 model (kept as the reference)
* hist_gradient_boost - histogram gradient boosting with native categorical
                        support and sqrt-balanced class weights
* logistic_regression - linear baseline

Run:
    python -m training.model_arena            # train, rank, deploy winners
    python -m training.model_arena --dry-run  # leaderboard only, keep current models
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

from core.schema import (ATTACK_CATEGORY_COLUMN, CATEGORICAL_FEATURES, FEATURE_COLUMNS,
                         NUMERIC_FEATURES, TARGET_COLUMN)
from training.common import load_clean, preprocessor

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "training.csv"
OUT = ROOT / "models" / "current_stage"
SEED = 42


# --------------------------------------------------------------------------- candidates
def _tree_preprocessor() -> ColumnTransformer:
    """Ordinal-encode categoricals so HistGradientBoosting can treat them natively.
    Column order: numeric first, then categorical."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), CATEGORICAL_FEATURES),
    ])


def candidates(task: str) -> dict:
    n_num, n_cat = len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES)
    cat_idx = list(range(n_num, n_num + n_cat))
    rf_kwargs = ({"n_estimators": 140, "max_depth": 24} if task == "detection"
                 else {"n_estimators": 170, "max_depth": 26})
    return {
        "random_forest": {
            "label": "Random Forest (ChronoFlux v7 reference)",
            "pipeline": Pipeline([
                ("preprocess", preprocessor()),
                ("classifier", RandomForestClassifier(random_state=SEED, n_jobs=-1,
                                                      class_weight="balanced_subsample",
                                                      **rf_kwargs)),
            ]),
            "balanced_weights": False,
        },
        "hist_gradient_boost": {
            "label": "Histogram Gradient Boosting (native categoricals, balanced)",
            "pipeline": Pipeline([
                ("preprocess", _tree_preprocessor()),
                ("classifier", HistGradientBoostingClassifier(
                    max_iter=400, learning_rate=0.08, max_leaf_nodes=48,
                    l2_regularization=1.0, categorical_features=cat_idx,
                    early_stopping=True, validation_fraction=0.1, random_state=SEED)),
            ]),
            "balanced_weights": True,
        },
        "hist_gradient_boost_cv": {
            "label": "Histogram Gradient Boosting, hyper-parameters tuned by 3-fold CV on train",
            "pipeline": None,          # built by tune_hgb() on the training split only
            "balanced_weights": True,
            "tune": True,
        },
        "logistic_regression": {
            "label": "Logistic Regression (linear baseline)",
            "pipeline": Pipeline([
                ("preprocess", preprocessor()),
                ("classifier", LogisticRegression(max_iter=600, class_weight="balanced")),
            ]),
            "balanced_weights": False,
        },
    }


HGB_GRID = [
    {"max_iter": 400, "learning_rate": 0.08, "max_leaf_nodes": 48, "min_samples_leaf": 20},
    {"max_iter": 600, "learning_rate": 0.05, "max_leaf_nodes": 64, "min_samples_leaf": 10},
    {"max_iter": 800, "learning_rate": 0.04, "max_leaf_nodes": 96, "min_samples_leaf": 10},
    {"max_iter": 600, "learning_rate": 0.05, "max_leaf_nodes": 64, "min_samples_leaf": 30},
]


def _hgb(params):
    n_num, n_cat = len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES)
    return Pipeline([
        ("preprocess", _tree_preprocessor()),
        ("classifier", HistGradientBoostingClassifier(
            l2_regularization=1.0, categorical_features=list(range(n_num, n_num + n_cat)),
            early_stopping=True, validation_fraction=0.1, random_state=SEED, **params)),
    ])


def tune_hgb(task, X_train, y_train):
    """Pick HGB hyper-parameters by stratified 3-fold CV on the TRAINING split only,
    so the holdout stays untouched (no selection bias)."""
    from sklearn.model_selection import StratifiedKFold
    scoring = "f1" if task == "detection" else "macro"
    best, best_score = None, -1
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for params in HGB_GRID:
        scores = []
        for tr, va in cv.split(X_train, y_train):
            m = _hgb(params)
            yt = y_train.iloc[tr]
            m.fit(X_train.iloc[tr], yt,
                  classifier__sample_weight=np.sqrt(compute_sample_weight("balanced", yt)))
            pred = m.predict(X_train.iloc[va])
            scores.append(f1_score(y_train.iloc[va], pred, zero_division=0) if scoring == "f1"
                          else f1_score(y_train.iloc[va], pred, average="macro", zero_division=0))
        if np.mean(scores) > best_score:
            best, best_score = params, float(np.mean(scores))
    return best, round(best_score, 4)


# --------------------------------------------------------------------------- evaluation
def _size_mb(model) -> float:
    buf = io.BytesIO()
    joblib.dump(model, buf, compress=3)
    return round(buf.tell() / 1e6, 2)


def evaluate(task: str, y_true, y_pred) -> dict:
    if task == "detection":
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
    labels = sorted(set(y_true))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {str(c): round(float(v), 4) for c, v in zip(labels, per_class)},
    }


def feature_importance(model, X_val, y_val, task: str, n_rows: int = 2500) -> list:
    """Model-agnostic permutation importance on a validation subsample."""
    if len(X_val) > n_rows:
        X_val = X_val.sample(n_rows, random_state=SEED)
        y_val = y_val.loc[X_val.index]
    scoring = "f1" if task == "detection" else "f1_macro"
    result = permutation_importance(model, X_val, y_val, scoring=scoring, n_repeats=3,
                                    random_state=SEED, n_jobs=-1)
    ranked = sorted(zip(FEATURE_COLUMNS, result.importances_mean), key=lambda t: -t[1])
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in ranked[:12]]


def label_ambiguity(df) -> dict:
    """How separable are the classes at all? For each class, the share of rows whose
    exact feature vector also occurs with a DIFFERENT label, plus the macro-F1 of an
    oracle that memorises every vector (scored on its own training data). This is an
    upper bound that no flow-feature model can beat."""
    key = df[FEATURE_COLUMNS].astype(str).agg("|".join, axis=1)
    y = df[ATTACK_CATEGORY_COLUMN]
    n_labels = y.groupby(key).nunique()
    ambiguous = key.map(n_labels) > 1
    majority = y.groupby(key).agg(lambda s: s.value_counts().index[0])
    oracle = key.map(majority)
    classes = sorted(y.unique())
    return {
        "share_of_rows_with_conflicting_duplicates": {
            str(c): round(float(ambiguous[y == c].mean()), 3) for c in classes},
        "oracle_macro_f1_upper_bound": round(float(f1_score(y, oracle, average="macro")), 4),
        "oracle_per_class_f1": {str(c): round(float(v), 3) for c, v in
                                zip(classes, f1_score(y, oracle, labels=classes, average=None))},
        "note": "Identical flow-feature vectors labelled with different attack categories "
                "(notably Analysis, Backdoor, DoS) cap achievable per-class accuracy.",
    }


def run_task(task: str, df, deploy: bool) -> dict:
    if task == "detection":
        X, y = df[FEATURE_COLUMNS], df[TARGET_COLUMN]
        primary = "f1"
    else:
        df = df[df[ATTACK_CATEGORY_COLUMN].str.lower().ne("unknown")]
        X, y = df[FEATURE_COLUMNS], df[ATTACK_CATEGORY_COLUMN]
        primary = "macro_f1"
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=.20, random_state=SEED, stratify=y)

    board, fitted = [], {}
    for key, spec in candidates(task).items():
        t0 = time.time()
        fit_kwargs = {}
        tuned = None
        if spec.get("tune"):
            params, cv_score = tune_hgb(task, X_train, y_train)
            spec["pipeline"] = _hgb(params)
            tuned = {"params": params, "cv_score": cv_score}
        if spec["balanced_weights"]:
            # sqrt of balanced weights: helps rare classes without wrecking precision
            fit_kwargs["classifier__sample_weight"] = np.sqrt(
                compute_sample_weight("balanced", y_train))
        model = spec["pipeline"]
        model.fit(X_train, y_train, **fit_kwargs)
        metrics = evaluate(task, y_val, model.predict(X_val))
        entry = {"key": key, "model": spec["label"], **metrics,
                 "train_seconds": round(time.time() - t0, 1), "size_mb": _size_mb(model)}
        if tuned:
            entry["tuning"] = tuned
        board.append(entry)
        fitted[key] = model
        print(f"  [{task}] {key:<22} {primary}={metrics[primary]:.4f} "
              f"acc={metrics['accuracy']:.4f} size={entry['size_mb']}MB")

    board.sort(key=lambda e: -e[primary])
    winner = board[0]
    reference = next(e for e in board if e["key"] == "random_forest")
    winner_model = fitted[winner["key"]]
    importance = feature_importance(winner_model, X_val, y_val, task)

    result = {
        "task": task,
        "ranking_metric": primary,
        "winner": winner["key"],
        "winner_model": winner["model"],
        "improvement_vs_reference": round(winner[primary] - reference[primary], 4),
        "leaderboard": board,
        "feature_importance": importance,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "deployed": deploy,
    }
    if task == "stage":
        result["label_ambiguity"] = label_ambiguity(df)

    if deploy:
        name = "detection" if task == "detection" else "stage"
        joblib.dump(winner_model, OUT / f"{name}_model.joblib", compress=3)
        legacy = {k: winner[k] for k in winner if k not in
                  ("key", "model", "train_seconds", "size_mb", "per_class_f1")}
        legacy.update(model_type=winner["model"], train_rows=int(len(X_train)))
        if task == "detection":
            legacy.update(evaluation_rows=int(len(X_val)),
                          evaluation_note="stratified holdout (model arena)")
        else:
            classes = [str(c) for c in winner_model.named_steps["classifier"].classes_]
            legacy.update(validation_rows=int(len(X_val)), classes=classes,
                          per_class_f1=winner["per_class_f1"])
            (OUT / "stage_classes.json").write_text(json.dumps(classes, indent=2),
                                                    encoding="utf-8")
        (OUT / f"{name}_metrics.json").write_text(json.dumps(legacy, indent=2),
                                                  encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="ChronoFlux model arena")
    parser.add_argument("--dry-run", action="store_true",
                        help="rank models but do not replace the deployed ones")
    parser.add_argument("--tasks", nargs="+", default=["detection", "stage"],
                        choices=["detection", "stage"])
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    df = load_clean(TRAIN)
    print(f"Model arena on {len(df):,} rows from {TRAIN.name}")
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "dataset": TRAIN.name, "split": "stratified 80/20 holdout, seed 42",
              "tasks": {}}
    for task in args.tasks:
        report["tasks"][task] = run_task(task, df, deploy=not args.dry_run)
        r = report["tasks"][task]
        print(f"  -> {task}: winner {r['winner']} "
              f"({r['ranking_metric']} {r['improvement_vs_reference']:+.4f} vs v7 reference)")

    arena_path = OUT / "model_arena.json"
    previous = {}
    if arena_path.exists():
        try:
            previous = json.loads(arena_path.read_text(encoding="utf-8")).get("tasks", {})
        except Exception:
            previous = {}
    report["tasks"] = {**previous, **report["tasks"]}
    arena_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Leaderboard written to {arena_path.relative_to(ROOT)}")
    return report


if __name__ == "__main__":
    main()
