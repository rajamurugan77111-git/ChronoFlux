from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.schema import FEATURE_COLUMNS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, ATTACK_CATEGORY_COLUMN


def load_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = FEATURE_COLUMNS + [TARGET_COLUMN, ATTACK_CATEGORY_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    out = df[required].copy()
    for c in NUMERIC_FEATURES:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype("string").fillna("UNKNOWN")
    out[TARGET_COLUMN] = pd.to_numeric(out[TARGET_COLUMN], errors="coerce")
    out = out.dropna(subset=[TARGET_COLUMN])
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)
    out[ATTACK_CATEGORY_COLUMN] = out[ATTACK_CATEGORY_COLUMN].astype("string").fillna("Unknown").str.strip().replace({"Backdoors":"Backdoor", "normal":"Normal"})
    return out


def preprocessor(scale_numeric: bool = True):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
