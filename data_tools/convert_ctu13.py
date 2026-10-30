"""Convert CTU-13 bidirectional NetFlow (.binetflow / CSV) → ChronoFlux schema.

Usage:
  python data_tools/convert_ctu13.py path/to/capture.binetflow
  python data_tools/convert_ctu13.py path/to/folder --max-rows 100000

Official dataset:
  https://www.stratosphereips.org/datasets-ctu13
Place files under data/raw/ctu13/
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dataset_adapters import convert_ctu13, detect_dataset
from core.schema import FEATURE_COLUMNS

OUT_DIR = ROOT / "data" / "converted"


def _read_ctu(path: Path) -> pd.DataFrame:
    # CTU files are often comma-separated with header StartTime,Dur,Proto,...
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep=",", low_memory=False, encoding="latin-1")


def convert_file(path: Path, max_rows: int | None, out_dir: Path) -> Path:
    print(f"Reading {path} …")
    df = _read_ctu(path)
    kind = detect_dataset(df)
    if kind not in {"ctu13", "unknown"}:
        print(f"[WARN] detect_dataset={kind}; attempting CTU-13 conversion anyway")
    converted = convert_ctu13(df, max_rows=max_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"ctu13_{path.stem}_chronoflux.csv"
    keep = FEATURE_COLUMNS + ["attack_cat", "label"]
    converted[keep].to_csv(dest, index=False)
    print(f"[OK] {len(converted)} rows → {dest}")
    print(converted["attack_cat"].value_counts().head(10).to_string())
    return dest


def main():
    parser = argparse.ArgumentParser(description="Convert CTU-13 NetFlow CSV to ChronoFlux")
    parser.add_argument("input", help="CTU .binetflow/.csv file or directory")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out)
    if src.is_dir():
        files = sorted(list(src.glob("*.binetflow")) + list(src.glob("*.csv")))
        if not files:
            raise SystemExit(f"No CTU files in {src}")
        for f in files:
            convert_file(f, args.max_rows, out_dir)
    else:
        convert_file(src, args.max_rows, out_dir)

    print("\nNext: upload converted or original CTU CSV in Offline Analysis (auto-detect),")
    print("or merge into data/training.csv for detector retraining.")


if __name__ == "__main__":
    main()
