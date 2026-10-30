"""Convert CSE-CIC-IDS2018 CICFlowMeter CSV → ChronoFlux schema.

Usage:
  python data_tools/convert_cic_ids2018.py path/to/Friday-WorkingHours.csv
  python data_tools/convert_cic_ids2018.py path/to/folder --max-rows 50000

Place official CSVs under data/raw/cicids2018/ (download from UNB / AWS Open Data).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.dataset_adapters import convert_cic_ids2018, detect_dataset
from core.schema import FEATURE_COLUMNS

OUT_DIR = ROOT / "data" / "converted"


def convert_file(path: Path, max_rows: int | None, out_dir: Path) -> Path:
    print(f"Reading {path} …")
    df = pd.read_csv(path, low_memory=False)
    kind = detect_dataset(df)
    if kind not in {"cicids2018", "unknown"}:
        print(f"[WARN] detect_dataset={kind}; attempting CIC conversion anyway")
    converted = convert_cic_ids2018(df, max_rows=max_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"cicids2018_{path.stem}_chronoflux.csv"
    keep = FEATURE_COLUMNS + ["attack_cat", "label"]
    converted[keep].to_csv(dest, index=False)
    print(f"[OK] {len(converted)} rows → {dest}")
    print(converted["attack_cat"].value_counts().head(10).to_string())
    return dest


def main():
    parser = argparse.ArgumentParser(description="Convert CIC-IDS-2018 CSV to ChronoFlux")
    parser.add_argument("input", help="CSV file or directory of CICFlowMeter CSVs")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap per file")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory")
    args = parser.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out)
    if src.is_dir():
        files = sorted(src.glob("*.csv"))
        if not files:
            raise SystemExit(f"No CSV files in {src}")
        for f in files:
            convert_file(f, args.max_rows, out_dir)
    else:
        convert_file(src, args.max_rows, out_dir)

    print("\nNext:")
    print("  1. Copy/merge converted CSVs into data/training.csv + data/testing.csv if desired")
    print("  2. Or upload the converted CSV in Offline Analysis")
    print("  3. Or upload the original CIC CSV — the API auto-detects and converts")


if __name__ == "__main__":
    main()
