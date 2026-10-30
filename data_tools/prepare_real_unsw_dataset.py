from pathlib import Path
import hashlib
import sys

import pandas as pd
import requests

# ============================================================
# ChronoFlux - Real UNSW-NB15 Dataset Downloader + Converter
# ============================================================
# This script DOES NOT generate synthetic/fake rows.
# It downloads the published UNSW-NB15 pre-split CSVs from a
# public mirror and converts them to ChronoFlux's current
# 36-column schema.
#
# Official dataset page:
# https://research.unsw.edu.au/projects/unsw-nb15-dataset
#
# Official row counts:
# Training: 175,341
# Testing : 82,332
# ============================================================

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": {
        "filename": "UNSW_NB15_training-set.csv",
        "urls": [
            "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
            "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_training-set.csv?download=true",
        ],
        "sha256": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
        "expected_rows": 175341,
    },
    "test": {
        "filename": "UNSW_NB15_testing-set.csv",
        "urls": [
            "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv",
            "https://huggingface.co/datasets/Mouwiya/UNSW-NB15-small/resolve/main/UNSW_NB15_testing-set.csv?download=true",
        ],
        "sha256": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
        "expected_rows": 82332,
    },
}

# Exact 34 model features currently used by ChronoFlux
FEATURES = [
    "proto", "service", "state",
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "dwin",
    "tcprtt", "synack", "ackdat", "smean", "dmean",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst",
]
TARGETS = ["attack_cat", "label"]
KEEP_COLUMNS = FEATURES + TARGETS


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_fallback(info, dest: Path):
    if dest.exists():
        current_hash = sha256sum(dest)
        if current_hash == info["sha256"]:
            print(f"[OK] Existing verified file: {dest}")
            return

        print(f"[WARN] Existing file hash mismatch, re-downloading: {dest}")
        dest.unlink()

    last_error = None

    for url in info["urls"]:
        try:
            print(f"\nDownloading {info['filename']} ...")
            print(f"Source: {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

            digest = sha256sum(dest)
            if digest != info["sha256"]:
                raise RuntimeError(
                    f"SHA256 mismatch.\nExpected: {info['sha256']}\nActual:   {digest}"
                )

            print(f"[OK] Download verified: {dest}")
            return

        except Exception as exc:
            last_error = exc
            print(f"[FAIL] {exc}")
            if dest.exists():
                dest.unlink()

    raise RuntimeError(f"All download sources failed: {last_error}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standard UNSW split uses lowercase names.
    # Make the script tolerant of a few common casing variants.
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        rename_map[c] = lc

    df = df.rename(columns=rename_map)

    # attack_cat can occasionally contain leading/trailing spaces
    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
        df.loc[df["label"].eq(0), "attack_cat"] = "Normal"

    return df


def prepare(source: Path, output: Path, expected_rows: int):
    print(f"\nReading: {source}")
    df = pd.read_csv(source, low_memory=False)
    df = normalize_columns(df)

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Keep only genuine source columns used by the current ChronoFlux model.
    out = df[KEEP_COLUMNS].copy()

    # Basic integrity checks
    if len(out) != expected_rows:
        print(
            f"[WARN] Expected {expected_rows:,} rows, got {len(out):,}. "
            "Check the source file."
        )

    # Preserve real rows exactly; no resampling, no fake timestamps,
    # no generated attack progression.
    out.to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print(f"Shape: {out.shape}")
    print("\nBinary label distribution:")
    print(out["label"].value_counts(dropna=False).sort_index())
    print("\nAttack category distribution:")
    print(out["attack_cat"].value_counts(dropna=False))

    if out["label"].nunique() < 2:
        raise RuntimeError(
            "This split contains only one binary class. "
            "Do not use it for evaluation."
        )

    return out


def main():
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)

    train_raw = download_dir / FILES["train"]["filename"]
    test_raw = download_dir / FILES["test"]["filename"]

    download_with_fallback(FILES["train"], train_raw)
    download_with_fallback(FILES["test"], test_raw)

    train_out = OUTPUT_DIR / "Dataset_ChronoFlux_Clean.csv"
    test_out = OUTPUT_DIR / "Testing_Dataset_ChronoFlux_Clean.csv"

    train = prepare(
        train_raw,
        train_out,
        FILES["train"]["expected_rows"],
    )
    test = prepare(
        test_raw,
        test_out,
        FILES["test"]["expected_rows"],
    )

    # Friendly flat copies for the simplified project
    Path("data").mkdir(exist_ok=True)
    train.to_csv("data/training.csv", index=False)
    test.to_csv("data/testing.csv", index=False)

    print("\n" + "=" * 70)
    print("CHRONOFLUX DATASET READY")
    print("=" * 70)
    print(f"Training rows: {len(train):,}")
    print(f"Testing rows : {len(test):,}")
    print("Columns      : 36")
    print("\nCreated:")
    print("  data/raw/Dataset_ChronoFlux_Clean.csv")
    print("  data/raw/Testing_Dataset_ChronoFlux_Clean.csv")
    print("  data/training.csv")
    print("  data/testing.csv")
    print("\nNO synthetic rows were generated.")
    print("NO pseudo timestamps were added.")
    print("NO train/test oversampling was performed.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
