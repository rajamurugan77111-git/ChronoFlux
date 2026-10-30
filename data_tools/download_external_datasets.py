"""Download ChronoFlux-usable open datasets (no AWS CLI required).

CIC-IDS-2018: processed CICFlowMeter CSVs from public AWS S3
  (NOT the multi-hundred-GB raw PCAP corpus)

CTU-13: official bidirectional labeled archive (~1.9 GB)

Usage:
  python data_tools/download_external_datasets.py
  python data_tools/download_external_datasets.py --cic-only
  python data_tools/download_external_datasets.py --ctu-only
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CIC_DIR = ROOT / "data" / "raw" / "cicids2018"
CTU_DIR = ROOT / "data" / "raw" / "ctu13"

CTU_URLS = [
    "https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset.tar.bz2",
    "https://s3.amazonaws.com/mcfp/datasets/CTU-13-Dataset.tar.bz2",
]

BUCKET = "cse-cic-ids2018"
PREFIX = "Processed Traffic Data for ML Algorithms/"


def download_cic():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    CIC_DIR.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    print(f"[INFO] Listing s3://{BUCKET}/{PREFIX}")
    token = None
    keys = []
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": PREFIX, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj["Key"]
            if key.lower().endswith(".csv"):
                keys.append((key, obj["Size"]))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    if not keys:
        raise SystemExit("No CIC CSV objects found in public S3 prefix")

    print(f"[INFO] Found {len(keys)} CSV file(s)")
    for key, size in keys:
        name = Path(key).name
        dest = CIC_DIR / name
        if dest.exists() and dest.stat().st_size == size:
            print(f"[SKIP] {name} already complete ({size/1e6:.1f} MB)")
            continue
        print(f"[GET ] {name} ({size/1e6:.1f} MB)")
        s3.download_file(BUCKET, key, str(dest))
        print(f"[OK  ] {name}")

    files = list(CIC_DIR.glob("*.csv"))
    print(f"[DONE] CIC-IDS-2018 CSVs → {CIC_DIR} ({len(files)} files, {sum(f.stat().st_size for f in files)/1e9:.2f} GB)")


def _download_url(url: str, dest: Path) -> bool:
    print(f"[INFO] Downloading {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ChronoFlux/7.1"})
        with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as out:
            total = resp.headers.get("Content-Length")
            total_i = int(total) if total else None
            done = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if total_i:
                    print(f"\r  {done/1e6:.1f}/{total_i/1e6:.1f} MB ({100*done/total_i:.1f}%)", end="", flush=True)
                else:
                    print(f"\r  {done/1e6:.1f} MB", end="", flush=True)
        print()
        return dest.exists() and dest.stat().st_size > 1_000_000
    except Exception as exc:
        print(f"\n[WARN] Failed {url}: {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def download_ctu():
    CTU_DIR.mkdir(parents=True, exist_ok=True)
    archive = CTU_DIR / "CTU-13-Dataset.tar.bz2"
    if not (archive.exists() and archive.stat().st_size > 100_000_000):
        ok = False
        for url in CTU_URLS:
            if _download_url(url, archive):
                ok = True
                break
        if not ok:
            raise SystemExit("CTU-13 download failed from all mirrors")
    else:
        print(f"[OK] Existing archive: {archive} ({archive.stat().st_size/1e9:.2f} GB)")

    print(f"[INFO] Extracting {archive} …")
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(CTU_DIR)
    flows = list(CTU_DIR.rglob("*.binetflow"))
    print(f"[DONE] CTU-13 extracted → {CTU_DIR}")
    print(f"  .binetflow files: {len(flows)}")
    for f in flows[:12]:
        print(f"  - {f.relative_to(CTU_DIR)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cic-only", action="store_true")
    parser.add_argument("--ctu-only", action="store_true")
    args = parser.parse_args()
    do_cic = not args.ctu_only
    do_ctu = not args.cic_only

    print("NOTE: Full CIC-IDS-2018 raw PCAPs are hundreds of GB and are NOT downloaded.")
    print("      This pulls the official processed CSV flows ChronoFlux uses.\n")

    if do_cic:
        download_cic()
    if do_ctu:
        download_ctu()

    print("\nNext:")
    print("  python data_tools/convert_cic_ids2018.py data/raw/cicids2018 --max-rows 200000")
    print("  python data_tools/convert_ctu13.py data/raw/ctu13 --max-rows 200000")


if __name__ == "__main__":
    main()
