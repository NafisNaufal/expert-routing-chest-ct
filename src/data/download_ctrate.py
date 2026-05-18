"""
download_ctrate.py
------------------
Fetches the small CT-RATE metadata: radiology report CSVs and volume metadata
CSVs (rescale slope/intercept, spacing). It does NOT bulk-download CT volumes —
those are streamed one at a time by prepare_ctrate.py, processed into key
slices, and discarded. This keeps disk usage low (the server has ~241 GB total;
the full CT-RATE volume set is ~21 TB).

Official source: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE
CT-RATE is gated — set HF_TOKEN (or pass --hf_token) after accepting the terms.

Usage:
    export HF_TOKEN=your_token_here
    python src/data/download_ctrate.py --output ~/icsdg_data/ct_rate
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfFileSystem

REPO_ID = "ibrahimhamamci/CT-RATE"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, default="~/icsdg_data/ct_rate")
    p.add_argument("--hf_token", type=str, default=None)
    return p.parse_args()


def stream_to(fs, hf_path, dest):
    """Stream a remote file straight to disk (no persistent HF cache copy)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with fs.open(hf_path, "rb") as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out)


def main():
    args = parse_args()
    output_path = Path(args.output).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: no HF token provided — CT-RATE is gated and the "
              "download will likely fail. Set HF_TOKEN or pass --hf_token.")

    fs = HfFileSystem(token=token)
    base = f"datasets/{REPO_ID}/dataset"

    # Radiology report CSVs.
    print("Downloading radiology report CSVs ...")
    report_files = fs.glob(f"{base}/radiology_text_reports/*.csv")
    for hf_path in report_files:
        dest = output_path / "radiology_text_reports" / Path(hf_path).name
        stream_to(fs, hf_path, dest)
        print(f"  {dest.name}")

    # Volume metadata CSVs (rescale slope/intercept, spacing).
    print("Downloading metadata CSVs ...")
    meta_files = fs.glob(f"{base}/metadata/*.csv")
    for hf_path in meta_files:
        dest = output_path / "metadata" / Path(hf_path).name
        stream_to(fs, hf_path, dest)
        print(f"  {dest.name}")

    print(f"\nDone. CSVs saved under {output_path}")
    print("Next: python src/data/prepare_ctrate.py "
          f"--ctrate_root {args.output} --output_root <processed_root>")


if __name__ == "__main__":
    main()
