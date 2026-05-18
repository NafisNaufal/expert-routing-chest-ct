"""
prepare_ctrate.py
-----------------
Streaming CT-RATE preprocessing. Volumes are downloaded ONE AT A TIME from
HuggingFace, converted into key axial slices, and the raw volume is deleted
immediately — so peak disk use stays tiny even though the full dataset is ~21 TB.

Produces two instruction sets:

  ctrate_train.json    detection-routing instructions (training)
      The gpt response emits the structured routing token
      <VISTA3D(lung tumor)> when the paired report describes pulmonary
      findings — this is what teaches VILA-M3 to delegate localisation.

  ctrate_holdout.json  retrieval records (evaluation only)
      Carries `query_text` (the report findings) + key slices; used by
      eval_retrieval.py. No retrieval instruction-tuning is performed.

Requires the report/metadata CSVs from download_ctrate.py.

Usage:
    export HF_TOKEN=your_token_here
    python src/data/prepare_ctrate.py \
        --ctrate_root ~/icsdg_data/ct_rate \
        --output_root ~/icsdg_data/processed \
        --max_volumes 2000
"""

import argparse
import json
import os
import random
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ID = "ibrahimhamamci/CT-RATE"

# Pulmonary findings that should trigger routing to the VISTA3D lung-tumor expert.
_PULMONARY_RE = re.compile(
    r"\b(nodul|mass|lesion|opacit|consolidat|tumou?r|ground.?glass|ggo|"
    r"spiculat|metasta|malignan)\w*", re.IGNORECASE)

_FINDING_RE = re.compile(
    r"(findings?|impression|conclusion)[:\s]*(.*?)(?=findings?|impression|"
    r"conclusion|$)", re.IGNORECASE | re.DOTALL)


# ── Report / metadata indices ─────────────────────────────────────────────────

def extract_findings(report_text: str) -> str:
    report_text = report_text.strip()
    matches = _FINDING_RE.findall(report_text)
    if matches:
        findings = max(matches, key=lambda m: len(m[1]))[1].strip()
        if len(findings) > 20:
            return findings
    return report_text


def _norm_id(name: str) -> str:
    return str(name).strip().replace(".nii.gz", "").replace(".nii", "")


def load_report_index(ctrate_root: Path) -> dict:
    import pandas as pd

    report_dir = ctrate_root / "radiology_text_reports"
    csvs = list(report_dir.glob("*.csv")) if report_dir.exists() else []
    if not csvs:
        raise FileNotFoundError(
            f"No report CSVs under {report_dir}. Run download_ctrate.py first.")

    combined = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    combined.columns = [c.strip().lower() for c in combined.columns]
    vol_col = next((c for c in combined.columns if "volume" in c or "file" in c), None)
    rep_col = next((c for c in combined.columns
                    if "finding" in c or "report" in c or "text" in c), None)
    if vol_col is None or rep_col is None:
        raise ValueError(f"Cannot find volume/report columns: {list(combined.columns)}")

    index = {}
    for _, row in combined.iterrows():
        text = str(row[rep_col]).strip()
        if text and text.lower() != "nan":
            index[_norm_id(row[vol_col])] = text
    print(f"Loaded {len(index)} reports")
    return index


def load_metadata_index(ctrate_root: Path) -> dict:
    """Return {volume_id: (rescale_slope, rescale_intercept)}."""
    import pandas as pd

    meta_dir = ctrate_root / "metadata"
    csvs = list(meta_dir.glob("*.csv")) if meta_dir.exists() else []
    if not csvs:
        print("WARNING: no metadata CSVs found — assuming volumes are already "
              "in Hounsfield Units (rescale slope=1, intercept=0).")
        return {}

    combined = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    combined.columns = [c.strip().lower() for c in combined.columns]
    vol_col = next((c for c in combined.columns if "volume" in c or "file" in c), None)
    slope_col = next((c for c in combined.columns if "slope" in c), None)
    inter_col = next((c for c in combined.columns if "intercept" in c), None)

    index = {}
    if vol_col and slope_col and inter_col:
        for _, row in combined.iterrows():
            try:
                index[_norm_id(row[vol_col])] = (
                    float(row[slope_col]), float(row[inter_col]))
            except (ValueError, TypeError):
                continue
    print(f"Loaded rescale metadata for {len(index)} volumes")
    return index


# ── Volume → slices ───────────────────────────────────────────────────────────

def window_ct(vol: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)


def sample_key_slices(vol: np.ndarray, n: int) -> list:
    z = vol.shape[0]
    margin = max(1, int(z * 0.10))
    indices = np.linspace(margin, z - margin - 1, n, dtype=int)
    return [vol[i] for i in indices]


def save_slices(slices, slices_dir: Path, volume_id: str, size: int) -> list:
    paths = []
    for i, sl in enumerate(slices):
        img = Image.fromarray(sl).convert("RGB").resize((size, size), Image.BILINEAR)
        img.save(slices_dir / f"{volume_id}_{i:02d}.png")
        paths.append(f"slices/{volume_id}_{i:02d}.png")
    return paths


def process_volume(nii_path: Path, slope: float, intercept: float,
                   max_slices: int, slices_dir: Path, volume_id: str,
                   slice_size: int) -> list:
    import nibabel as nib

    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)
    if vol.ndim == 3:
        vol = vol.transpose(2, 0, 1)            # (H,W,Z) -> (Z,H,W)
    vol = vol * slope + intercept               # CT-RATE rescale to Hounsfield Units
    vol = window_ct(vol)
    slices = sample_key_slices(vol, max_slices)
    return save_slices(slices, slices_dir, volume_id, slice_size)


# ── Instruction record builders ───────────────────────────────────────────────

def make_detection_record(volume_id, slice_paths, findings) -> dict:
    prompt = ("<image>\n" * len(slice_paths)
              + "Identify and localise thoracic abnormalities in the provided "
                "chest CT scan. Describe the findings in clinical terms.")
    if _PULMONARY_RE.search(findings):
        response = (
            "The chest CT shows pulmonary findings that require volumetric "
            "localisation. <VISTA3D(lung tumor)>\n" + findings)
    else:
        response = findings
    return {
        "id": f"det_{volume_id}",
        "type": "detection",
        "volume_id": volume_id,
        "images": slice_paths,
        "routes": bool(_PULMONARY_RE.search(findings)),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": response},
        ],
    }


def make_retrieval_record(volume_id, slice_paths, findings) -> dict:
    return {
        "id": f"ret_{volume_id}",
        "type": "retrieval",
        "volume_id": volume_id,
        "images": slice_paths,
        "query_text": findings[:500].strip(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrate_root", default="~/icsdg_data/ct_rate")
    p.add_argument("--output_root", default="~/icsdg_data/processed")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--max_volumes", type=int, default=2000)
    p.add_argument("--max_slices", type=int, default=16)
    p.add_argument("--slice_size", type=int, default=224)
    p.add_argument("--holdout_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    ctrate_root = Path(args.ctrate_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    slices_dir = output_root / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)

    report_index = load_report_index(ctrate_root)
    metadata_index = load_metadata_index(ctrate_root)

    from huggingface_hub import HfFileSystem
    token = args.hf_token or os.environ.get("HF_TOKEN")
    fs = HfFileSystem(token=token)

    # Derive HF volume paths directly from report-index keys. A recursive
    # `fs.glob("train/**/*.nii.gz")` over ~50k files hangs for many minutes;
    # CT-RATE's layout is fully predictable from the volume name:
    #   dataset/train/train_X/train_X_Y/train_X_Y_Z.nii.gz
    base = f"datasets/{REPO_ID}/dataset"

    def hf_path_for(volume_id: str):
        parts = volume_id.split("_")
        if len(parts) < 4 or parts[0] != "train":
            return None
        f1 = "_".join(parts[:2])          # train_X
        f2 = "_".join(parts[:3])          # train_X_Y
        return f"{base}/train/{f1}/{f2}/{volume_id}.nii.gz"

    all_volumes = [p for p in (hf_path_for(v) for v in sorted(report_index))
                   if p is not None]
    print(f"Derived {len(all_volumes)} train volume paths from the report index")

    random.shuffle(all_volumes)
    all_volumes = all_volumes[: args.max_volumes]
    n_holdout = int(len(all_volumes) * args.holdout_fraction)
    holdout_set = set(all_volumes[:n_holdout])
    print(f"Selected {len(all_volumes)} volumes "
          f"({len(all_volumes) - n_holdout} train / {n_holdout} holdout)")

    train_records, holdout_records = [], []
    tmp_dir = Path(tempfile.mkdtemp(prefix="ctrate_stream_"))

    try:
        for hf_path in tqdm(all_volumes, desc="Streaming volumes"):
            volume_id = _norm_id(Path(hf_path).name)

            report_text = report_index.get(volume_id)
            if report_text is None:
                continue
            findings = extract_findings(report_text)
            slope, intercept = metadata_index.get(volume_id, (1.0, 0.0))

            tmp_file = tmp_dir / f"{volume_id}.nii.gz"
            try:
                with fs.open(hf_path, "rb") as src, open(tmp_file, "wb") as out:
                    shutil.copyfileobj(src, out)
                slice_paths = process_volume(
                    tmp_file, slope, intercept, args.max_slices,
                    slices_dir, volume_id, args.slice_size)
            except Exception as e:
                print(f"  Warning: skipping {volume_id}: {e}")
                continue
            finally:
                tmp_file.unlink(missing_ok=True)

            if hf_path in holdout_set:
                holdout_records.append(
                    make_retrieval_record(volume_id, slice_paths, findings))
            else:
                train_records.append(
                    make_detection_record(volume_id, slice_paths, findings))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    train_out = output_root / "ctrate_train.json"
    holdout_out = output_root / "ctrate_holdout.json"
    with open(train_out, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(holdout_out, "w") as f:
        json.dump(holdout_records, f, indent=2)

    n_routed = sum(r["routes"] for r in train_records)
    print(f"\nPreprocessing complete.")
    print(f"  Train (detection)   : {len(train_records)} "
          f"({n_routed} emit <VISTA3D(lung tumor)>)")
    print(f"  Holdout (retrieval) : {len(holdout_records)}")
    print(f"  Train JSON   : {train_out}")
    print(f"  Holdout JSON : {holdout_out}")


if __name__ == "__main__":
    main()
