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
    """Read ONLY the sampled axial slices via nibabel's lazy dataobj — never
    loads the full ~300-slice volume into RAM (the cause of OOM kills)."""
    import nibabel as nib

    img = nib.load(str(nii_path))
    if len(img.shape) != 3:
        raise ValueError(f"expected a 3D volume, got shape {img.shape}")

    z = img.shape[2]                            # NIfTI axis order (H, W, Z)
    margin = max(1, int(z * 0.10))
    indices = np.linspace(margin, z - margin - 1, max_slices, dtype=int)

    slices = []
    for k in indices:
        sl = np.asarray(img.dataobj[:, :, int(k)], dtype=np.float32)
        sl = sl * slope + intercept             # CT-RATE rescale to Hounsfield Units
        slices.append(window_ct(sl))
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
        "report_text": findings.strip(),       # used by the contrastive objective
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
    p.add_argument("--max_volumes", type=int, default=10000)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel volume downloads (network-bound; 8 is safe)")
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

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download

    token = args.hf_token or os.environ.get("HF_TOKEN")
    # hf_transfer = Rust-based multi-connection downloader; big speedup if present.
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        print("hf_transfer enabled (fast downloads)")
    except ImportError:
        print("hf_transfer not installed — `pip install hf_transfer` for faster "
              "downloads")

    # Derive repo-relative volume paths from report-index keys. A recursive
    # glob over ~50k files hangs for minutes; CT-RATE's layout is predictable:
    #   dataset/train/train_X/train_X_Y/train_X_Y_Z.nii.gz
    def hf_path_for(volume_id: str):
        parts = volume_id.split("_")
        if len(parts) < 4 or parts[0] != "train":
            return None
        f1 = "_".join(parts[:2])          # train_X
        f2 = "_".join(parts[:3])          # train_X_Y
        return f"dataset/train/{f1}/{f2}/{volume_id}.nii.gz"

    all_volumes = [p for p in (hf_path_for(v) for v in sorted(report_index))
                   if p is not None]
    print(f"Derived {len(all_volumes)} train volume paths from the report index")

    random.shuffle(all_volumes)
    all_volumes = all_volumes[: args.max_volumes]
    n_holdout = int(len(all_volumes) * args.holdout_fraction)
    holdout_set = set(all_volumes[:n_holdout])
    print(f"Selected {len(all_volumes)} volumes "
          f"({len(all_volumes) - n_holdout} train / {n_holdout} holdout)")

    def fetch_and_process(rel_path: str):
        """Download one volume to a private temp dir, slice it, delete the raw."""
        volume_id = _norm_id(Path(rel_path).name)
        report_text = report_index.get(volume_id)
        if report_text is None:
            return None
        findings = extract_findings(report_text)
        slope, intercept = metadata_index.get(volume_id, (1.0, 0.0))

        # Idempotency: if every expected slice for this volume is already on
        # disk from a previous run, skip download/processing entirely.
        expected = [slices_dir / f"{volume_id}_{i:02d}.png"
                    for i in range(args.max_slices)]
        if all(p.exists() for p in expected):
            slice_paths = [f"slices/{p.name}" for p in expected]
        else:
            task_tmp = Path(tempfile.mkdtemp(prefix="ctv_"))
            try:
                local = hf_hub_download(
                    repo_id=REPO_ID, repo_type="dataset", filename=rel_path,
                    token=token, cache_dir=str(task_tmp))
                slice_paths = process_volume(
                    Path(local), slope, intercept, args.max_slices,
                    slices_dir, volume_id, args.slice_size)
            except Exception as e:
                print(f"  Warning: skipping {volume_id}: {e}")
                return None
            finally:
                shutil.rmtree(task_tmp, ignore_errors=True)

        if rel_path in holdout_set:
            return "holdout", make_retrieval_record(volume_id, slice_paths, findings)
        return "train", make_detection_record(volume_id, slice_paths, findings)

    train_records, holdout_records = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_and_process, p) for p in all_volumes]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Streaming volumes"):
            result = fut.result()
            if result is None:
                continue
            kind, rec = result
            (holdout_records if kind == "holdout" else train_records).append(rec)

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
