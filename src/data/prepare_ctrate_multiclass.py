"""
prepare_ctrate_multiclass.py
----------------------------
Multi-class extension of prepare_ctrate.py. Same streaming CT-RATE pipeline,
but generates training samples for FIVE routing classes instead of one.

Why multi-class:
- Demonstrates the framework's "general expert routing" claim — same VLM,
  same adapter, routes to different VISTA3D classes based on query intent.
- Each CT volume now yields multiple training samples (one per relevant
  query/class), inflating the effective dataset size.
- Trains the model to bind a structured routing token to the specific
  pathology or anatomical structure named in the query — not just to any
  thoracic abnormality.

Routing vocabulary (all are confirmed VISTA3D classes):
    lung tumor     pulmonary nodules / masses / lesions
    heart          cardiac structures / cardiomegaly
    liver          liver outline (visible in lower chest CT)
    aorta          aortic arch / thoracic aorta
    lung           general lung anatomy / parenchyma

Each routing class has:
- A regex over the report findings (when applicable; anatomy classes don't
  need report keywords, they're always present in chest CT).
- A list of query templates phrased to elicit that specific routing token.
- A target VISTA3D label suitable for downstream segmentation.

For each volume, we generate one training record per relevant class plus
one descriptive (no-route) record. The retrieval record is unchanged from
the single-class pipeline.

Usage:
    export HF_TOKEN=your_token_here
    python src/data/prepare_ctrate_multiclass.py \\
        --ctrate_root ~/erct_data/ct_rate \\
        --output_root ~/erct_data/processed_multiclass \\
        --max_volumes 9000
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


# ── Routing vocabulary ────────────────────────────────────────────────────────
# Each entry defines a class the VLM can be trained to route to. The
# `report_regex` is None for anatomy classes that are present in every chest
# CT regardless of pathology (we still want queries about them to route).
ROUTING_CLASSES = {
    "lung tumor": {
        "report_regex": re.compile(
            r"\b(nodul|mass|lesion|opacit|consolidat|tumou?r|"
            r"ground.?glass|ggo|spiculat|metasta|malignan)\w*",
            re.IGNORECASE),
        "query_templates": [
            "Identify and localise pulmonary nodules in this chest CT scan.",
            "Find any pulmonary masses or lesions visible in the scan.",
            "Localise lung tumours or suspicious pulmonary findings.",
        ],
        "response_prefix": (
            "The chest CT shows pulmonary findings that require volumetric "
            "localisation. "),
    },
    "heart": {
        "report_regex": None,            # heart is always present in chest CT
        "query_templates": [
            "Identify and localise the cardiac structures in this CT.",
            "Show the heart in this chest CT scan.",
            "Localise the heart for size and shape assessment.",
        ],
        "response_prefix": (
            "Cardiac localisation is delegated to the volumetric expert. "),
    },
    "liver": {
        "report_regex": re.compile(
            r"\b(liver|hepatic|hepato)\w*", re.IGNORECASE),
        "query_templates": [
            "Localise the liver in this CT scan.",
            "Show the hepatic outline on this scan.",
            "Identify the liver boundaries.",
        ],
        "response_prefix": (
            "Hepatic localisation is delegated to the volumetric expert. "),
    },
    "aorta": {
        "report_regex": re.compile(
            r"\b(aorta|aortic|aneurysm|dissection)\w*", re.IGNORECASE),
        "query_templates": [
            "Identify the aortic arch in this scan.",
            "Localise the thoracic aorta.",
            "Show the aorta on this chest CT.",
        ],
        "response_prefix": (
            "Aortic localisation is delegated to the volumetric expert. "),
    },
    "lung": {
        "report_regex": None,            # lung is always present in chest CT
        "query_templates": [
            "Show the lung parenchyma boundaries.",
            "Localise the lungs in this CT scan.",
            "Identify the lung anatomy.",
        ],
        "response_prefix": (
            "Pulmonary localisation is delegated to the volumetric expert. "),
    },
}

# Descriptive (no-route) query templates — train the model to NOT emit a
# routing token when the query asks for general description rather than
# spatial localisation.
DESCRIPTIVE_QUERIES = [
    "Describe the findings in this chest CT scan in clinical terms.",
    "Summarise the radiological observations from this CT.",
    "Provide a clinical impression of the scan.",
]

_FINDING_RE = re.compile(
    r"(findings?|impression|conclusion)[:\s]*(.*?)(?=findings?|impression|"
    r"conclusion|$)", re.IGNORECASE | re.DOTALL)


# ── Report / metadata indices (unchanged from single-class) ───────────────────

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
    import pandas as pd
    meta_dir = ctrate_root / "metadata"
    csvs = list(meta_dir.glob("*.csv")) if meta_dir.exists() else []
    if not csvs:
        print("WARNING: no metadata CSVs found — assuming HU units.")
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


# ── Volume → slices (unchanged) ───────────────────────────────────────────────

def window_ct(vol: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)


def process_volume(nii_path: Path, slope: float, intercept: float,
                   max_slices: int, slices_dir: Path, volume_id: str,
                   slice_size: int) -> list:
    import nibabel as nib
    img = nib.load(str(nii_path))
    if len(img.shape) != 3:
        raise ValueError(f"expected a 3D volume, got shape {img.shape}")
    z = img.shape[2]
    margin = max(1, int(z * 0.10))
    indices = np.linspace(margin, z - margin - 1, max_slices, dtype=int)
    paths = []
    for i, k in enumerate(indices):
        sl = np.asarray(img.dataobj[:, :, int(k)], dtype=np.float32)
        sl = sl * slope + intercept
        sl = window_ct(sl)
        Image.fromarray(sl).convert("RGB").resize(
            (slice_size, slice_size), Image.BILINEAR
        ).save(slices_dir / f"{volume_id}_{i:02d}.png")
        paths.append(f"slices/{volume_id}_{i:02d}.png")
    return paths


# ── Multi-class instruction record builders ───────────────────────────────────

def detect_applicable_classes(findings: str) -> list:
    """
    Return the list of routing classes whose criteria are satisfied by this
    report. Anatomy classes (regex=None) are always included; pathology
    classes are included only if their keyword regex matches the findings.
    """
    applicable = []
    for class_name, spec in ROUTING_CLASSES.items():
        if spec["report_regex"] is None:
            applicable.append(class_name)
        elif spec["report_regex"].search(findings):
            applicable.append(class_name)
    return applicable


def make_routing_record(volume_id: str, slice_paths: list, class_name: str,
                        query: str, findings: str, sample_idx: int) -> dict:
    """One causal-LM sample: query → <VISTA3D(class)> + clinical context."""
    spec = ROUTING_CLASSES[class_name]
    prompt = "<image>\n" * len(slice_paths) + query
    response = (
        spec["response_prefix"]
        + f"<VISTA3D({class_name})>\n"
        + findings
    )
    return {
        "id": f"det_{volume_id}_{class_name.replace(' ', '_')}_{sample_idx}",
        "type": "detection",
        "volume_id": volume_id,
        "routing_class": class_name,
        "images": slice_paths,
        "routes": True,
        "report_text": findings.strip(),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt",   "value": response},
        ],
    }


def make_descriptive_record(volume_id: str, slice_paths: list,
                            query: str, findings: str) -> dict:
    """Negative-routing sample: descriptive query → findings without token."""
    prompt = "<image>\n" * len(slice_paths) + query
    return {
        "id": f"desc_{volume_id}",
        "type": "detection",
        "volume_id": volume_id,
        "routing_class": None,
        "images": slice_paths,
        "routes": False,
        "report_text": findings.strip(),
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt",   "value": findings},
        ],
    }


def make_retrieval_record(volume_id: str, slice_paths: list, findings: str) -> dict:
    return {
        "id": f"ret_{volume_id}",
        "type": "retrieval",
        "volume_id": volume_id,
        "images": slice_paths,
        "query_text": findings[:500].strip(),
    }


def build_records_for_volume(volume_id: str, slice_paths: list, findings: str,
                             rng: random.Random,
                             include_descriptive: bool = True) -> list:
    """Produce the full set of training records for a single volume."""
    records = []
    applicable = detect_applicable_classes(findings)
    for class_name in applicable:
        templates = ROUTING_CLASSES[class_name]["query_templates"]
        # Use one random query per class per volume to keep the dataset diverse
        # without exploding its size (otherwise N_classes * N_templates samples).
        query = rng.choice(templates)
        records.append(make_routing_record(
            volume_id, slice_paths, class_name, query, findings,
            sample_idx=len(records)))
    if include_descriptive:
        query = rng.choice(DESCRIPTIVE_QUERIES)
        records.append(make_descriptive_record(
            volume_id, slice_paths, query, findings))
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrate_root", default="~/erct_data/ct_rate")
    p.add_argument("--output_root", default="~/erct_data/processed_multiclass")
    p.add_argument("--hf_token", default=None)
    p.add_argument("--max_volumes", type=int, default=9000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max_slices", type=int, default=16)
    p.add_argument("--slice_size", type=int, default=224)
    p.add_argument("--holdout_fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reuse_slices_from",
                   default="~/erct_data/processed",
                   help="If set, reuse already-extracted slices from this "
                        "directory instead of re-downloading volumes. Falls "
                        "back to streaming download when slices are missing.")
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    ctrate_root = Path(args.ctrate_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    slices_dir = output_root / "slices"
    slices_dir.mkdir(parents=True, exist_ok=True)
    reuse_root = Path(args.reuse_slices_from).expanduser() if args.reuse_slices_from else None

    report_index = load_report_index(ctrate_root)
    metadata_index = load_metadata_index(ctrate_root)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from huggingface_hub import hf_hub_download

    token = args.hf_token or os.environ.get("HF_TOKEN")
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        print("hf_transfer enabled (fast downloads)")
    except ImportError:
        print("hf_transfer not installed — slower downloads")

    def hf_path_for(volume_id: str):
        parts = volume_id.split("_")
        if len(parts) < 4 or parts[0] != "train":
            return None
        f1 = "_".join(parts[:2])
        f2 = "_".join(parts[:3])
        return f"dataset/train/{f1}/{f2}/{volume_id}.nii.gz"

    all_volumes = [p for p in (hf_path_for(v) for v in sorted(report_index))
                   if p is not None]
    print(f"Derived {len(all_volumes)} train volume paths")

    rng_shuf = random.Random(args.seed)
    rng_shuf.shuffle(all_volumes)
    all_volumes = all_volumes[: args.max_volumes]
    n_holdout = int(len(all_volumes) * args.holdout_fraction)
    holdout_set = set(all_volumes[:n_holdout])
    print(f"Selected {len(all_volumes)} volumes "
          f"({len(all_volumes) - n_holdout} train / {n_holdout} holdout)")

    def get_slice_paths(volume_id: str, rel_path: str):
        """Reuse existing slices if available; otherwise stream-download."""
        expected_local = [slices_dir / f"{volume_id}_{i:02d}.png"
                          for i in range(args.max_slices)]
        if all(p.exists() for p in expected_local):
            return [f"slices/{p.name}" for p in expected_local]

        # Reuse from single-class run if requested
        if reuse_root is not None:
            reuse_slices_dir = reuse_root / "slices"
            expected_reuse = [reuse_slices_dir / f"{volume_id}_{i:02d}.png"
                              for i in range(args.max_slices)]
            if all(p.exists() for p in expected_reuse):
                # Symlink into the multi-class output for self-containedness
                for src in expected_reuse:
                    dst = slices_dir / src.name
                    if not dst.exists():
                        os.symlink(src, dst)
                return [f"slices/{p.name}" for p in expected_reuse]

        # Fall back to download + slice
        task_tmp = Path(tempfile.mkdtemp(prefix="ctv_"))
        try:
            local = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", filename=rel_path,
                token=token,
                cache_dir=str(task_tmp), local_dir=str(task_tmp),
                local_dir_use_symlinks=False)
            slope, intercept = metadata_index.get(volume_id, (1.0, 0.0))
            return process_volume(
                Path(local), slope, intercept, args.max_slices,
                slices_dir, volume_id, args.slice_size)
        finally:
            shutil.rmtree(task_tmp, ignore_errors=True)

    def fetch_and_process(rel_path: str):
        volume_id = _norm_id(Path(rel_path).name)
        report_text = report_index.get(volume_id)
        if report_text is None:
            return None
        findings = extract_findings(report_text)

        try:
            slice_paths = get_slice_paths(volume_id, rel_path)
        except Exception as e:
            print(f"  Skip {volume_id}: {e}")
            return None

        if rel_path in holdout_set:
            return "holdout", [make_retrieval_record(volume_id, slice_paths, findings)]

        # Multi-class training records: one per applicable class + descriptive
        records = build_records_for_volume(
            volume_id, slice_paths, findings, rng,
            include_descriptive=True)
        return "train", records

    train_records, holdout_records = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(fetch_and_process, p) for p in all_volumes]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Streaming volumes"):
            try:
                result = fut.result()
            except Exception as e:
                print(f"  Worker error: {e}")
                continue
            if result is None:
                continue
            kind, recs = result
            if kind == "holdout":
                holdout_records.extend(recs)
            else:
                train_records.extend(recs)

    train_out = output_root / "ctrate_train_multiclass.json"
    holdout_out = output_root / "ctrate_holdout_multiclass.json"
    with open(train_out, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(holdout_out, "w") as f:
        json.dump(holdout_records, f, indent=2)

    # Per-class summary
    from collections import Counter
    class_counts = Counter(r.get("routing_class") or "(descriptive)"
                           for r in train_records)

    print(f"\nMulti-class preprocessing complete.")
    print(f"  Train samples       : {len(train_records)}")
    print(f"  Holdout retrieval   : {len(holdout_records)}")
    print(f"  Train JSON          : {train_out}")
    print(f"  Holdout JSON        : {holdout_out}")
    print(f"\n  Per-class breakdown:")
    for cls, n in class_counts.most_common():
        print(f"    {cls:<24} {n:>6}")


if __name__ == "__main__":
    main()
