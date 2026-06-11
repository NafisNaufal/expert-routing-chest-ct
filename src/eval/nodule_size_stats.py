"""
nodule_size_stats.py
--------------------
Computes the nodule size distribution for the LIDC-IDRI evaluation subset.

Reads ground-truth NIfTI masks referenced in lidc_eval.json and reports:
  - Effective sphere diameter per scan  (from voxel count × voxel spacing)
  - Mean / median / range
  - Histogram by clinical size bins
  - Fraction in the plausible VISTA3D lung-tumor range (> 10 mm)

Run locally after copying lidc_eval.json from the HPC — no GPU required.

Usage:
    python src/eval/nodule_size_stats.py \
        --eval_json /path/to/lidc_eval.json
    python src/eval/nodule_size_stats.py \
        --eval_json results/lidc_eval.json --output_json results/nodule_sizes.json
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


SIZE_BINS   = [0, 3, 6, 10, 20, float("inf")]
SIZE_LABELS = ["< 3 mm", "3–6 mm", "6–10 mm", "10–20 mm", "> 20 mm"]


def effective_diameter_mm(mask: np.ndarray, zooms: tuple) -> float:
    """Equivalent sphere diameter in mm from a binary voxel mask."""
    n_vox = int(mask.sum())
    if n_vox == 0:
        return 0.0
    voxel_vol_mm3 = float(np.prod(np.abs(zooms)))
    vol_mm3       = n_vox * voxel_vol_mm3
    radius_mm     = (3.0 * vol_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)
    return float(2.0 * radius_mm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_json",   required=True,
                    help="Path to lidc_eval.json (on local after HPC copy)")
    ap.add_argument("--output_json", default=None,
                    help="Optional: save per-scan diameters to JSON")
    args = ap.parse_args()

    with open(args.eval_json) as f:
        records = json.load(f)

    diameters   = []
    scan_ids    = []
    missing     = 0

    for rec in records:
        gt_path = rec.get("gt_mask_path")
        if not gt_path or not Path(gt_path).exists():
            missing += 1
            continue
        img    = nib.load(gt_path)
        mask   = (img.get_fdata() > 0).astype(np.uint8)
        zooms  = img.header.get_zooms()[:3]   # (dz, dy, dx) in mm
        d      = effective_diameter_mm(mask, zooms)
        if d > 0:
            diameters.append(d)
            scan_ids.append(rec.get("volume_id", gt_path))

    if missing:
        print(f"Warning: {missing} records had missing or inaccessible mask paths.")

    d = np.array(diameters)
    n = len(d)

    print(f"\n── Nodule Size Distribution ({n} scans) ──────────────────────────────")
    print(f"  Mean   : {d.mean():.1f} mm  (SD {d.std(ddof=1):.1f} mm)")
    print(f"  Median : {np.median(d):.1f} mm")
    print(f"  Range  : [{d.min():.1f}, {d.max():.1f}] mm")
    print()
    print(f"  {'Size bin':<12}  {'N':>5}  {'%':>7}")
    print(f"  {'─'*28}")
    for i, label in enumerate(SIZE_LABELS):
        lo_v = SIZE_BINS[i]
        hi_v = SIZE_BINS[i + 1]
        cnt  = int(((d >= lo_v) & (d < hi_v)).sum())
        print(f"  {label:<12}  {cnt:>5}  {100*cnt/n:>6.1f}%")

    frac_large = float((d >= 10).mean())
    print(f"\n  Fraction ≥ 10 mm (plausibly within VISTA3D lung-tumor range): "
          f"{100*frac_large:.1f}%")
    print(f"  Fraction  < 6 mm (sub-solid / ground-glass nodules,  "
          f"hardest for lung-mass prior): {100*(d < 6).mean():.1f}%")

    # ── LaTeX helper ──────────────────────────────────────────────────────────
    print(f"\n── LaTeX-ready description ──────────────────────────────────────────")
    print(f"  Nodule effective diameter: mean {d.mean():.1f}\\,mm (SD {d.std(ddof=1):.1f}\\,mm,")
    print(f"  range [{d.min():.1f}\\,--\\,{d.max():.1f}]\\,mm); "
          f"{100*(d<6).mean():.0f}\\%\\ of nodules $<6$\\,mm.")

    if args.output_json:
        out = {
            "n_scans": n,
            "mean_mm": round(float(d.mean()), 2),
            "sd_mm":   round(float(d.std(ddof=1)), 2),
            "median_mm": round(float(np.median(d)), 2),
            "min_mm":  round(float(d.min()), 2),
            "max_mm":  round(float(d.max()), 2),
            "bins": {
                label: int(((d >= SIZE_BINS[i]) & (d < SIZE_BINS[i+1])).sum())
                for i, label in enumerate(SIZE_LABELS)
            },
            "per_scan": [
                {"scan_id": sid, "diameter_mm": round(float(di), 2)}
                for sid, di in zip(scan_ids, diameters)
            ],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
