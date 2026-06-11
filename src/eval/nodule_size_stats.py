"""
nodule_size_stats.py
--------------------
Computes the nodule size distribution for the LIDC-IDRI evaluation subset.

Reads diameter_mm directly from nodule_annotations.json (pre-computed by
the LIDC data-preparation pipeline). Matches the 115 eval scans via patient
IDs extracted from the NIfTI filenames in the nifti directory.

Usage:
    python src/eval/nodule_size_stats.py \
        --nifti_dir ~/icsdg_data/lidc_idri/nifti \
        --annotations ~/icsdg_data/lidc_idri/nodule_annotations.json \
        --output_json results/nodule_sizes.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

SIZE_BINS   = [0, 3, 6, 10, 20, float("inf")]
SIZE_LABELS = ["< 3 mm", "3–6 mm", "6–10 mm", "10–20 mm", "> 20 mm"]

PATIENT_RE  = re.compile(r"lidc_(LIDC-IDRI-\d+)_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti_dir",   default=str(Path.home() / "icsdg_data/lidc_idri/nifti"),
                    help="Directory containing the 115 eval NIfTI volumes.")
    ap.add_argument("--annotations", default=str(Path.home() / "icsdg_data/lidc_idri/nodule_annotations.json"),
                    help="nodule_annotations.json from the LIDC prep pipeline.")
    ap.add_argument("--output_json", default=None)
    args = ap.parse_args()

    # ── Collect patient IDs from the 115 eval NIfTI filenames ────────────────
    nifti_dir = Path(args.nifti_dir).expanduser()
    eval_patients = set()
    for f in nifti_dir.glob("lidc_LIDC-IDRI-*.nii.gz"):
        m = PATIENT_RE.match(f.name)
        if m:
            eval_patients.add(m.group(1))
    print(f"Eval NIfTI files found : {len(list(nifti_dir.glob('*.nii.gz')))}")
    print(f"Unique patient IDs     : {len(eval_patients)}")

    # ── Load annotations and filter to eval patients ─────────────────────────
    with open(Path(args.annotations).expanduser()) as f:
        ann = json.load(f)

    diameters  = []
    scan_ids   = []
    empty_scans = 0
    for entry in ann:
        if entry["patient_id"] not in eval_patients:
            continue
        nodules = entry.get("nodules", [])
        if not nodules:
            empty_scans += 1
            continue
        for nod in nodules:
            d = nod.get("diameter_mm")
            if d and d > 0:
                diameters.append(float(d))
                scan_ids.append(entry["patient_id"])

    print(f"Matched annotation entries : {sum(1 for e in ann if e['patient_id'] in eval_patients)}")
    print(f"Entries with no nodules    : {empty_scans}")
    print(f"Total nodule measurements  : {len(diameters)}\n")

    d = np.array(diameters)
    n = len(d)
    if n == 0:
        print("No nodule diameters found — check patient_id format.")
        return

    print(f"── Nodule Size Distribution ({n} nodules, {len(eval_patients)} patients) ──")
    print(f"  Mean   : {d.mean():.1f} mm  (SD {d.std(ddof=1):.1f} mm)")
    print(f"  Median : {np.median(d):.1f} mm")
    print(f"  Range  : [{d.min():.1f}, {d.max():.1f}] mm\n")
    print(f"  {'Size bin':<12}  {'N':>5}  {'%':>7}")
    print(f"  {'─'*28}")
    for i, label in enumerate(SIZE_LABELS):
        lo_v = SIZE_BINS[i]
        hi_v = SIZE_BINS[i + 1]
        cnt  = int(((d >= lo_v) & (d < hi_v)).sum())
        print(f"  {label:<12}  {cnt:>5}  {100*cnt/n:>6.1f}%")

    frac_large = float((d >= 10).mean())
    frac_small = float((d < 6).mean())
    print(f"\n  Fraction ≥ 10 mm (plausibly within VISTA3D lung-tumor range): "
          f"{100*frac_large:.1f}%")
    print(f"  Fraction  < 6 mm (sub-solid / ground-glass, hardest for lung-mass prior): "
          f"{100*frac_small:.1f}%")

    print(f"\n── LaTeX-ready description ──────────────────────────────────────────")
    print(f"  Nodule diameter (mean $\\pm$ SD): "
          f"${d.mean():.1f}\\,\\pm\\,{d.std(ddof=1):.1f}$\\,mm, "
          f"range $[{d.min():.1f},\\,{d.max():.1f}]$\\,mm; "
          f"{100*frac_small:.0f}\\%\\ of nodules $<6$\\,mm.")

    if args.output_json:
        out = {
            "n_nodules": n,
            "n_patients": len(eval_patients),
            "mean_mm":   round(float(d.mean()), 2),
            "sd_mm":     round(float(d.std(ddof=1)), 2),
            "median_mm": round(float(np.median(d)), 2),
            "min_mm":    round(float(d.min()), 2),
            "max_mm":    round(float(d.max()), 2),
            "frac_lt6mm":   round(frac_small, 4),
            "frac_ge10mm":  round(frac_large, 4),
            "bins": {
                label: int(((d >= SIZE_BINS[i]) & (d < SIZE_BINS[i+1])).sum())
                for i, label in enumerate(SIZE_LABELS)
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
