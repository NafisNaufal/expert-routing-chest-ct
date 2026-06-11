"""
analyze_results.py
------------------
CPU-only post-hoc analysis of detection results.

Reads per-scan Dice / IoU from detection_*.json files produced by
eval_detection.py and computes:
  - Mean ± SD per condition
  - 95 % CI (bootstrap, 10 000 iterations)
  - Wilcoxon signed-rank tests between key condition pairs
  - Scan-level detection rate  (Dice > 0)
  - Routing rate               (VISTA3D was invoked)
  - LaTeX-ready number strings for paper tables

Run locally after copying result JSONs from the HPC — no GPU required.

Usage:
    python src/eval/analyze_results.py --results_dir results/
    python src/eval/analyze_results.py --results_dir results/ --n_boot 50000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


# ── Label map: (condition, class_constraint) -> display name ──────────────────
# Add MC-LoRA entries if you have them.
LABEL_MAP = {
    ("baseline",       "none"):        "VILA-M3 baseline (no LoRA)",
    ("direct_vista3d", "none"):        "Direct VISTA3D (no VLM)",
    ("finetuned",      "none"):        "SC-LoRA",
    ("finetuned",      "lung_tumor"):  "SC-LoRA + constrained",
    ("mc_finetuned",   "none"):        "MC-LoRA",
    ("mc_finetuned",   "lung_tumor"):  "MC-LoRA + constrained",
}

# Pairs to test with Wilcoxon signed-rank.
WILCOXON_PAIRS = [
    (("finetuned",    "none"),       ("finetuned",    "lung_tumor"),  "SC-LoRA vs SC+Cstr"),
    (("mc_finetuned", "none"),       ("mc_finetuned", "lung_tumor"),  "MC-LoRA vs MC+Cstr"),
    (("finetuned",    "lung_tumor"), ("direct_vista3d", "none"),      "SC+Cstr vs Direct VISTA3D"),
    (("mc_finetuned", "lung_tumor"), ("direct_vista3d", "none"),      "MC+Cstr vs Direct VISTA3D"),
    (("finetuned",    "none"),       ("direct_vista3d", "none"),      "SC-LoRA vs Direct VISTA3D"),
]


def bootstrap_ci(data: np.ndarray, n_boot: int = 10_000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(data, size=len(data), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = 1 - ci
    return float(np.percentile(means, 100 * alpha / 2)), \
           float(np.percentile(means, 100 * (1 - alpha / 2)))


def load_condition(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    per_scan = data.get("per_scan", [])
    dice = np.array([s["dice"] for s in per_scan])
    iou  = np.array([s["iou"]  for s in per_scan])
    # routing: either the model called VISTA3D or (for direct) always True
    routed = np.array([
        1 if (data["condition"] == "direct_vista3d"
              or "VISTA3D" in s.get("response", ""))
        else 0
        for s in per_scan
    ], dtype=float)
    key = (data["condition"], data["class_constraint"])
    return {
        "key":        key,
        "label":      LABEL_MAP.get(key, f"{key[0]}|{key[1]}"),
        "dice":       dice,
        "iou":        iou,
        "routed":     routed,
        "n":          len(dice),
        "meta":       data,
    }


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/")
    ap.add_argument("--n_boot",      type=int,   default=10_000)
    ap.add_argument("--ci",          type=float, default=0.95)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    json_files  = sorted(results_dir.glob("detection_*.json"))
    if not json_files:
        print(f"No detection_*.json files in {results_dir}")
        return

    conditions = {}
    for path in json_files:
        c = load_condition(path)
        conditions[c["key"]] = c
        print(f"Loaded: {path.name}  ({c['n']} scans)")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*105}")
    print(f"{'Condition':<42} {'N':>4} {'Mean Dice':>10} {'±SD':>8} "
          f"{'95% CI':>22} {'Dice>0':>8} {'Routed':>8}")
    print(f"{'─'*105}")

    for key, c in sorted(conditions.items()):
        d = c["dice"]
        lo, hi    = bootstrap_ci(d, n_boot=args.n_boot, ci=args.ci)
        det_rate  = (d > 0).mean() * 100
        rout_rate = c["routed"].mean() * 100
        print(f"{c['label']:<42} {c['n']:>4} {d.mean():>10.4f} {d.std(ddof=1):>8.4f} "
              f"[{lo:.4f}, {hi:.4f}]{det_rate:>7.1f}%{rout_rate:>7.1f}%")

    # ── Wilcoxon tests ────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("Wilcoxon signed-rank tests (two-sided):")
    print(f"{'─'*80}")
    any_test = False
    for key_a, key_b, label in WILCOXON_PAIRS:
        if key_a not in conditions or key_b not in conditions:
            continue
        a = conditions[key_a]["dice"]
        b = conditions[key_b]["dice"]
        n = min(len(a), len(b))
        try:
            stat, p = stats.wilcoxon(a[:n], b[:n], alternative="two-sided")
            print(f"  {label:<48}: W={stat:>8.1f}, p={p:.4f}  {sig_stars(p)}")
        except ValueError as e:
            print(f"  {label:<48}: skipped ({e})")
        any_test = True
    if not any_test:
        print("  (No matching condition pairs found — check JSON filenames.)")

    # ── Sensitivity / detection rate note ─────────────────────────────────────
    print(f"\n{'─'*80}")
    print("Note on sensitivity / FPR:")
    print("  All 115 evaluation scans contain consensus nodules (≥3 radiologist")
    print("  agreement), so there are no true negative cases in this subset.")
    print("  Classical FPR cannot be computed. Reported instead:")
    print("    Dice > 0  → scan-level positive detection rate.")
    print("    Routed    → fraction of scans where VISTA3D was invoked.")

    # ── LaTeX-ready strings ───────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("LaTeX-ready numbers (paste into Table 1):")
    for key, c in sorted(conditions.items()):
        d     = c["dice"]
        lo, hi = bootstrap_ci(d, n_boot=args.n_boot, ci=args.ci)
        print(f"  {c['label']}:")
        print(f"    Dice: ${d.mean():.3f} \\pm {d.std(ddof=1):.3f}$ "
              f"\\quad 95\\%~CI: $[{lo:.3f},\\,{hi:.3f}]$")


if __name__ == "__main__":
    main()
