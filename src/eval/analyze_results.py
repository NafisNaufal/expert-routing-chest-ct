"""
analyze_results.py
------------------
CPU-only post-hoc analysis of detection results.

Reads per-scan Dice / IoU from *_detection*.json files and computes:
  - Mean ± SD per condition
  - 95 % CI (bootstrap, 10 000 iterations)
  - Wilcoxon signed-rank tests between key condition pairs
  - Scan-level detection rate  (Dice > 0)
  - Routing rate               (from top-level routing_rate field)
  - LaTeX-ready number strings for paper tables

Usage:
    python src/eval/analyze_results.py --results_dir results/
    python src/eval/analyze_results.py --results_dir results/ --n_boot 50000
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


# ── Explicit filename → (sort_key, display_label) map ────────────────────────
# Only files listed here are loaded; others in results/ are ignored.
FILE_MAP = {
    "baseline_detection.json":                     ("0_baseline",        "Baseline (no LoRA, no routing)"),
    "direct_vista3d_detection.json":               ("1_direct",          "Direct VISTA3D (ceiling)"),
    "finetuned_detection.json":                    ("2_sc9k",            "SC-LoRA 9k (unconstrained)"),
    "finetuned_detection_10k.json":                ("3_sc10k",           "SC-LoRA 10k (unconstrained)"),
    "finetuned_detection_10k_constrained.json":    ("4_sc10k_cstr",      "SC-LoRA 10k + constrained (partial)"),
    "finetuned_detection_10k_constrained_v2.json": ("5_sc10k_cstr_full", "SC-LoRA 10k + constrained (full)"),
    "finetuned_detection_multiclass.json":         ("6_mc",              "MC-LoRA (unconstrained)"),
    "finetuned_detection_multiclass_constrained.json": ("7_mc_cstr",     "MC-LoRA + constrained (full)"),
}

# Wilcoxon pairs: (file_a_stem, file_b_stem, display_label)
WILCOXON_PAIRS = [
    ("finetuned_detection_10k",              "finetuned_detection_10k_constrained_v2", "SC-LoRA 10k  vs  SC+Cstr (full)"),
    ("finetuned_detection_multiclass",       "finetuned_detection_multiclass_constrained", "MC-LoRA  vs  MC+Cstr"),
    ("finetuned_detection_10k_constrained_v2", "direct_vista3d_detection",             "SC+Cstr (full)  vs  Direct"),
    ("finetuned_detection_multiclass_constrained", "direct_vista3d_detection",         "MC+Cstr  vs  Direct"),
    ("finetuned_detection_10k",              "direct_vista3d_detection",               "SC-LoRA 10k  vs  Direct"),
    ("finetuned_detection_multiclass",       "finetuned_detection_10k",                "MC-LoRA  vs  SC-LoRA 10k"),
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


def load_file(path: Path, label: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    per_scan = data.get("per_scan", [])
    dice = np.array([s["dice"] for s in per_scan])
    iou  = np.array([s["iou"]  for s in per_scan])
    routing_rate = float(data.get("routing_rate", 0.0))
    return {
        "stem":         path.stem,
        "label":        label,
        "dice":         dice,
        "iou":          iou,
        "routing_rate": routing_rate,
        "n":            len(dice),
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
    conditions  = {}   # stem -> condition dict

    for fname, (sort_key, label) in FILE_MAP.items():
        path = results_dir / fname
        if not path.exists():
            print(f"MISSING (skipped): {fname}")
            continue
        c = load_file(path, label)
        c["sort_key"] = sort_key
        conditions[c["stem"]] = c
        print(f"Loaded: {fname}  ({c['n']} scans, routing={c['routing_rate']:.3f})")

    if not conditions:
        print(f"No matching detection JSON files found in {results_dir}")
        return

    ordered = sorted(conditions.values(), key=lambda x: x["sort_key"])

    # ── Summary table ─────────────────────────────────────────────────────────
    W = 110
    print(f"\n{'─'*W}")
    print(f"{'Condition':<45} {'N':>4} {'Mean Dice':>10} {'±SD':>8} "
          f"{'95% CI':>22} {'Dice>0%':>8} {'Routed%':>8}")
    print(f"{'─'*W}")

    for c in ordered:
        d = c["dice"]
        lo, hi   = bootstrap_ci(d, n_boot=args.n_boot, ci=args.ci)
        det_rate = (d > 0).mean() * 100
        rout_pct = c["routing_rate"] * 100
        print(f"{c['label']:<45} {c['n']:>4} {d.mean():>10.4f} {d.std(ddof=1):>8.4f} "
              f"[{lo:.4f}, {hi:.4f}]{det_rate:>7.1f}%{rout_pct:>7.1f}%")

    # ── Wilcoxon tests ────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("Wilcoxon signed-rank tests (two-sided):")
    print(f"{'─'*80}")
    any_test = False
    for stem_a, stem_b, label in WILCOXON_PAIRS:
        if stem_a not in conditions or stem_b not in conditions:
            print(f"  {label:<52}: skipped (file missing)")
            continue
        a = conditions[stem_a]["dice"]
        b = conditions[stem_b]["dice"]
        n = min(len(a), len(b))
        try:
            stat, p = stats.wilcoxon(a[:n], b[:n], alternative="two-sided")
            print(f"  {label:<52}: W={stat:>8.1f}  p={p:.4f}  {sig_stars(p)}")
        except ValueError as e:
            print(f"  {label:<52}: skipped ({e})")
        any_test = True

    if not any_test:
        print("  (No matching condition pairs found.)")

    # ── Notes ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("Note: All 115 scans contain ≥3-radiologist consensus nodules → no true")
    print("negatives in this subset. Classical FPR cannot be computed.")
    print("  Dice > 0   = scan-level positive detection rate.")
    print("  Routed%    = fraction of scans where VISTA3D was invoked (top-level")
    print("               routing_rate from eval JSON, not per-scan response field).")

    # ── LaTeX-ready strings ───────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("LaTeX-ready numbers (paste into Table 1):")
    for c in ordered:
        d      = c["dice"]
        lo, hi = bootstrap_ci(d, n_boot=args.n_boot, ci=args.ci)
        print(f"  {c['label']}:")
        print(f"    Dice: ${d.mean():.3f} \\pm {d.std(ddof=1):.3f}$  "
              f"95\\%~CI: $[{lo:.3f},\\,{hi:.3f}]$  "
              f"Routing: {c['routing_rate']*100:.1f}\\%")


if __name__ == "__main__":
    main()
