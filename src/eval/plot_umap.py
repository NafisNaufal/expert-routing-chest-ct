"""
plot_umap.py
------------
UMAP visualisation of holdout image-side and text-side embeddings, comparing
the pre-fine-tuned baseline against the joint-LoRA fine-tuned model.

Inputs: two .npz files written by `eval_retrieval.py --save_embeddings`, one
for the baseline (no LoRA) and one for the fine-tuned model. Each contains
`image_emb`, `text_emb`, and `ids` arrays for the same hold-out volumes.

The resulting figure shows the two embedding clouds side by side: the
fine-tuned panel reveals whether the contrastive objective pulled image
and text embeddings of the same volume closer than the baseline does.

Usage:
    python src/eval/plot_umap.py \
        --baseline_npz results/baseline_embeddings.npz \
        --finetuned_npz results/finetuned_embeddings.npz \
        --output paper/figures/embeddings_umap.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_npz", required=True)
    p.add_argument("--finetuned_npz", required=True)
    p.add_argument("--output", default="paper/figures/embeddings_umap.pdf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_pair_lines", type=int, default=30,
                   help="How many matched image-text pairs to draw guide lines for.")
    return p.parse_args()


def _l2norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def _project(image_emb, text_emb, seed):
    """Fit one UMAP on the concatenation so the two clouds share a coord system."""
    stacked = np.concatenate([_l2norm(image_emb), _l2norm(text_emb)], axis=0)
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed,
    )
    coords = reducer.fit_transform(stacked)
    n = image_emb.shape[0]
    return coords[:n], coords[n:]


def _panel(ax, img_xy, txt_xy, title, n_lines):
    rng = np.random.default_rng(0)
    n = img_xy.shape[0]
    idx = rng.choice(n, size=min(n_lines, n), replace=False)
    for i in idx:
        ax.plot([img_xy[i, 0], txt_xy[i, 0]],
                [img_xy[i, 1], txt_xy[i, 1]],
                color="gray", alpha=0.25, linewidth=0.6, zorder=1)
    ax.scatter(img_xy[:, 0], img_xy[:, 1], s=14, c="#1f77b4",
               marker="o", label=f"image-side ($n={n}$)",
               edgecolors="none", alpha=0.7, zorder=2)
    ax.scatter(txt_xy[:, 0], txt_xy[:, 1], s=14, c="#d62728",
               marker="^", label=f"text-side ($n={n}$)",
               edgecolors="none", alpha=0.7, zorder=2)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    args = parse_args()

    base = np.load(args.baseline_npz, allow_pickle=True)
    fine = np.load(args.finetuned_npz, allow_pickle=True)

    img_b, txt_b = _project(base["image_emb"], base["text_emb"], args.seed)
    img_f, txt_f = _project(fine["image_emb"], fine["text_emb"], args.seed)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    _panel(axes[0], img_b, txt_b,
           "Pre-fine-tuning (baseline VILA-M3)", args.n_pair_lines)
    _panel(axes[1], img_f, txt_f,
           "After joint LoRA (ours)", args.n_pair_lines)
    fig.suptitle(
        "Hold-out image-side vs text-side hidden-state embeddings (UMAP)",
        fontsize=11,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
