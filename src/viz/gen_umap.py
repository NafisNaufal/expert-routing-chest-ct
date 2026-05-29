"""
gen_umap.py
-----------
Generate a UMAP visualisation of query-conditioned VLM embeddings for the
multi-class routing paper figure.

Experimental design
-------------------
For N holdout CT volumes we run 5 inference calls each (one per routing class)
and record the last-token hidden state of the VILA-M3 language body, giving
N×5 vectors per adapter configuration.  UMAP projects all embeddings into 2-D.

If multi-class routing is functioning, same-class queries should cluster
together regardless of the input volume — demonstrating that the LoRA adapter
has learned query-conditioned representations.  We compare:
  • Baseline  (pre-fine-tuned VILA-M3, no LoRA)
  • Fine-tuned (multi-class LoRA adapter)

Expected outcome
----------------
Baseline:    five query classes intermixed (no query-specific structure)
Fine-tuned:  at minimum the ``liver'' class (100 % class precision) forms a
             separable cluster; other classes may show partial separation
             consistent with the Table 3 per-class routing results.

Usage
-----
# First run (extracts embeddings — ~30-60 min on A100):
python src/viz/gen_umap.py \\
    --config  configs/train_config_multiclass.yaml \\
    --holdout ~/icsdg_data/processed_multiclass/ctrate_holdout.json \\
    --lora    ./checkpoints_multiclass/lora_adapter_final \\
    --n       200 \\
    --output  paper/figures/umap_multiclass.pdf \\
    --cache   results/umap_embeddings.npz

# Subsequent runs (re-plot without re-running inference):
python src/viz/gen_umap.py \\
    --cache   results/umap_embeddings.npz \\
    --output  paper/figures/umap_multiclass.pdf
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image as PILImage
from tqdm import tqdm


# ── Query templates (same as eval_routing_multiclass.py) ─────────────────────
QUERY_PER_CLASS = {
    "lung tumor": "Identify and localise pulmonary nodules in this chest CT scan.",
    "heart":      "Identify and localise the cardiac structures in this CT.",
    "liver":      "Localise the liver in this CT scan.",
    "aorta":      "Identify the aortic arch in this scan.",
    "lung":       "Show the lung parenchyma boundaries.",
}
CLASS_ORDER = list(QUERY_PER_CLASS.keys())

# ── Colour palette matching paper's pgfplots figures ─────────────────────────
# Each class gets a distinctive colour that is also print-friendly.
CLASS_COLOURS = {
    "lung tumor": "#2C5F8F",   # clrblue
    "heart":      "#CC7100",   # clrorange
    "liver":      "#1F7864",   # clrteal
    "aorta":      "#8B2FC9",   # purple
    "lung":       "#B8292B",   # red
}
CLASS_MARKERS = {
    "lung tumor": "o",
    "heart":      "s",
    "liver":      "^",
    "aorta":      "D",
    "lung":       "P",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="configs/train_config_multiclass.yaml")
    p.add_argument("--holdout",  default="~/icsdg_data/processed_multiclass/ctrate_holdout.json")
    p.add_argument("--vila_repo",default="./VLM-Radiology-Agent-Framework")
    p.add_argument("--model",    default="MONAI/Llama3-VILA-M3-8B")
    p.add_argument("--lora",     default="./checkpoints_multiclass/lora_adapter_final")
    p.add_argument("--n",        type=int, default=200,
                   help="Volumes to embed (each yields 5 embeddings per adapter).")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output",   default="paper/figures/umap_multiclass.pdf")
    p.add_argument("--cache",    default="results/umap_embeddings.npz",
                   help="Save/load embeddings to skip re-running inference.")
    p.add_argument("--umap_neighbors", type=int, default=15)
    p.add_argument("--umap_min_dist",  type=float, default=0.10)
    return p.parse_args()


def add_vila_to_path(vila_framework: str) -> None:
    root = Path(vila_framework).expanduser().resolve()
    for p in [root, root / "m3", root / "m3" / "demo",
              root / "thirdparty" / "VILA"]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def init_pg_if_needed() -> None:
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29501")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


# ── Embedding extraction ──────────────────────────────────────────────────────

def _find_last_decoder_layer(model):
    """
    Navigate through PeftModel / LlavaLlamaForCausalLM wrappers to find the
    last LLaMA decoder layer, which we hook to capture hidden states.

    Typical hierarchies:
      Baseline:   LlavaLlamaForCausalLM  → .model (LlavaLlamaModel) → .layers
      Fine-tuned: PeftModel → .base_model.model (LlavaLlamaForCausalLM)
                           → .model (LlavaLlamaModel) → .layers
    """
    m = model
    # Unwrap PeftModel
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model            # now LlavaLlamaForCausalLM
    # Unwrap LlavaLlamaForCausalLM → LlavaLlamaModel
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers[-1]
    # Fallback: bare LlavaLlamaModel
    if hasattr(m, "layers"):
        return m.layers[-1]
    raise RuntimeError(f"Cannot locate decoder layers in {type(model)}")


@torch.no_grad()
def extract_embedding(model, tokenizer, image_processor,
                      query: str, images, device,
                      last_layer) -> np.ndarray:
    """
    Return last-token hidden state of the last decoder layer as a unit vector.

    We use model.generate() (identical call to eval_routing_multiclass.py) with
    a forward hook on the last decoder layer to intercept hidden states.
    Calling model() directly fails for LLaVA because image-token injection is
    only triggered inside generate()/prepare_inputs_for_generation().
    """
    from llava.mm_utils import tokenizer_image_token, process_images      # type: ignore
    from llava.conversation import conv_templates, SeparatorStyle          # type: ignore
    from llava.constants import IMAGE_TOKEN_INDEX                          # type: ignore

    prompt_text = "<image>\n" * len(images) + query
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], prompt_text)
    if conv.sep_style == SeparatorStyle.LLAMA_3:
        conv.append_message(conv.roles[1], "")
    else:
        conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    imgs_tensor = process_images(images, image_processor, model.config).to(
        device=device, dtype=next(model.parameters()).dtype)

    # ── forward hook captures last decoder layer output ───────────────────
    captured: dict = {}

    def _hook(module, inp, out):
        # LLaMA decoder layer returns (hidden_states, ...) tuple
        h = out[0] if isinstance(out, tuple) else out
        captured["h"] = h.detach()          # (1, seq_len, hidden_dim)

    handle = last_layer.register_forward_hook(_hook)
    try:
        model.generate(
            input_ids,
            images=[imgs_tensor],
            max_new_tokens=1,
            min_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        handle.remove()

    if "h" not in captured:
        return np.zeros(4096, dtype=np.float32)

    # Take last token of the prefill pass (the hook fires once per generate call
    # for the full input sequence; subsequent tokens are 1-token KV-cache steps).
    vec = captured["h"][0, -1, :].float().cpu().numpy()
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec /= norm
    return vec


def collect_embeddings(records, cfg, args, use_lora: bool) -> np.ndarray:
    """
    Returns array of shape (N * 5, hidden_dim).
    Rows are ordered: vol_0/lung_tumor, vol_0/heart, ..., vol_0/lung,
                      vol_1/lung_tumor, ...
    """
    add_vila_to_path(args.vila_repo)
    init_pg_if_needed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_root = Path(cfg["data"]["processed_root"]).expanduser()
    retrieval_slices = cfg["contrastive"]["retrieval_slices"]

    tag = "fine-tuned" if use_lora else "baseline"
    print(f"\nLoading model ({tag})...")
    from llava.model.builder import load_pretrained_model  # type: ignore
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model,
        model_name="llava_llama",
        model_base=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if use_lora:
        from peft import PeftModel                        # type: ignore
        adapter = str(Path(args.lora).expanduser())
        print(f"  Applying LoRA: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    # Locate last decoder layer once — reused for every forward pass
    last_layer = _find_last_decoder_layer(model)
    print(f"  Hooking layer: {type(last_layer).__name__}")

    all_vecs = []
    for record in tqdm(records, desc=f"Embedding ({tag})"):
        rels = record["images"]
        if len(rels) > retrieval_slices:
            step = len(rels) / retrieval_slices
            rels = [rels[int(i * step)] for i in range(retrieval_slices)]
        images = [
            PILImage.open(processed_root / r).convert("RGB")
            for r in rels if (processed_root / r).exists()
        ]
        if not images:
            # pad with zeros so indices stay aligned
            all_vecs.extend([np.zeros(4096, dtype=np.float32)] * len(CLASS_ORDER))
            continue

        for cls in CLASS_ORDER:
            try:
                vec = extract_embedding(
                    model, tokenizer, image_processor,
                    QUERY_PER_CLASS[cls], images, device, last_layer)
            except Exception as e:
                print(f"  error ({record['volume_id']}/{cls}): {e}")
                vec = np.zeros(4096, dtype=np.float32)
            all_vecs.append(vec)

    # free GPU memory before next model load
    del model
    torch.cuda.empty_cache()

    return np.array(all_vecs, dtype=np.float32)  # (N*5, hidden_dim)


# ── UMAP + plotting ───────────────────────────────────────────────────────────

def run_umap(embeddings: np.ndarray, n_neighbors: int, min_dist: float):
    try:
        import umap                                       # type: ignore
    except ImportError:
        raise ImportError(
            "umap-learn not installed. Run: pip install umap-learn")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


def make_figure(umap_base: np.ndarray, umap_ft: np.ndarray,
                n_vols: int, output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    titles = ["Baseline (no LoRA)", "Fine-tuned (multi-class LoRA)"]
    datasets = [umap_base, umap_ft]

    for ax, coords, title in zip(axes, datasets, titles):
        for i, cls in enumerate(CLASS_ORDER):
            # rows for this class: every 5th row starting at index i
            idx = list(range(i, n_vols * len(CLASS_ORDER), len(CLASS_ORDER)))
            xy = coords[idx]
            ax.scatter(
                xy[:, 0], xy[:, 1],
                c=CLASS_COLOURS[cls],
                marker=CLASS_MARKERS[cls],
                s=18, alpha=0.70, linewidths=0,
                label=cls,
                zorder=3,
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    # single shared legend below both panels
    handles = [
        mpatches.Patch(color=CLASS_COLOURS[c], label=c)
        for c in CLASS_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Query-conditioned VLM embeddings (UMAP, cosine distance)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print(f"\nFigure saved to {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cache = Path(args.cache)

    # ── 1. Embeddings ─────────────────────────────────────────────────────────
    if cache.exists():
        print(f"Loading cached embeddings from {cache}")
        npz = np.load(cache)
        emb_base = npz["baseline"]
        emb_ft   = npz["finetuned"]
        n_vols   = emb_base.shape[0] // len(CLASS_ORDER)
        print(f"  baseline shape:  {emb_base.shape}")
        print(f"  finetuned shape: {emb_ft.shape}")
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        with open(Path(args.holdout).expanduser()) as f:
            records = [r for r in json.load(f) if r.get("type") == "retrieval"]
        rng = random.Random(args.seed)
        rng.shuffle(records)
        records = records[: args.n]
        n_vols = len(records)
        print(f"Embedding {n_vols} volumes × {len(CLASS_ORDER)} queries × "
              f"2 adapters = {n_vols * len(CLASS_ORDER) * 2} total calls\n")

        emb_base = collect_embeddings(records, cfg, args, use_lora=False)
        emb_ft   = collect_embeddings(records, cfg, args, use_lora=True)

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, baseline=emb_base, finetuned=emb_ft)
        print(f"Embeddings cached to {cache}")

    # ── 2. UMAP ───────────────────────────────────────────────────────────────
    print("\nRunning UMAP on baseline embeddings...")
    umap_base = run_umap(emb_base, args.umap_neighbors, args.umap_min_dist)
    print("Running UMAP on fine-tuned embeddings...")
    umap_ft   = run_umap(emb_ft,   args.umap_neighbors, args.umap_min_dist)

    # ── 3. Figure ─────────────────────────────────────────────────────────────
    make_figure(umap_base, umap_ft, n_vols, args.output)


if __name__ == "__main__":
    main()
