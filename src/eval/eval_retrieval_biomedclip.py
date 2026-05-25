"""
eval_retrieval_biomedclip.py
----------------------------
Retrieval evaluation using BiomedCLIP as the dedicated cross-modal encoder
in our hybrid framework.

BiomedCLIP is a published medical vision-language model trained on PMC-15M
(approximately 15 million biomedical image-text pairs from PubMed Central
open-access articles). In our hybrid architecture, the VLM handles intent
classification and expert routing, while BiomedCLIP handles embedding-space
case retrieval — each component does what it is best at.

Image side  : each CT volume is represented by its key axial slices. Each
              slice is encoded with BiomedCLIP's image tower; the L2-normalised
              slice embeddings are averaged to produce a single volume-level
              representation.
Text side   : each radiology report is encoded with BiomedCLIP's text tower.

Retrieval is performed by FAISS over inner product on L2-normalised vectors
(equivalent to cosine similarity). Recall@K measures whether a volume's own
report retrieves its own image-side embedding within the top-K candidates.

Usage:
    python src/eval/eval_retrieval_biomedclip.py \\
        --config configs/train_config.yaml \\
        --output_json results/biomedclip_retrieval_10k.json \\
        --save_embeddings results/biomedclip_embeddings.npz
"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


# BiomedCLIP from the Microsoft / NIH BiomedCLIP release. The model card is
# `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`. It uses the
# `open_clip` checkpoint format, so we load it via open_clip rather than the
# standard transformers `AutoModel` interface — transformers does not yet
# support the SigLIP-style architecture this checkpoint uses.
MODEL_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--holdout_json",
                   default="~/icsdg_data/processed/ctrate_holdout.json")
    p.add_argument("--output_json",
                   default="results/biomedclip_retrieval.json")
    p.add_argument("--save_embeddings", default=None,
                   help="Path to .npz file for storing raw image and text "
                        "embeddings (used for UMAP visualisation).")
    return p.parse_args()


@torch.no_grad()
def encode_volume(model, preprocess, images, device):
    """
    Average-pool BiomedCLIP image embeddings across the key slices of one
    CT volume. Returns an L2-normalised numpy vector.
    """
    tensors = torch.stack([preprocess(img) for img in images]).to(device)
    features = model.encode_image(tensors)
    features = features / features.norm(dim=-1, keepdim=True)
    pooled = features.mean(dim=0)
    pooled = pooled / pooled.norm()
    return pooled.float().cpu().numpy()


@torch.no_grad()
def encode_report(model, tokenizer, text, device, max_tokens=256):
    """
    Encode a radiology report with BiomedCLIP's text tower. Returns an
    L2-normalised numpy vector.
    """
    tokens = tokenizer([text], context_length=max_tokens).to(device)
    features = model.encode_text(tokens)
    features = features / features.norm(dim=-1, keepdim=True)
    return features[0].float().cpu().numpy()


def compute_recall_at_k(query_emb, index_emb, query_ids, index_ids, k_values):
    """Standard text-to-image Recall@K under FAISS inner-product search."""
    dim = index_emb.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(index_emb.astype(np.float32))
    _, top_idx = faiss_index.search(query_emb.astype(np.float32), max(k_values))
    hits = {k: 0 for k in k_values}
    for q_idx, row in enumerate(top_idx):
        retrieved = [index_ids[i] for i in row]
        for k in k_values:
            if query_ids[q_idx] in retrieved[:k]:
                hits[k] += 1
    n = len(query_ids)
    return {f"Recall@{k}": round(hits[k] / n, 4) for k in k_values}


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_root = Path(cfg["data"]["processed_root"]).expanduser()
    k_values = cfg["eval"]["retrieval_k"]
    retrieval_slices = cfg["contrastive"]["retrieval_slices"]

    print(f"Loading BiomedCLIP from {MODEL_TAG}")
    import open_clip   # imported lazily so the rest of the file is parseable
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_TAG)
    tokenizer = open_clip.get_tokenizer(MODEL_TAG)
    model = model.to(device).eval()

    with open(Path(args.holdout_json).expanduser()) as f:
        records = [r for r in json.load(f) if r.get("type") == "retrieval"]
    print(f"Hold-out retrieval records: {len(records)}")

    image_emb, text_emb, ids = [], [], []
    skipped = 0
    for record in tqdm(records, desc="Embedding"):
        # Sample the same key slices as during VILA-M3 retrieval evaluation,
        # so the two pipelines are directly comparable.
        rels = record["images"]
        if len(rels) > retrieval_slices:
            step = len(rels) / retrieval_slices
            rels = [rels[int(i * step)] for i in range(retrieval_slices)]
        images = [
            Image.open(processed_root / r).convert("RGB")
            for r in rels if (processed_root / r).exists()
        ]
        query_text = record.get("query_text", "")
        if not images or not query_text:
            skipped += 1
            continue

        try:
            img_e = encode_volume(model, preprocess, images, device)
            txt_e = encode_report(model, tokenizer, query_text, device)
        except Exception as e:
            print(f"  skip {record.get('volume_id')}: {e}")
            skipped += 1
            continue

        image_emb.append(img_e)
        text_emb.append(txt_e)
        ids.append(record["volume_id"])

    image_emb = np.stack(image_emb)
    text_emb = np.stack(text_emb)
    print(f"Embedded {len(ids)} volumes (dim {image_emb.shape[1]});"
          f" skipped {skipped}")

    if args.save_embeddings:
        out = Path(args.save_embeddings).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out,
            image_emb=image_emb,
            text_emb=text_emb,
            ids=np.array(ids, dtype=object),
        )
        print(f"Embeddings saved to {out}")

    # Text-to-image retrieval: each report queries the image-embedding index.
    recall = compute_recall_at_k(text_emb, image_emb, ids, ids, k_values)
    print("\n── BiomedCLIP Retrieval Results ─────────────────")
    for metric, value in recall.items():
        print(f"  {metric}: {value:.4f}")

    # Also report matched-pair cosine similarity — directly comparable to the
    # VILA-M3 numbers from the earlier ablation.
    paired_sim = (image_emb * text_emb).sum(axis=1)
    print(f"  Matched-pair cosine: mean={paired_sim.mean():.4f} "
          f"std={paired_sim.std():.4f}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model":                MODEL_TAG,
            "n_queries":            len(ids),
            "retrieval_slices":     retrieval_slices,
            "metrics":              recall,
            "matched_pair_cosine":  {
                "mean": round(float(paired_sim.mean()), 4),
                "std":  round(float(paired_sim.std()),  4),
            },
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
