"""
eval_retrieval.py
-----------------
Text-to-image case retrieval on the CT-RATE hold-out split.

VILA-M3 is a generative VLM, not a dual-encoder. We obtain retrieval
embeddings from its last-layer hidden states (mean-pooled):

  * Index side  : key CT slices + a neutral prompt  -> image-conditioned vector
  * Query side  : the radiology report text only    -> text vector

Both vectors live in the same hidden space; retrieval is cosine similarity
(FAISS inner product on L2-normalised vectors). Recall@K measures whether a
volume's own report retrieves its own images within the top-K.

The fine-tuned model is expected to beat the pre-fine-tuned baseline because
the CT-RATE LoRA aligns the hidden space to thoracic clinical language.

Usage:
    # Baseline (no fine-tuning)
    python src/eval/eval_retrieval.py \
        --config configs/train_config.yaml \
        --holdout_json ~/icsdg_data/processed/ctrate_holdout.json \
        --output_json results/baseline_retrieval.json

    # Fine-tuned model
    python src/eval/eval_retrieval.py \
        --config configs/train_config.yaml \
        --holdout_json ~/icsdg_data/processed/ctrate_holdout.json \
        --lora_adapter ./checkpoints/lora_adapter_final \
        --output_json results/finetuned_retrieval.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
import yaml
from tqdm import tqdm
from PIL import Image


def add_vila_to_path(vila_framework: str) -> None:
    root = Path(vila_framework).expanduser().resolve()
    for p in [root, root / "m3", root / "thirdparty" / "VILA"]:
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)


def init_process_group_if_needed() -> None:
    """VILA's forward() calls calculate_loss_weight() -> dist.all_reduce(),
    which requires an initialised process group even for single-GPU runs.
    A trivial 1-rank group makes all_reduce a no-op."""
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


# ── Embedding extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def embed(model, tokenizer, image_processor, text, images, device):
    """
    Mean-pooled, L2-normalised last-hidden-state embedding.
    `images` is a list of PIL images (image side) or None (text side).
    """
    from llava.mm_utils import tokenizer_image_token

    if images:
        prompt = "<image>\n" * len(images) + text
        input_ids = tokenizer_image_token(
            prompt, tokenizer, return_tensors="pt"
        ).unsqueeze(0).to(device)
        pixel = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        if isinstance(pixel, list):
            pixel = torch.stack(pixel)
        pixel = pixel.to(device=device, dtype=model.dtype)
        image_arg = pixel
    else:
        input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=512).input_ids.to(device)
        image_arg = None

    # VILA's forward needs an explicit attention_mask (it calls
    # attention_mask.sum(-1) and won't synthesise one), and it multiplies
    # outputs.loss by a weight so labels must be passed too. Both the loss and
    # the mask values are otherwise unused here.
    attention_mask = torch.ones_like(input_ids)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=image_arg,
        labels=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]          # (1, seq_len, hidden_dim)
    emb = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


# ── Recall@K ──────────────────────────────────────────────────────────────────

def compute_recall_at_k(query_emb, index_emb, query_ids, index_ids, k_values):
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


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--holdout_json", default="~/icsdg_data/processed/ctrate_holdout.json")
    p.add_argument("--lora_adapter", default=None,
                   help="Path to a saved LoRA adapter; omit for the baseline")
    p.add_argument("--output_json", default="results/retrieval_results.json")
    p.add_argument("--vila_repo", default="./VLM-Radiology-Agent-Framework")
    return p.parse_args()


def main():
    args = parse_args()
    add_vila_to_path(args.vila_repo)
    init_process_group_if_needed()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_root = Path(cfg["data"]["processed_root"]).expanduser()
    k_values = cfg["eval"]["retrieval_k"]
    max_slices = cfg["data"]["max_slices"]
    base_name = cfg["model"]["name"]

    from llava.model.builder import load_pretrained_model  # noqa

    print(f"Loading base model: {base_name}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=base_name,
        model_name="llava_llama",
        model_base=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if args.lora_adapter:
        from peft import PeftModel
        print(f"Applying LoRA adapter: {args.lora_adapter}")
        model = PeftModel.from_pretrained(model, str(Path(args.lora_adapter).expanduser()))
    model.eval()

    with open(Path(args.holdout_json).expanduser()) as f:
        records = [r for r in json.load(f) if r.get("type") == "retrieval"]
    print(f"Hold-out retrieval records: {len(records)}")

    index_emb, query_emb, ids = [], [], []
    for record in tqdm(records, desc="Embedding"):
        images = [
            Image.open(processed_root / rel).convert("RGB")
            for rel in record["images"][:max_slices]
            if (processed_root / rel).exists()
        ]
        query_text = record.get("query_text", "")
        if not images or not query_text:
            continue

        index_emb.append(embed(model, tokenizer, image_processor,
                               "Describe this chest CT scan.", images, device))
        query_emb.append(embed(model, tokenizer, image_processor,
                               query_text, None, device))
        ids.append(record["volume_id"])

    index_emb = np.stack(index_emb)
    query_emb = np.stack(query_emb)
    print(f"Embedded {len(ids)} volumes (dim {index_emb.shape[1]})")

    recall = compute_recall_at_k(query_emb, index_emb, ids, ids, k_values)
    print("\n── Retrieval Results ──────────────────────")
    for metric, value in recall.items():
        print(f"  {metric}: {value:.4f}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": base_name,
            "lora_adapter": args.lora_adapter,
            "n_queries": len(ids),
            "metrics": recall,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
