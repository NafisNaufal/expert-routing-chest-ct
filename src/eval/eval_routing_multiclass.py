"""
eval_routing_multiclass.py
--------------------------
Multi-class routing evaluation. Tests whether the fine-tuned VLM correctly
emits the *intended* routing token in response to class-specific queries.

For each held-out CT volume we run five separate inference calls, one per
target routing class (lung tumor, heart, liver, aorta, lung). The model
should emit a structured routing token whose class matches the query
intent. We report:

  format precision (per class): fraction of queries where the model emits
    any <VISTA3D(...)> token regardless of class.
  class precision (per class): fraction of queries where the emitted class
    exactly matches the target class for that query.

This is the headline experiment validating the framework's "multi-class
expert routing" claim. A high class precision across all five classes
demonstrates that the VLM has learned query-conditioned routing -- the
same input image yields different routing tokens depending on what is
asked.

Usage:
    python src/eval/eval_routing_multiclass.py \\
        --config configs/train_config_multiclass.yaml \\
        --holdout_json ~/icsdg_data/processed_multiclass/ctrate_holdout.json \\
        --lora_adapter ./checkpoints_multiclass/lora_adapter_final \\
        --output_json results/multiclass_routing.json
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image as PILImage
from tqdm import tqdm


# ── Routing classes and the queries that should elicit each ───────────────────
# Each class lists one canonical query template + the expected class string
# that a correct routing token must contain.
QUERY_PER_CLASS = {
    "lung tumor": "Identify and localise pulmonary nodules in this chest CT scan.",
    "heart":      "Identify and localise the cardiac structures in this CT.",
    "liver":      "Localise the liver in this CT scan.",
    "aorta":      "Identify the aortic arch in this scan.",
    "lung":       "Show the lung parenchyma boundaries.",
}

VISTA3D_STRICT_PATTERN     = re.compile(r"<VISTA3D\(([^)]+)\)>")
VISTA3D_PERMISSIVE_PATTERN = re.compile(
    r"<VISTA3D\([^\n>]*?(?:\)>|\)|\.|\n|$)", re.MULTILINE
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/train_config_multiclass.yaml")
    p.add_argument("--holdout_json",
                   default="~/icsdg_data/processed_multiclass/ctrate_holdout.json")
    p.add_argument("--vila_repo",    default="./VLM-Radiology-Agent-Framework")
    p.add_argument("--model_path",   default="MONAI/Llama3-VILA-M3-8B")
    p.add_argument("--lora_adapter",
                   default="./checkpoints_multiclass/lora_adapter_final")
    p.add_argument("--max_volumes",  type=int, default=200,
                   help="Number of holdout volumes to evaluate (each gets one "
                        "query per class, so total inference calls = "
                        "max_volumes * 5).")
    p.add_argument("--output_json",  default="results/multiclass_routing.json")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def init_process_group_if_needed() -> None:
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def add_vila_to_path(vila_framework: str) -> None:
    root = Path(vila_framework).expanduser().resolve()
    for p in [root, root / "m3", root / "m3" / "demo",
              root / "thirdparty" / "VILA"]:
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)


@torch.no_grad()
def generate_response(model, tokenizer, image_processor, query: str,
                      images, device) -> str:
    """Run a single inference call with the given query and images."""
    from llava.mm_utils import (tokenizer_image_token, process_images,  # type: ignore
                                KeywordsStoppingCriteria)
    from llava.conversation import conv_templates, SeparatorStyle  # type: ignore
    from llava.constants import IMAGE_TOKEN_INDEX  # type: ignore

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
    images_tensor = process_images(images, image_processor, model.config).to(
        device=device, dtype=next(model.parameters()).dtype)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    output_ids = model.generate(
        input_ids,
        images=[images_tensor],
        do_sample=False,
        max_new_tokens=128,
        min_new_tokens=2,
        use_cache=True,
        stopping_criteria=[stopping],
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def parse_routing_class(response: str) -> tuple[bool, str | None]:
    """
    Returns (has_format, emitted_class).
        has_format    : True if any structured or near-structured routing token
                        is present.
        emitted_class : the class string in the first strict match (lowercased,
                        stripped), or None if no strict match was found.
    """
    strict = VISTA3D_STRICT_PATTERN.findall(response)
    if strict:
        return True, strict[0].strip().lower()
    if VISTA3D_PERMISSIVE_PATTERN.search(response):
        # Format intent present but class not cleanly parseable -- count as
        # format-positive, class-unknown.
        return True, None
    return False, None


def main():
    args = parse_args()
    init_process_group_if_needed()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    add_vila_to_path(args.vila_repo)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed_root = Path(cfg["data"]["processed_root"]).expanduser()
    retrieval_slices = cfg["contrastive"]["retrieval_slices"]

    print(f"Loading base model: {args.model_path}")
    from llava.model.builder import load_pretrained_model  # type: ignore
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_name="llava_llama",
        model_base=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if args.lora_adapter:
        from peft import PeftModel  # type: ignore
        adapter = str(Path(args.lora_adapter).expanduser())
        print(f"Applying LoRA adapter: {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    # Pull the holdout retrieval records (they have image paths and IDs) and
    # randomly subsample to keep total inference budget tractable.
    with open(Path(args.holdout_json).expanduser()) as f:
        records = [r for r in json.load(f) if r.get("type") == "retrieval"]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    records = records[: args.max_volumes]
    print(f"Evaluating {len(records)} volumes "
          f"× {len(QUERY_PER_CLASS)} queries each "
          f"= {len(records) * len(QUERY_PER_CLASS)} total calls\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    # results[target_class] = list of dicts {volume_id, response, emitted_class,
    #                                        has_format, class_match}
    results = {cls: [] for cls in QUERY_PER_CLASS}

    for record in tqdm(records, desc="Volumes"):
        rels = record["images"]
        if len(rels) > retrieval_slices:
            step = len(rels) / retrieval_slices
            rels = [rels[int(i * step)] for i in range(retrieval_slices)]
        images = [
            PILImage.open(processed_root / r).convert("RGB")
            for r in rels if (processed_root / r).exists()
        ]
        if not images:
            continue

        for target_class, query in QUERY_PER_CLASS.items():
            try:
                response = generate_response(
                    model, tokenizer, image_processor, query, images, device)
            except Exception as e:
                print(f"  inference error ({record['volume_id']}/{target_class}): {e}")
                continue

            has_format, emitted = parse_routing_class(response)
            class_match = (emitted == target_class)
            results[target_class].append({
                "volume_id":     record["volume_id"],
                "response":      response,
                "emitted_class": emitted,
                "has_format":    has_format,
                "class_match":   class_match,
            })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    per_class_summary = {}
    print("\n── Multi-class Routing Results ───────────────────────────────")
    print(f"{'class':<14} {'format':>8} {'class':>8}   n")
    print("-" * 50)
    for cls, rows in results.items():
        if not rows:
            continue
        n = len(rows)
        fmt_rate = sum(r["has_format"]  for r in rows) / n
        cls_rate = sum(r["class_match"] for r in rows) / n
        per_class_summary[cls] = {
            "n": n,
            "format_precision": round(fmt_rate, 4),
            "class_precision": round(cls_rate, 4),
        }
        print(f"{cls:<14} {fmt_rate:>7.2%} {cls_rate:>7.2%}  {n}")

    macro_format = sum(s["format_precision"] for s in per_class_summary.values()) \
        / max(len(per_class_summary), 1)
    macro_class  = sum(s["class_precision"]  for s in per_class_summary.values()) \
        / max(len(per_class_summary), 1)
    print("-" * 50)
    print(f"{'MACRO':<14} {macro_format:>7.2%} {macro_class:>7.2%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model":          args.model_path,
            "lora_adapter":   args.lora_adapter,
            "n_volumes":      len(records),
            "per_class":      per_class_summary,
            "macro": {
                "format_precision": round(macro_format, 4),
                "class_precision":  round(macro_class,  4),
            },
            "raw":            results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
