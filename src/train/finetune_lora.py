"""
finetune_lora.py
----------------
Joint LoRA fine-tuning of VILA-M3 8B on CT-RATE. One LoRA adapter is trained
with TWO objectives so a single fine-tuned model handles both paper tasks:

  1. Detection routing  — causal-LM loss teaching the model to emit the
     structured token <VISTA3D(lung tumor)> for thoracic localisation prompts.

  2. Case retrieval     — a CLIP-style symmetric InfoNCE loss that pulls each
     volume's image embedding (key CT slices) towards its radiology-report
     embedding, so the model's hidden space becomes retrieval-aligned.
     Without this, VILA-M3's embeddings give random retrieval (see paper §4).

Both objectives update the same adapter; each optimiser step accumulates
`gradient_accumulation_steps` detection micro-batches plus one contrastive
batch. Forward/backward are run sequentially so peak memory is one objective's
worth, not the sum.

Usage:
    python src/train/finetune_lora.py \
        --config configs/train_config.yaml \
        --data_path ~/icsdg_data/processed/ctrate_train.json \
        --vila_repo ./VLM-Radiology-Agent-Framework
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

# Image-side prompt for the retrieval embedding — MUST match eval_retrieval.py.
RETRIEVAL_IMAGE_PROMPT = "Describe this chest CT scan."


def add_vila_to_path(vila_framework: str) -> None:
    root = Path(vila_framework).expanduser().resolve()
    for p in [root, root / "m3", root / "thirdparty" / "VILA"]:
        p = str(p)
        if p not in sys.path:
            sys.path.insert(0, p)


def init_process_group_if_needed() -> None:
    """VILA's forward() calls calculate_loss_weight() -> dist.all_reduce(),
    which requires an initialised process group even for single-GPU runs."""
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def report_from_record(rec: dict) -> str:
    """Clean radiology-report text for a detection record (the contrastive
    text side). Uses the explicit field if present, else strips the routing
    lead-in from the gpt response."""
    if rec.get("report_text"):
        return rec["report_text"]
    val = rec["conversations"][1]["value"]
    marker = "<VISTA3D(lung tumor)>\n"
    return val.split(marker, 1)[1] if marker in val else val


# ── Datasets ──────────────────────────────────────────────────────────────────

class DetectionDataset(Dataset):
    """Detection-routing records -> causal-LM training samples."""

    def __init__(self, records, tokenizer, image_processor, processed_root,
                 max_slices, max_length=2048, model_dtype=torch.bfloat16):
        from llava.constants import IGNORE_INDEX
        self.records = [r for r in records if r.get("type") == "detection"]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.processed_root = Path(processed_root).expanduser()
        self.max_slices = max_slices
        self.max_length = max_length
        self.model_dtype = model_dtype
        self.ignore_index = IGNORE_INDEX

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        from llava.mm_utils import tokenizer_image_token
        from llava.conversation import conv_templates

        rec = self.records[idx]
        images = []
        for rel in rec["images"][: self.max_slices]:
            p = self.processed_root / rel
            if p.exists():
                images.append(PILImage.open(p).convert("RGB"))

        human = rec["conversations"][0]["value"].replace("<image>\n", "")
        human = "<image>\n" * len(images) + human
        gpt = rec["conversations"][1]["value"]

        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], human)
        conv.append_message(conv.roles[1], gpt)
        full = conv.get_prompt()

        # Empty-string (not None) assistant turn -> get_prompt() ends exactly
        # at the assistant header, giving a clean prefix for loss masking.
        conv_p = conv_templates["llama_3"].copy()
        conv_p.append_message(conv_p.roles[0], human)
        conv_p.append_message(conv_p.roles[1], "")
        prefix = conv_p.get_prompt()

        full_ids = tokenizer_image_token(full, self.tokenizer, return_tensors="pt")
        prefix_ids = tokenizer_image_token(prefix, self.tokenizer, return_tensors="pt")
        if full_ids.shape[0] > self.max_length:
            full_ids = full_ids[: self.max_length]

        labels = full_ids.clone()
        labels[: min(prefix_ids.shape[0], full_ids.shape[0])] = self.ignore_index

        if images:
            img = self.image_processor.preprocess(
                images, return_tensors="pt")["pixel_values"]
            if isinstance(img, list):
                img = torch.stack(img)
            img = img.to(self.model_dtype)
        else:
            img = torch.zeros(0, 3, 224, 224, dtype=self.model_dtype)

        return {
            "input_ids": full_ids,
            "attention_mask": torch.ones_like(full_ids),
            "labels": labels,
            "images": img,
        }


class RetrievalDataset(Dataset):
    """Same volumes, viewed as (key slices, report text) contrastive pairs."""

    def __init__(self, records, image_processor, processed_root,
                 retrieval_slices, model_dtype=torch.bfloat16):
        self.records = [r for r in records if r.get("type") == "detection"]
        self.image_processor = image_processor
        self.processed_root = Path(processed_root).expanduser()
        self.retrieval_slices = retrieval_slices
        self.model_dtype = model_dtype

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image as PILImage

        rec = self.records[idx]
        # Evenly sample `retrieval_slices` of the available key slices.
        rels = rec["images"]
        if len(rels) > self.retrieval_slices:
            step = len(rels) / self.retrieval_slices
            rels = [rels[int(i * step)] for i in range(self.retrieval_slices)]

        images = [PILImage.open(self.processed_root / r).convert("RGB")
                  for r in rels if (self.processed_root / r).exists()]
        img = self.image_processor.preprocess(
            images, return_tensors="pt")["pixel_values"]
        if isinstance(img, list):
            img = torch.stack(img)
        return {"images": img.to(self.model_dtype),
                "report_text": report_from_record(rec)}


# ── LoRA ──────────────────────────────────────────────────────────────────────

def build_lora_model(model, lora_cfg):
    """LoRA scoped to the language-model attention (vision tower stays frozen)."""
    suffixes = tuple(lora_cfg["target_modules"])
    targets = [n for n, _ in model.named_modules()
               if "llm" in n.split(".") and n.endswith(suffixes)]
    if not targets:
        raise RuntimeError("No LoRA targets found under model.llm")
    cfg = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"], target_modules=targets,
        bias=lora_cfg["bias"], task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ── Objectives ────────────────────────────────────────────────────────────────

def detection_loss(model, sample, device):
    """Causal-LM loss for one detection-routing sample."""
    out = model(
        input_ids=sample["input_ids"].unsqueeze(0).to(device),
        attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
        labels=sample["labels"].unsqueeze(0).to(device),
        images=sample["images"].to(device),
        return_dict=True,
    )
    return out.loss


def _embed(model, tokenizer, text, images, device):
    """Differentiable last-token embedding (image side or text side)."""
    from llava.mm_utils import tokenizer_image_token

    if images is not None:
        prompt = "<image>\n" * images.shape[0] + text
        input_ids = tokenizer_image_token(
            prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
        image_arg = images.to(device)
    else:
        input_ids = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=512).input_ids.to(device)
        image_arg = None
    out = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        images=image_arg,
        labels=input_ids,                       # VILA needs labels (loss unused)
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states[-1][0, -1, :]      # last-token hidden state


def contrastive_loss(model, tokenizer, batch, temperature, device):
    """CLIP-style symmetric InfoNCE over a batch of (slices, report) pairs."""
    img_embs, txt_embs = [], []
    for vol in batch:
        img_embs.append(_embed(model, tokenizer, RETRIEVAL_IMAGE_PROMPT,
                               vol["images"], device))
        txt_embs.append(_embed(model, tokenizer, vol["report_text"], None, device))

    img = F.normalize(torch.stack(img_embs).float(), dim=-1)
    txt = F.normalize(torch.stack(txt_embs).float(), dim=-1)
    logits = img @ txt.t() / temperature
    target = torch.arange(len(batch), device=device)
    return 0.5 * (F.cross_entropy(logits, target)
                  + F.cross_entropy(logits.t(), target))


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--data_path", default="~/icsdg_data/processed/ctrate_train.json")
    p.add_argument("--vila_repo", default="./VLM-Radiology-Agent-Framework")
    return p.parse_args()


def main():
    args = parse_args()
    add_vila_to_path(args.vila_repo)
    init_process_group_if_needed()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["train"]["seed"])

    from llava.model.builder import load_pretrained_model

    print(f"Loading VILA-M3: {cfg['model']['name']}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=cfg["model"]["name"], model_name="llava_llama",
        model_base=None, device_map="auto", torch_dtype=torch.bfloat16)
    # VILA's builder hardcodes the vision tower / projector to float16 while
    # the LLM ends up bf16 — unify everything to bf16 to avoid dtype clashes.
    model = model.to(torch.bfloat16)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # VILA's repack_multimodal_data (training path) reads llm.pad_token_id as a
    # direct attribute — it normally only lives on llm.config.
    model.llm.pad_token_id = tokenizer.pad_token_id

    model = build_lora_model(model, cfg["lora"])
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    device = "cuda"

    with open(Path(args.data_path).expanduser()) as f:
        records = json.load(f)

    tcfg, ccfg = cfg["train"], cfg["contrastive"]
    det_ds = DetectionDataset(records, tokenizer, image_processor,
                              cfg["data"]["processed_root"], cfg["data"]["max_slices"])
    ret_ds = RetrievalDataset(records, image_processor,
                              cfg["data"]["processed_root"], ccfg["retrieval_slices"])
    print(f"Detection samples: {len(det_ds)} | Retrieval volumes: {len(ret_ds)}")

    nw = tcfg["dataloader_num_workers"]
    det_loader = DataLoader(det_ds, batch_size=1, shuffle=True,
                            num_workers=nw, collate_fn=lambda b: b[0])
    ret_loader = DataLoader(ret_ds, batch_size=ccfg["batch_size"], shuffle=True,
                            num_workers=nw, collate_fn=list, drop_last=True)

    def ret_stream():
        while True:
            for b in ret_loader:
                yield b
    ret_iter = ret_stream()

    grad_accum = tcfg["gradient_accumulation_steps"]
    steps_per_epoch = len(det_loader) // grad_accum
    total_steps = steps_per_epoch * tcfg["num_epochs"]
    warmup = int(total_steps * tcfg["warmup_ratio"])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["learning_rate"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"Optimiser steps: {total_steps} ({steps_per_epoch}/epoch), warmup {warmup}")

    global_step = 0
    for epoch in range(tcfg["num_epochs"]):
        det_iter = iter(det_loader)
        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch + 1}")
        for _ in pbar:
            optimizer.zero_grad()

            det_total = 0.0
            for _ in range(grad_accum):
                loss = detection_loss(model, next(det_iter), device) / grad_accum
                loss.backward()
                det_total += loss.item()

            closs = contrastive_loss(model, tokenizer, next(ret_iter),
                                     ccfg["temperature"], device) * ccfg["loss_weight"]
            closs.backward()

            torch.nn.utils.clip_grad_norm_(params, tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % tcfg["logging_steps"] == 0 or global_step == 1:
                pbar.set_postfix(det_loss=f"{det_total:.3f}",
                                 con_loss=f"{closs.item():.3f}")

        ckpt = Path(tcfg["output_dir"]) / f"epoch_{epoch + 1}"
        model.save_pretrained(str(ckpt))
        print(f"  checkpoint saved -> {ckpt}")

    final = Path(tcfg["output_dir"]) / "lora_adapter_final"
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"\nLoRA adapter saved to {final}")


if __name__ == "__main__":
    main()
