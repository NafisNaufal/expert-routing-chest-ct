"""
finetune_lora.py
----------------
LoRA fine-tuning of VILA-M3 8B on CT-RATE detection-routing instructions.

VILA-M3 is a LLaVA-architecture model (model_type "llava_llama"). It is NOT a
standard HuggingFace AutoModelForCausalLM — it must be loaded via
llava.model.builder.load_pretrained_model, and its forward() accepts `images`
directly and runs the multimodal token-merge internally.

Two things the previous version got wrong and this version fixes:
  1. Image tokens. The literal string "<image>" must be converted to the
     special IMAGE_TOKEN_INDEX (-200) via llava's tokenizer_image_token —
     a plain tokenizer turns it into ordinary text and images are never fed in.
  2. Prompt format. VILA-M3 expects the llama_3 conversation template, not a
     bare "USER:/ASSISTANT:" string.

Only detection-routing instructions are used for training. Retrieval is handled
at eval time by extracting hidden-state embeddings (see eval_retrieval.py).

Usage:
    python src/train/finetune_lora.py \
        --config configs/train_config.yaml \
        --data_path ~/icsdg_data/processed/ctrate_train.json \
        --vila_repo ./VLM-Radiology-Agent-Framework
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, TaskType


def add_vila_to_path(vila_framework: str) -> None:
    """Add the VILA-M3 framework + llava submodule to sys.path."""
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class CTRATEInstructionDataset(Dataset):
    """
    Detection-routing instruction records from ctrate_train.json. Each record
    has a list of key-slice image paths and a two-turn human->gpt conversation.
    """

    def __init__(self, data_path, tokenizer, image_processor, processed_root,
                 max_slices=16, max_length=2048, model_dtype=torch.bfloat16):
        from llava.constants import IGNORE_INDEX  # noqa

        with open(data_path) as f:
            records = json.load(f)
        # Train only on detection-routing instructions.
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

    def _build_prompts(self, human_value, gpt_value):
        """Return (full_prompt, prefix_prompt) using the llama_3 template."""
        from llava.conversation import conv_templates

        conv = conv_templates["llama_3"].copy()
        conv.append_message(conv.roles[0], human_value)
        conv.append_message(conv.roles[1], gpt_value)
        full_prompt = conv.get_prompt()

        conv_p = conv_templates["llama_3"].copy()
        conv_p.append_message(conv_p.roles[0], human_value)
        conv_p.append_message(conv_p.roles[1], None)
        prefix_prompt = conv_p.get_prompt()
        return full_prompt, prefix_prompt

    def __getitem__(self, idx):
        from PIL import Image as PILImage
        from llava.mm_utils import tokenizer_image_token

        record = self.records[idx]

        # Load slice images that actually exist on disk.
        images = []
        for rel_path in record["images"][: self.max_slices]:
            img_path = self.processed_root / rel_path
            if img_path.exists():
                images.append(PILImage.open(img_path).convert("RGB"))

        # Conversation turns. Re-sync the <image> token count to the number of
        # images actually loaded — VILA consumes exactly one image per token.
        human_value = record["conversations"][0]["value"].replace("<image>\n", "")
        human_value = "<image>\n" * len(images) + human_value
        gpt_value = record["conversations"][1]["value"]

        full_prompt, prefix_prompt = self._build_prompts(human_value, gpt_value)

        full_ids = tokenizer_image_token(full_prompt, self.tokenizer,
                                         return_tensors="pt")
        prefix_ids = tokenizer_image_token(prefix_prompt, self.tokenizer,
                                           return_tensors="pt")

        if full_ids.shape[0] > self.max_length:
            full_ids = full_ids[: self.max_length]

        # Mask the prompt prefix so loss is computed only on the gpt response.
        labels = full_ids.clone()
        prefix_len = min(prefix_ids.shape[0], full_ids.shape[0])
        labels[:prefix_len] = self.ignore_index

        attention_mask = torch.ones_like(full_ids)

        # Preprocess images -> (num_images, 3, H, W) in the model dtype.
        if images:
            image_tensor = self.image_processor.preprocess(
                images, return_tensors="pt"
            )["pixel_values"]
            if isinstance(image_tensor, list):
                image_tensor = torch.stack(image_tensor)
            image_tensor = image_tensor.to(self.model_dtype)
        else:
            image_tensor = torch.zeros(0, 3, 224, 224, dtype=self.model_dtype)

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": image_tensor,
        }


class MultimodalCollator:
    """Pads text fields and concatenates per-sample image tensors."""

    def __init__(self, pad_token_id, ignore_index):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, features):
        max_len = max(f["input_ids"].shape[0] for f in features)

        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n_pad = max_len - f["input_ids"].shape[0]
            input_ids.append(torch.cat([
                f["input_ids"],
                torch.full((n_pad,), self.pad_token_id, dtype=torch.long)]))
            attention_mask.append(torch.cat([
                f["attention_mask"], torch.zeros(n_pad, dtype=torch.long)]))
            labels.append(torch.cat([
                f["labels"],
                torch.full((n_pad,), self.ignore_index, dtype=torch.long)]))

        # VILA consumes images by counting IMAGE_TOKEN_INDEX across the batch,
        # so a single concatenated (total_images, 3, H, W) tensor is correct.
        images = torch.cat([f["images"] for f in features], dim=0)

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
            "images": images,
        }


# ── LoRA setup ────────────────────────────────────────────────────────────────

def build_lora_model(model, lora_cfg):
    """Attach a LoRA adapter scoped to the language-model attention only."""
    # Restrict targets to the LLM tower — exclude the vision tower, which also
    # has q_proj/v_proj modules that must stay frozen.
    suffixes = tuple(lora_cfg["target_modules"])
    target_names = [
        name for name, _ in model.named_modules()
        if ".llm." in name and name.endswith(suffixes)
    ]
    if not target_names:
        raise RuntimeError(
            "No LoRA target modules found under model.llm — check that the "
            "VILA-M3 checkpoint loaded correctly."
        )

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=target_names,
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--data_path", default="~/icsdg_data/processed/ctrate_train.json")
    p.add_argument("--vila_repo", default="./VLM-Radiology-Agent-Framework",
                   help="Path to the VLM-Radiology-Agent-Framework repo root")
    p.add_argument("--resume_from", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    add_vila_to_path(args.vila_repo)
    init_process_group_if_needed()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["train"]["seed"])

    from llava.model.builder import load_pretrained_model  # noqa
    from llava.constants import IGNORE_INDEX  # noqa

    model_name = cfg["model"]["name"]
    print(f"Loading VILA-M3: {model_name}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_name,
        model_name="llava_llama",
        model_base=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_lora_model(model, cfg["lora"])
    model.config.use_cache = False

    dataset = CTRATEInstructionDataset(
        data_path=str(Path(args.data_path).expanduser()),
        tokenizer=tokenizer,
        image_processor=image_processor,
        processed_root=cfg["data"]["processed_root"],
        max_slices=cfg["data"]["max_slices"],
    )
    print(f"Detection-routing training samples: {len(dataset)}")

    collator = MultimodalCollator(tokenizer.pad_token_id, IGNORE_INDEX)

    train_cfg = cfg["train"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg.get("save_total_limit", 1),
        logging_steps=train_cfg["logging_steps"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        remove_unused_columns=False,
        report_to="none",
        seed=train_cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("Starting training ...")
    trainer.train(resume_from_checkpoint=args.resume_from)

    adapter_path = Path(train_cfg["output_dir"]) / "lora_adapter_final"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nLoRA adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()
