"""
eval_detection.py
-----------------
Zero-shot abnormality detection on LIDC-IDRI via segmentation.

Detection is performed by routing to VISTA3D (lung tumor class), which
produces a voxel-level segmentation mask. The mask is evaluated against
LIDC-IDRI consensus contours using Dice Similarity Coefficient (DSC) and IoU.

Three conditions:
  finetuned      — LoRA VILA-M3 routes to VISTA3D  (our method)
  baseline       — pre-trained VILA-M3 routes to VISTA3D  (no LoRA)
  direct_vista3d — VISTA3D called directly without VLM  (ablation)

The routing rate (% of queries where the model correctly emits
<VISTA3D(lung tumor)>) is also reported as a secondary contribution metric.

Usage:
    python src/eval/eval_detection.py \\
        --config configs/train_config.yaml \\
        --eval_json /data/processed/lidc_eval.json \\
        --vila_repo ../VLM-Radiology-Agent-Framework \\
        --condition finetuned \\
        --lora_adapter ./checkpoints/lora_adapter_final
"""

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml
from tqdm import tqdm


def init_process_group_if_needed() -> None:
    """VILA's forward() calls calculate_loss_weight() -> dist.all_reduce(),
    which requires an initialised process group even for single-GPU runs."""
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


# ── Segmentation metrics ──────────────────────────────────────────────────────

def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def load_mask(path: str) -> np.ndarray:
    return (nib.load(path).get_fdata() > 0).astype(np.uint8)


# ── VISTA3D invocation ────────────────────────────────────────────────────────

def call_vista3d(nii_path: str, output_dir: Path, vista3d_expert) -> np.ndarray | None:
    """Run VISTA3D with lung tumor prompt. Returns binary mask or None."""
    try:
        vista3d_expert.run(
            img_file=nii_path,
            input="<VISTA3D(lung tumor)>",
            output_dir=str(output_dir),
            prompt="Detect and localise pulmonary nodules.",
        )
        seg_path = output_dir / "segmentation.nii.gz"
        if seg_path.exists():
            data = nib.load(str(seg_path)).get_fdata()
            return (data == 23).astype(np.uint8)   # label 23 = lung tumor
    except Exception as e:
        print(f"  VISTA3D error: {e}")
    return None


def run_via_vlm(record, processed_root, output_dir, model, tokenizer, image_processor, vista3d, device):
    """
    Run VILA-M3. If the model emits <VISTA3D(lung tumor)>, intercept and
    call VISTA3D. Returns (mask_or_None, was_routed).
    """
    from PIL import Image as PILImage

    images = [
        PILImage.open(processed_root / rel).convert("RGB")
        for rel in record["images"]
        if (processed_root / rel).exists()
    ]
    if not images:
        return None, False

    from llava.mm_utils import tokenizer_image_token  # type: ignore
    from llava.conversation import conv_templates  # type: ignore

    prompt_text = (
        "<image>\n" * len(images)
        + "Identify and localise pulmonary nodules in this chest CT scan."
    )
    conv = conv_templates["llama_3"].copy()
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
    if isinstance(image_tensor, list):
        image_tensor = torch.stack(image_tensor)
    image_tensor = image_tensor.to(device=device, dtype=next(model.parameters()).dtype)

    with torch.no_grad():
        out = model.generate(input_ids, images=image_tensor, max_new_tokens=128)
    response = tokenizer.decode(out[0], skip_special_tokens=False)

    if "VISTA3D" in response and "lung tumor" in response.lower():
        mask = call_vista3d(record.get("nii_path", ""), output_dir, vista3d)
        return mask, True

    return None, False


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/train_config.yaml")
    p.add_argument("--eval_json",    default="/data/processed/lidc_eval.json")
    p.add_argument("--vila_repo",    default="../VLM-Radiology-Agent-Framework")
    p.add_argument("--model_path",   default="MONAI/Llama3-VILA-M3-8B")
    p.add_argument("--lora_adapter", default=None)
    p.add_argument("--condition",
                   choices=["finetuned", "baseline", "direct_vista3d"],
                   default="finetuned")
    p.add_argument("--output_json",  default=None)
    return p.parse_args()


def main():
    args = parse_args()
    init_process_group_if_needed()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    processed_root = Path(cfg["data"]["processed_root"])
    vila_repo = str(Path(args.vila_repo).expanduser())
    for p in [vila_repo, f"{vila_repo}/m3", f"{vila_repo}/m3/demo",
              f"{vila_repo}/thirdparty/VILA"]:
        if p not in sys.path:
            sys.path.insert(0, p)

    with open(args.eval_json) as f:
        eval_records = json.load(f)
    print(f"Records   : {len(eval_records)}")
    print(f"Condition : {args.condition}\n")

    from experts.expert_monai_vista3d import ExpertVista3D  # type: ignore
    vista3d = ExpertVista3D()

    model = tokenizer = image_processor = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.condition != "direct_vista3d":
        from llava.model.builder import load_pretrained_model  # type: ignore

        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=args.model_path,
            model_name="llava_llama",
            model_base=None,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        if args.condition == "finetuned" and args.lora_adapter:
            from peft import PeftModel  # type: ignore
            adapter = str(Path(args.lora_adapter).expanduser())
            model = PeftModel.from_pretrained(model, adapter)
            print(f"LoRA adapter: {adapter}")
        model.eval()

    all_dice, all_iou = [], []
    routed_count = 0

    for record in tqdm(eval_records, desc="Evaluating"):
        import tempfile

        gt_mask_path = record.get("gt_mask_path")
        if not gt_mask_path or not Path(gt_mask_path).exists():
            continue
        gt_mask = load_mask(gt_mask_path)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            if args.condition == "direct_vista3d":
                pred_mask = call_vista3d(record.get("nii_path", ""), tmp_path, vista3d)
                routed = pred_mask is not None
            else:
                pred_mask, routed = run_via_vlm(
                    record, processed_root, tmp_path, model,
                    tokenizer, image_processor, vista3d, device
                )

            if routed:
                routed_count += 1

            if pred_mask is None:
                pred_mask = np.zeros_like(gt_mask)
            elif pred_mask.shape != gt_mask.shape:
                from scipy.ndimage import zoom
                factors = tuple(g / p for g, p in zip(gt_mask.shape, pred_mask.shape))
                pred_mask = zoom(pred_mask, factors, order=0).astype(np.uint8)

        all_dice.append(dice_coefficient(pred_mask, gt_mask))
        all_iou.append(iou_score(pred_mask, gt_mask))

    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
    mean_iou  = float(np.mean(all_iou))  if all_iou  else 0.0
    routing_rate = routed_count / len(eval_records) if eval_records else 0.0

    print(f"\n── Detection Results ({args.condition}) ──────────────────")
    print(f"  Scans evaluated : {len(eval_records)}")
    print(f"  Routing rate    : {routing_rate:.2%}  ({routed_count}/{len(eval_records)})")
    print(f"  Mean Dice (DSC) : {mean_dice:.4f}")
    print(f"  Mean IoU        : {mean_iou:.4f}")

    output_path = Path(args.output_json or f"results/detection_{args.condition}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "condition":    args.condition,
            "model":        args.model_path,
            "lora_adapter": args.lora_adapter,
            "n_scans":      len(eval_records),
            "routing_rate": round(routing_rate, 4),
            "mean_dice":    round(mean_dice, 4),
            "mean_iou":     round(mean_iou,  4),
            "per_scan":     [{"dice": round(d, 4), "iou": round(i, 4)}
                             for d, i in zip(all_dice, all_iou)],
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
