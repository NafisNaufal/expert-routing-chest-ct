"""
eval_lits_liver.py
------------------
Validates MC-LoRA liver routing end-to-end on LiTS (Liver Tumour Segmentation
Challenge) scans. For each scan the script runs two conditions:

  direct_vista3d  — VISTA3D called directly with the liver class prompt.
                    This establishes the per-scan ceiling.
  mc_finetuned    — MC-LoRA VILA-M3 receives a liver query; if it routes to
                    <VISTA3D(liver)>, VISTA3D is called. This validates that
                    the liver routing class (100% precision without constraint)
                    translates into real segmentation performance.

Evaluation metric: Dice Similarity Coefficient vs. LiTS ground-truth liver
                   masks (binary: 1 = liver parenchyma + tumour).

LiTS data format expected:
  <lits_root>/
    volumes/       # NIfTI CT volumes  (volume-N.nii or volume-N.nii.gz)
    segmentations/ # NIfTI masks        (segmentation-N.nii or .nii.gz)
                   # label 1 = liver,  label 2 = liver tumour

The script binarises the LiTS mask as (label >= 1) to produce a liver mask.
VISTA3D's liver label in the 127-class vocabulary must be confirmed from the
model card — pass it via --vista3d_liver_label (default: 1).

Usage (run on HPC, GPU required):
    python src/eval/eval_lits_liver.py \\
        --lits_root   /data/lits \\
        --max_scans   30 \\
        --vila_repo   ./VLM-Radiology-Agent-Framework \\
        --lora_adapter ./checkpoints_multiclass/lora_adapter_final \\
        --output_json results/lits_liver.json

Download LiTS (Medical Segmentation Decathlon subset is sufficient):
    # Via HuggingFace:
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download; \\
               snapshot_download('ibrahim-hamamci/LiTS17', local_dir='/data/lits')"
    # Or from the original challenge:
    # https://competitions.codalab.org/competitions/17094
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

VISTA3D_STRICT_PATTERN = re.compile(r"<VISTA3D\(([^)]+)\)>")
VISTA3D_PERMISSIVE_PATTERN = re.compile(
    r"<VISTA3D\([^\n>]*?(?:\)>|\)|\.|\n|$)", re.MULTILINE
)
LIVER_QUERY = "Localise the liver in this CT scan."


def dice_coefficient(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    intersection = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter / union) if union else 1.0


def load_nii_mask(path: str, label_threshold: int = 1) -> np.ndarray:
    return (nib.load(path).get_fdata() >= label_threshold).astype(np.uint8)


def align_shape(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.shape == gt.shape:
        return pred
    from scipy.ndimage import zoom
    factors = tuple(g / p for g, p in zip(gt.shape, pred.shape))
    return zoom(pred, factors, order=0).astype(np.uint8)


def call_vista3d(nii_path: str, out_dir: Path, vista3d_expert,
                 vista3d_class: str, vista3d_label: int) -> np.ndarray | None:
    try:
        vista3d_expert.run(
            img_file=nii_path,
            input=f"<VISTA3D({vista3d_class})>",
            output_dir=str(out_dir),
            prompt=f"Segment the {vista3d_class}.",
        )
        seg_path = out_dir / "segmentation.nii.gz"
        if seg_path.exists():
            data = nib.load(str(seg_path)).get_fdata()
            return (data == vista3d_label).astype(np.uint8)
    except Exception as e:
        print(f"  VISTA3D error: {e}")
    return None


def init_process_group_if_needed():
    import torch.distributed as dist
    if dist.is_available() and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)


def add_vila_to_path(vila_repo: str):
    root = Path(vila_repo).expanduser().resolve()
    for p in [root, root / "m3", root / "m3" / "demo",
              root / "thirdparty" / "VILA"]:
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def sample_slices(volume_path: str, n_slices: int = 8) -> list:
    """Extract uniformly spaced axial slices from a CT volume for VLM input."""
    from PIL import Image as PILImage
    img   = nib.load(volume_path)
    data  = img.get_fdata()                    # (X, Y, Z)
    n_z   = data.shape[2]
    idxs  = np.linspace(0, n_z - 1, n_slices, dtype=int)
    slices = []
    for idx in idxs:
        sl = data[:, :, idx]
        # Window to soft-tissue range [-200, 300 HU] then normalise to 0–255
        sl  = np.clip(sl, -200, 300)
        sl  = ((sl + 200) / 500 * 255).astype(np.uint8)
        img_pil = PILImage.fromarray(sl).convert("RGB")
        slices.append(img_pil)
    return slices


@torch.no_grad()
def query_vlm(model, tokenizer, image_processor, query: str,
              images: list, device) -> str:
    from llava.mm_utils import (tokenizer_image_token, process_images,   # type: ignore
                                KeywordsStoppingCriteria)
    from llava.conversation import conv_templates, SeparatorStyle        # type: ignore
    from llava.constants import IMAGE_TOKEN_INDEX                        # type: ignore

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
        input_ids, images=[images_tensor], do_sample=False,
        max_new_tokens=128, min_new_tokens=2, use_cache=True,
        stopping_criteria=[stopping], pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


def collect_lits_pairs(lits_root: Path, max_scans: int) -> list[dict]:
    vol_dir = lits_root / "volumes"
    seg_dir = lits_root / "segmentations"
    pairs   = []
    for vol in sorted(vol_dir.glob("volume-*.nii*"))[:max_scans]:
        stem = vol.name.replace("volume-", "segmentation-").replace(".nii.gz", "").replace(".nii", "")
        seg  = next(seg_dir.glob(f"{stem}.nii*"), None)
        if seg:
            pairs.append({"vol": str(vol), "seg": str(seg)})
    return pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lits_root",          default="/data/lits")
    p.add_argument("--max_scans",          type=int, default=30)
    p.add_argument("--vila_repo",          default="./VLM-Radiology-Agent-Framework")
    p.add_argument("--model_path",         default="MONAI/Llama3-VILA-M3-8B")
    p.add_argument("--lora_adapter",       default="./checkpoints_multiclass/lora_adapter_final")
    p.add_argument("--vista3d_liver_label",type=int, default=1,
                   help="VISTA3D label ID for liver. "
                        "Verify from model card: "
                        "https://huggingface.co/MONAI/vista3d")
    p.add_argument("--n_slices",           type=int, default=8,
                   help="Axial slices sampled per volume for VLM input.")
    p.add_argument("--output_json",        default="results/lits_liver.json")
    return p.parse_args()


def main():
    args = parse_args()
    init_process_group_if_needed()
    add_vila_to_path(args.vila_repo)

    lits_root = Path(args.lits_root).expanduser()
    pairs = collect_lits_pairs(lits_root, args.max_scans)
    if not pairs:
        print(f"No LiTS volume/segmentation pairs found under {lits_root}.")
        print("Expected layout:  <lits_root>/volumes/volume-N.nii.gz")
        print("                  <lits_root>/segmentations/segmentation-N.nii.gz")
        return
    print(f"Found {len(pairs)} LiTS scan pairs.")

    from experts.expert_monai_vista3d import ExpertVista3D  # type: ignore
    vista3d = ExpertVista3D()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from llava.model.builder import load_pretrained_model              # type: ignore
    from peft import PeftModel                                          # type: ignore

    print(f"Loading base model: {args.model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_name="llava_llama",
        model_base=None,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    adapter = str(Path(args.lora_adapter).expanduser())
    print(f"Applying LoRA adapter: {adapter}")
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    direct_dice, direct_iou   = [], []
    vlm_dice,    vlm_iou      = [], []
    routed_count = 0
    per_scan_results = []

    for pair in tqdm(pairs, desc="LiTS scans"):
        gt_mask = load_nii_mask(pair["seg"], label_threshold=1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # ── Condition 1: direct VISTA3D ──────────────────────────────────
            pred_direct = call_vista3d(
                pair["vol"], tmp_path / "direct",
                vista3d, vista3d_class="liver",
                vista3d_label=args.vista3d_liver_label,
            )
            if pred_direct is None:
                pred_direct = np.zeros_like(gt_mask)
            else:
                pred_direct = align_shape(pred_direct, gt_mask)
            d_direct = dice_coefficient(pred_direct, gt_mask)
            i_direct = iou_score(pred_direct, gt_mask)
            direct_dice.append(d_direct)
            direct_iou.append(i_direct)

            # ── Condition 2: MC-LoRA VLM routing ────────────────────────────
            images  = sample_slices(pair["vol"], n_slices=args.n_slices)
            response = query_vlm(model, tokenizer, image_processor,
                                 LIVER_QUERY, images, device)

            strict  = VISTA3D_STRICT_PATTERN.findall(response)
            emitted = strict[0].strip().lower() if strict else None
            routed  = emitted == "liver"

            if routed:
                routed_count += 1
                pred_vlm = call_vista3d(
                    pair["vol"], tmp_path / "vlm",
                    vista3d, vista3d_class="liver",
                    vista3d_label=args.vista3d_liver_label,
                )
                if pred_vlm is None:
                    pred_vlm = np.zeros_like(gt_mask)
                else:
                    pred_vlm = align_shape(pred_vlm, gt_mask)
            else:
                pred_vlm = np.zeros_like(gt_mask)

            d_vlm = dice_coefficient(pred_vlm, gt_mask)
            i_vlm = iou_score(pred_vlm, gt_mask)
            vlm_dice.append(d_vlm)
            vlm_iou.append(i_vlm)

            per_scan_results.append({
                "vol": pair["vol"],
                "direct_dice": round(d_direct, 4),
                "direct_iou":  round(i_direct, 4),
                "vlm_routed":  routed,
                "emitted_class": emitted,
                "vlm_dice":    round(d_vlm, 4),
                "vlm_iou":     round(i_vlm, 4),
                "response":    response,
            })

    n = len(pairs)
    print(f"\n── LiTS Liver Results ({n} scans) ──────────────────────────────────")
    print(f"  Direct VISTA3D  :  Dice {np.mean(direct_dice):.4f}  IoU {np.mean(direct_iou):.4f}")
    print(f"  MC-LoRA routing :  Dice {np.mean(vlm_dice):.4f}  IoU {np.mean(vlm_iou):.4f}")
    print(f"  Liver routing rate (unconstrained): {routed_count}/{n} = "
          f"{100*routed_count/n:.1f}%  (expect ~100% from training)")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "n_scans":             n,
            "vista3d_liver_label": args.vista3d_liver_label,
            "routing_rate":        round(routed_count / n, 4),
            "direct": {
                "mean_dice": round(float(np.mean(direct_dice)), 4),
                "mean_iou":  round(float(np.mean(direct_iou)),  4),
                "std_dice":  round(float(np.std(direct_dice, ddof=1)), 4),
            },
            "mc_lora": {
                "mean_dice": round(float(np.mean(vlm_dice)), 4),
                "mean_iou":  round(float(np.mean(vlm_iou)),  4),
                "std_dice":  round(float(np.std(vlm_dice, ddof=1)), 4),
            },
            "per_scan": per_scan_results,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
