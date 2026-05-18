"""
prepare_lidc.py
---------------
Converts LIDC-IDRI scans into NIfTI volumes + consensus segmentation masks and
extracts key axial slices, ready for zero-shot detection evaluation.

The volume and the consensus mask are both built from pylidc's `to_volume()`
voxel grid, so they are guaranteed to share the same shape and orientation —
VISTA3D runs on the NIfTI volume and its output mask can be compared to the
consensus mask directly (Dice / IoU) without resampling guesswork.

Reads:  <lidc_root>/nodule_annotations.json   (from download_lidc.py)
Writes: <lidc_root>/nifti/<volume_id>.nii.gz   (full-res CT for VISTA3D)
        <output_root>/masks/<volume_id>_gt.nii.gz   (consensus mask)
        <output_root>/slices/<volume_id>_NN.png     (key axial slices)
        <output_root>/lidc_eval.json                (evaluation records)

A voxel is positive in the consensus mask if >=3 of 4 radiologists contoured it.

Usage:
    python src/data/prepare_lidc.py \
        --lidc_root ~/icsdg_data/lidc_idri \
        --output_root ~/icsdg_data/processed \
        --max_scans 150
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def _numpy_compat():
    """pylidc references numpy aliases removed in numpy >= 1.24."""
    for alias, builtin in [("int", int), ("bool", bool),
                           ("float", float), ("complex", complex)]:
        if not hasattr(np, alias):
            setattr(np, alias, builtin)


# ── Slice utilities ───────────────────────────────────────────────────────────

def window_ct(vol: np.ndarray, wl: float = -600, ww: float = 1500) -> np.ndarray:
    lo, hi = wl - ww / 2, wl + ww / 2
    vol = np.clip(vol, lo, hi)
    return ((vol - lo) / (hi - lo) * 255).astype(np.uint8)


def sample_key_slices(vol_zhw: np.ndarray, n: int):
    z = vol_zhw.shape[0]
    margin = max(1, int(z * 0.10))
    indices = np.linspace(margin, z - margin - 1, n, dtype=int)
    return [vol_zhw[i] for i in indices], indices.tolist()


def save_slices(slices, slices_dir: Path, volume_id: str, size: int) -> list:
    slices_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, sl in enumerate(slices):
        img = Image.fromarray(sl).convert("RGB").resize((size, size), Image.BILINEAR)
        img.save(slices_dir / f"{volume_id}_{i:02d}.png")
        paths.append(f"slices/{volume_id}_{i:02d}.png")
    return paths


# ── Consensus mask (same voxel grid as scan.to_volume()) ──────────────────────

def build_consensus_mask(scan, vol_shape: tuple) -> np.ndarray:
    """>=3-of-4 radiologist agreement, in the to_volume() voxel grid."""
    mask = np.zeros(vol_shape, dtype=np.uint8)
    for cluster in scan.cluster_annotations():
        if len(cluster) < 3:
            continue
        bboxes = [ann.bbox() for ann in cluster]
        i0 = min(bb[0].start for bb in bboxes); i1 = max(bb[0].stop for bb in bboxes)
        j0 = min(bb[1].start for bb in bboxes); j1 = max(bb[1].stop for bb in bboxes)
        k0 = min(bb[2].start for bb in bboxes); k1 = max(bb[2].stop for bb in bboxes)

        vote = np.zeros((i1 - i0, j1 - j0, k1 - k0), dtype=np.uint8)
        for ann in cluster:
            bb = ann.bbox()
            bm = ann.boolean_mask().astype(np.uint8)
            vote[bb[0].start - i0: bb[0].start - i0 + bm.shape[0],
                 bb[1].start - j0: bb[1].start - j0 + bm.shape[1],
                 bb[2].start - k0: bb[2].start - k0 + bm.shape[2]] += bm
        mask[i0:i1, j0:j1, k0:k1] |= (vote >= 3).astype(np.uint8)
    return mask


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lidc_root", default="~/icsdg_data/lidc_idri")
    p.add_argument("--output_root", default="~/icsdg_data/processed")
    p.add_argument("--max_scans", type=int, default=150)
    p.add_argument("--max_slices", type=int, default=16)
    p.add_argument("--slice_size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()
    _numpy_compat()
    import pylidc as pl
    import nibabel as nib

    lidc_root = Path(args.lidc_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    slices_dir = output_root / "slices"
    nifti_dir = lidc_root / "nifti"
    masks_dir = output_root / "masks"
    for d in (slices_dir, nifti_dir, masks_dir):
        d.mkdir(parents=True, exist_ok=True)

    ann_path = lidc_root / "nodule_annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Annotation index not found at {ann_path}. Run download_lidc.py first.")
    with open(ann_path) as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotated scans from LIDC-IDRI")

    pl_scans = {str(s.id): s for s in pl.query(pl.Scan).all()}

    records = []
    for scan_meta in tqdm(annotations, desc="Processing LIDC-IDRI"):
        if len(records) >= args.max_scans:
            break

        patient_id = scan_meta["patient_id"]
        scan_id = scan_meta["scan_id"]
        volume_id = f"lidc_{patient_id}_{scan_id}"

        scan = pl_scans.get(scan_id)
        if scan is None:
            continue

        try:
            vol = scan.to_volume()                       # HU, (i, j, k)
            gt_mask = build_consensus_mask(scan, vol.shape)
            if gt_mask.sum() == 0:
                continue                                 # no >=3-consensus nodule

            spacing = float(scan.pixel_spacing or 1.0)
            thickness = float(scan.slice_thickness or 1.0)
            affine = np.diag([spacing, spacing, thickness, 1.0])

            nii_path = nifti_dir / f"{volume_id}.nii.gz"
            mask_path = masks_dir / f"{volume_id}_gt.nii.gz"
            nib.save(nib.Nifti1Image(vol.astype(np.int16), affine), str(nii_path))
            nib.save(nib.Nifti1Image(gt_mask, affine), str(mask_path))

            vol_zhw = window_ct(vol).transpose(2, 0, 1)  # (k, i, j) axial stack
            slices, z_idx = sample_key_slices(vol_zhw, args.max_slices)
            slice_paths = save_slices(slices, slices_dir, volume_id, args.slice_size)
        except Exception as e:
            print(f"  Warning: skipping {patient_id}: {e}")
            continue

        records.append({
            "id": volume_id,
            "volume_id": volume_id,
            "patient_id": patient_id,
            "scan_id": scan_id,
            "nii_path": str(nii_path),            # full-res NIfTI for VISTA3D
            "gt_mask_path": str(mask_path),       # consensus segmentation mask
            "images": slice_paths,
            "sampled_z_indices": z_idx,
            "volume_shape": list(vol.shape),
            "conversations": [{
                "from": "human",
                "value": ("<image>\n" * len(slice_paths)
                          + "Identify and localise pulmonary nodules in the "
                            "provided chest CT scan."),
            }],
        })

    out_path = output_root / "lidc_eval.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nPreprocessing complete.")
    print(f"  Evaluation records : {len(records)}")
    print(f"  NIfTI volumes      : {nifti_dir}")
    print(f"  Consensus masks    : {masks_dir}")
    print(f"  Output             : {out_path}")


if __name__ == "__main__":
    main()
