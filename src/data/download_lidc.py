"""
download_lidc.py
----------------
Downloads the LIDC-IDRI dataset from TCIA (The Cancer Imaging Archive)
and builds a nodule annotation index.

Official source: https://www.cancerimagingarchive.net/collection/lidc-idri/
Paper: Armato et al., Med. Phys. 38:915-931, 2011

Download options (choose one):
  A) tcia_utils Python package  ← automated, used by this script
  B) NBIA Data Retriever GUI     ← manual, from https://wiki.cancerimagingarchive.net/display/NBIA
  C) nbiatoolkit CLI             ← alternative to tcia_utils

After DICOM download, this script also writes:
    /data/lidc_idri/nodule_annotations.json

Usage:
    python src/data/download_lidc.py --output /data/lidc_idri

    # If DICOMs are already downloaded (e.g. via NBIA Data Retriever):
    python src/data/download_lidc.py \
        --output /data/lidc_idri \
        --dicom_home /path/to/existing/dicoms \
        --skip_download
"""

import argparse
import json
from pathlib import Path

import numpy as np


TCIA_COLLECTION = "LIDC-IDRI"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default="/data/lidc_idri",
        help="Root directory for LIDC-IDRI data and annotations",
    )
    p.add_argument(
        "--dicom_home",
        type=str,
        default=None,
        help="Path where DICOM files are (or will be) stored. "
             "Defaults to <output>/dicoms",
    )
    p.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip DICOM download (use if DICOMs are already present)",
    )
    p.add_argument(
        "--max_series",
        type=int,
        default=220,
        help="Cap on the number of LIDC series to download (disk-limited "
             "server). 0 = all 1018 (~125 GB). Default 220 leaves margin to "
             "select ~150 scans with consensus nodules.",
    )
    return p.parse_args()


# ── Option A: download via tcia_utils ────────────────────────────────────────

def download_via_tcia_utils(dicom_home: Path, max_series: int = 0) -> None:
    try:
        from tcia_utils import nbia  # type: ignore
    except ImportError:
        raise ImportError(
            "tcia_utils is not installed. Run: pip install tcia_utils\n"
            "Alternatively, download LIDC-IDRI manually via the NBIA Data "
            "Retriever from:\n"
            "  https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images"
        )

    print(f"Fetching series list for collection: {TCIA_COLLECTION}")
    series_list = nbia.getSeries(collection=TCIA_COLLECTION)
    print(f"Found {len(series_list)} series")
    if max_series and len(series_list) > max_series:
        series_list = series_list[:max_series]
        print(f"Capped to {max_series} series (--max_series, disk-limited)")

    dicom_home.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DICOMs to {dicom_home}")
    print("This will take a while (~125 GB). The download is resumable.\n")

    nbia.downloadSeries(
        series_list,
        path=str(dicom_home),
        format="path",
    )

    print(f"\nDICOM download complete: {dicom_home}")


# ── Reorganise DICOMs into the layout pylidc expects ──────────────────────────

def reorganize_for_pylidc(dicom_home: Path) -> None:
    """
    tcia_utils downloads each series into a flat <series-UID>/ folder, but
    pylidc locates DICOMs at <dicom_path>/<patient-id>/<study-UID>/<series-UID>/.
    This restructures the flat folders in place. Idempotent — already-organised
    LIDC-IDRI-* directories are left untouched.
    """
    import pydicom

    flat_dirs = []
    for d in dicom_home.iterdir():
        if not d.is_dir() or d.name.startswith("LIDC-IDRI"):
            continue
        if any(d.glob("*.dcm")):
            flat_dirs.append(d)

    if not flat_dirs:
        print("DICOMs already in pylidc layout — nothing to reorganise")
        return

    moved = 0
    for sdir in flat_dirs:
        dcms = list(sdir.glob("*.dcm"))
        try:
            ds = pydicom.dcmread(str(dcms[0]), stop_before_pixels=True)
            patient = str(ds.PatientID).strip()
            study = str(ds.StudyInstanceUID).strip()
            series = str(ds.SeriesInstanceUID).strip()
        except Exception as e:
            print(f"  Warning: cannot read {sdir.name}: {e}")
            continue
        target = dicom_home / patient / study / series
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        sdir.rename(target)
        moved += 1
    print(f"Reorganised {moved} series into <patient>/<study>/<series>/ layout")


# ── pylidc annotation index ───────────────────────────────────────────────────

def configure_pylidc(dicom_home: str) -> None:
    import configparser

    # pylidc expects exactly:  [dicom] / path = ...
    config = configparser.ConfigParser()
    config["dicom"] = {"path": dicom_home}
    cfg_path = Path.home() / ".pylidcrc"
    with open(cfg_path, "w") as f:
        config.write(f)
    print(f"pylidc configured → [dicom] path = {dicom_home}")


def build_annotation_index(output_dir: Path) -> None:
    try:
        import numpy as np
        if not hasattr(np, "int"):   np.int   = int
        if not hasattr(np, "bool"):  np.bool  = bool
        if not hasattr(np, "float"): np.float = float
        if not hasattr(np, "complex"): np.complex = complex
        import pylidc as pl
    except ImportError:
        raise ImportError(
            "pylidc is not installed. Run: pip install pylidc"
        )

    print("\nBuilding nodule annotation index ...")
    scans = pl.query(pl.Scan).all()
    print(f"Found {len(scans)} scans in pylidc database")

    records = []
    for scan in scans:
        try:
            clusters = scan.cluster_annotations()
        except Exception as e:
            print(f"  Skipping {scan.patient_id}: {e}")
            continue

        nodules = []
        for i, cluster in enumerate(clusters):
            # Require at least 3 of 4 radiologists to have marked the nodule
            if len(cluster) < 3:
                continue

            all_bboxes = np.array([ann.bbox_matrix() for ann in cluster])
            bbox_min = all_bboxes[:, :, 0].min(axis=0).tolist()  # [z, y, x]
            bbox_max = all_bboxes[:, :, 1].max(axis=0).tolist()

            malignancy = [a.malignancy for a in cluster if a.malignancy is not None]
            diameters = [a.diameter for a in cluster if a.diameter is not None]

            nodules.append({
                "nodule_id": i,
                "malignancy": float(np.mean(malignancy)) if malignancy else None,
                "diameter_mm": float(np.mean(diameters)) if diameters else None,
            })

        if not nodules:
            continue

        records.append({
            "scan_id": str(scan.id),
            "patient_id": scan.patient_id,
            "slice_thickness": scan.slice_thickness,
            "pixel_spacing": [scan.pixel_spacing, scan.pixel_spacing],
            "nodules": nodules,
        })

    out_path = output_dir / "nodule_annotations.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    total = sum(len(r["nodules"]) for r in records)
    print(f"Annotation index written → {out_path}")
    print(f"  Scans with ≥1 consensus nodule : {len(records)}")
    print(f"  Total consensus nodule clusters : {total}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dicom_home = Path(args.dicom_home) if args.dicom_home else output_dir / "dicoms"

    if not args.skip_download:
        download_via_tcia_utils(dicom_home, max_series=args.max_series)
    else:
        print(f"Skipping download. Using existing DICOMs at {dicom_home}")
        if not dicom_home.exists():
            raise FileNotFoundError(
                f"--skip_download set but {dicom_home} does not exist."
            )

    reorganize_for_pylidc(dicom_home)
    configure_pylidc(str(dicom_home))
    build_annotation_index(output_dir)


if __name__ == "__main__":
    main()
