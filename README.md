# Grounded Vision–Language Adaptation on Chest CT

**ICSDG 2025 · Universitas Brawijaya**

> Clinically-oriented zero-shot pulmonary nodule detection and case retrieval
> via expert-routed VLM adaptation (VILA-M3 + VISTA3D) on CT-RATE, evaluated
> on LIDC-IDRI.

**Authors:** Dionisius Seraf Saputra, Nafis Naufal Rahman

📄 **[Final paper (PDF)](paper/main%20paper.pdf)**

---

## Overview

```
CT-RATE (fine-tuning)          LIDC-IDRI (zero-shot eval)
       │                                │
  LoRA fine-tune                  VISTA3D routing
  VILA-M3 8B                      (via MONAI agent)
       │                                │
  Retrieval eval               Detection eval
  Recall@1/5/10                Dice + IoU
```

The core idea: use VILA-M3 as an orchestrator that routes spatial localisation
queries to VISTA3D (a 3D CT expert) while handling case retrieval internally.
LoRA fine-tuning on CT-RATE aligns the model to thoracic clinical language
without requiring bounding box annotations.

---

## Requirements

- Python 3.10
- 1× NVIDIA A100 80 GB
- NVIDIA driver 470.x / CUDA 11.6 host → **PyTorch cu118 build** (cu12x needs driver ≥ 525)
- ~241 GB disk — the data pipeline streams CT-RATE volumes one at a time and
  deletes the raw volume after slicing (the full set is ~21 TB).

---

## Setup

```bash
# Clone the repo with submodules (includes VLM-Radiology-Agent-Framework + VILA)
git clone --recursive <your-repo-url>
cd icsdg

# Create the conda env, install the pinned cu118 stack, patch transformers
bash setup.sh
conda activate icsdg

# Point the HF cache at the data disk (not the home partition)
export ICSDG_DATA_ROOT=$HOME/icsdg_data
export HF_HOME=$ICSDG_DATA_ROOT/hf_cache
```

> **Note:** If you forgot `--recursive`, run `git submodule update --init --recursive` inside the repo.

---

## Run Order

All paths below assume `ICSDG_DATA_ROOT=$HOME/icsdg_data` (see Setup). CT-RATE
is gated — accept the terms on HuggingFace and `export HF_TOKEN=...` first.

### 1. Download dataset metadata + LIDC

```bash
export HF_TOKEN=your_token_here

# CT-RATE: only the small report/metadata CSVs — volumes are streamed in step 2
python src/data/download_ctrate.py --output $ICSDG_DATA_ROOT/ct_rate

# LIDC-IDRI: capped DICOM download (~220 series, then ~150 used)
python src/data/download_lidc.py --output $ICSDG_DATA_ROOT/lidc_idri --max_series 220

# (or, if you already have DICOMs from the NBIA Data Retriever)
python src/data/download_lidc.py --output $ICSDG_DATA_ROOT/lidc_idri \
    --dicom_home /path/to/your/dicoms --skip_download
```

### 2. Preprocess

```bash
# CT-RATE: stream ~2,000 volumes, slice each, delete the raw volume
python src/data/prepare_ctrate.py \
    --ctrate_root $ICSDG_DATA_ROOT/ct_rate \
    --output_root $ICSDG_DATA_ROOT/processed \
    --max_volumes 2000

# LIDC-IDRI: DICOM → NIfTI volumes + consensus masks + key slices
python src/data/prepare_lidc.py \
    --lidc_root $ICSDG_DATA_ROOT/lidc_idri \
    --output_root $ICSDG_DATA_ROOT/processed \
    --max_scans 150
```

### 3. Evaluate baseline (pre fine-tuning)

```bash
python src/eval/eval_retrieval.py \
    --holdout_json $ICSDG_DATA_ROOT/processed/ctrate_holdout.json \
    --output_json results/baseline_retrieval.json

python src/eval/eval_detection.py \
    --eval_json $ICSDG_DATA_ROOT/processed/lidc_eval.json \
    --condition baseline \
    --output_json results/baseline_detection.json
```

### 4. Fine-tune (single A100)

```bash
python src/train/finetune_lora.py \
    --config configs/train_config.yaml \
    --data_path $ICSDG_DATA_ROOT/processed/ctrate_train.json
```

### 5. Evaluate fine-tuned model

```bash
python src/eval/eval_retrieval.py \
    --holdout_json $ICSDG_DATA_ROOT/processed/ctrate_holdout.json \
    --lora_adapter ./checkpoints/lora_adapter_final \
    --output_json results/finetuned_retrieval.json

python src/eval/eval_detection.py \
    --eval_json $ICSDG_DATA_ROOT/processed/lidc_eval.json \
    --condition finetuned \
    --lora_adapter ./checkpoints/lora_adapter_final \
    --output_json results/finetuned_detection.json
```

---

## Configuration

All hyperparameters live in `configs/train_config.yaml`:

| Parameter | Value | Notes |
|---|---|---|
| Model | VILA-M3 8B | `MONAI/Llama3-VILA-M3-8B` |
| LoRA rank | 16 | α = 32 |
| LoRA targets | q_proj, v_proj | attention layers only |
| Learning rate | 1e-4 | cosine annealing |
| Epochs | 3 | |
| Batch size | 4 | 1 per device × 4 grad accumulation |
| CT slices | 16 | uniform axial sampling |
| Hold-out | 10% | CT-RATE retrieval evaluation |

---

## Project Structure

```
icsdg/
├── paper/
│   └── main paper.pdf         ← final compiled paper
├── configs/
│   └── train_config.yaml
├── src/
│   ├── data/
│   │   ├── download_ctrate.py
│   │   ├── download_lidc.py
│   │   ├── prepare_ctrate.py
│   │   └── prepare_lidc.py
│   ├── train/
│   │   └── finetune_lora.py
│   └── eval/
│       ├── eval_retrieval.py
│       └── eval_detection.py
├── VLM-Radiology-Agent-Framework/  ← git submodule (MONAI/NVIDIA)
├── requirements.txt
└── setup.sh
```

---

## Citation

```bibtex
@article{saputra2025vilaM3chest,
  title   = {Grounded Vision--Language Adaptation on Chest CT and Reports for
             Clinically-Oriented Zero-Shot Pulmonary Nodule Detection and Case Retrieval},
  author  = {Saputra, Dionisius Seraf and Rahman, Nafis Naufal},
  journal = {Journal of Physics: Conference Series},
  year    = {2025}
}
```

---

## Acknowledgements

Built on [VILA-M3](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework)
by NVIDIA / MONAI. Datasets: [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
· [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/).
