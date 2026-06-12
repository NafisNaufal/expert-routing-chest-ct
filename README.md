# Multi-Class Expert Routing for Zero-Shot Chest CT Analysis via Vision–Language Models

**Nafis Naufal Rahman · Dionisius Seraf Saputra**  
Universitas Brawijaya · *under review at Visual Computing for Industry, Biomedicine, and Art*

---

## Overview

We fine-tune **VILA-M3 8B** on CT-RATE via **LoRA** and route volumetric localisation queries to **VISTA3D** (a 127-class 3D CT segmentation expert) through the MONAI VLM Agent Framework. Two adapters are studied:

- **SC-LoRA** — single-class adapter (9k CT-RATE pairs, `lung tumor` target)
- **MC-LoRA** — multi-class adapter (39k pairs, 5 VISTA3D classes)

A **format-vs-class precision decomposition** separates routing failures into structural misformation (remediable by training) and semantic misclassification (remediable by class-constrained inference). Zero-shot evaluation is on **LIDC-IDRI** (115 consensus-annotated scans), which is never seen during training.

```
CT-RATE (fine-tuning)              LIDC-IDRI (zero-shot eval)
        │                                    │
   LoRA fine-tune                     VISTA3D routing
   VILA-M3 8B                         (via MONAI agent)
        │                                    │
  Format precision            Dice + IoU vs expert masks
  Class precision
```

### Key results

| Condition | Fmt. prec. | Cls. prec. | Dice ± SD |
|---|---|---|---|
| VILA-M3 baseline | 0% | 0% | 0.000 |
| Direct VISTA3D (ceiling) | — | 100% | 0.273 ± 0.306 |
| SC-LoRA (unconstrained) | 100% | 59.1% | 0.167 ± 0.280 |
| SC-LoRA + constrained | 100% | 100%* | 0.273 ± 0.306 |
| MC-LoRA (unconstrained) | 100% | 0.9% | 0.001 ± 0.012 |
| **MC-LoRA + constrained** | **100%** | **100%\*** | **0.273 ± 0.306** |

\* Effective class precision under class-constrained inference (Algorithm 1 in paper).

---

## Requirements

- Python 3.10
- 1× NVIDIA A100 80 GB
- NVIDIA driver 470.x / CUDA 11.6 → **PyTorch cu118** (cu12x needs driver ≥ 525)
- ~241 GB disk — CT-RATE volumes are streamed one at a time and deleted after slicing

---

## Setup

```bash
git clone --recursive https://github.com/NafisNaufal/expert-routing-chest-ct.git
cd expert-routing-chest-ct

bash setup.sh
conda activate icsdg

export ICSDG_DATA_ROOT=$HOME/icsdg_data
export HF_HOME=$ICSDG_DATA_ROOT/hf_cache
```

> If you forgot `--recursive`: `git submodule update --init --recursive`

---

## Run Order

CT-RATE is gated — accept the terms on HuggingFace and set `HF_TOKEN` first.

### 1. Download data

```bash
export HF_TOKEN=your_token_here

# CT-RATE metadata (volumes are streamed later)
python src/data/download_ctrate.py --output $ICSDG_DATA_ROOT/ct_rate

# LIDC-IDRI
python src/data/download_lidc.py --output $ICSDG_DATA_ROOT/lidc_idri --max_series 220
```

### 2. Preprocess

```bash
# Single-class instruction data (9k volumes)
python src/data/prepare_ctrate.py \
    --ctrate_root $ICSDG_DATA_ROOT/ct_rate \
    --output_root $ICSDG_DATA_ROOT/processed \
    --max_volumes 9000

# Multi-class instruction data (39k samples from same volumes)
python src/data/prepare_ctrate_multiclass.py \
    --ctrate_root $ICSDG_DATA_ROOT/ct_rate \
    --output_root $ICSDG_DATA_ROOT/processed

# LIDC-IDRI → NIfTI + consensus masks
python src/data/prepare_lidc.py \
    --lidc_root $ICSDG_DATA_ROOT/lidc_idri \
    --output_root $ICSDG_DATA_ROOT/processed
```

### 3. Baseline evaluation (pre fine-tuning)

```bash
python src/eval/eval_detection.py \
    --eval_json $ICSDG_DATA_ROOT/processed/lidc_eval.json \
    --condition baseline \
    --output_json results/baseline_detection.json

python src/eval/eval_retrieval.py \
    --holdout_json $ICSDG_DATA_ROOT/processed/ctrate_holdout.json \
    --output_json results/baseline_retrieval.json
```

### 4. Fine-tune

```bash
# Single-class adapter
python src/train/finetune_lora.py --config configs/train_config.yaml

# Multi-class adapter
python src/train/finetune_lora.py --config configs/train_config_multiclass.yaml
```

### 5. Evaluate fine-tuned models

```bash
# SC-LoRA detection
python src/eval/eval_detection.py \
    --lora_adapter ./checkpoints/lora_adapter_final \
    --condition finetuned \
    --output_json results/finetuned_detection_10k.json

# MC-LoRA routing precision (per class)
python src/eval/eval_routing_multiclass.py \
    --lora_adapter ./checkpoints/lora_mc_final \
    --output_json results/multiclass_routing.json

# Aggregate results + bootstrap CIs + Wilcoxon tests
python src/eval/analyze_results.py
```

---

## Configuration

| Parameter | SC-LoRA | MC-LoRA |
|---|---|---|
| Model | VILA-M3 8B | VILA-M3 8B |
| LoRA rank / α | 16 / 32 | 16 / 32 |
| LoRA targets | q\_proj, v\_proj | q\_proj, v\_proj |
| Training samples | 9,000 | ~39,000 |
| Routing classes | lung tumor | lung tumor, heart, liver, aorta, lung |
| Learning rate | 1e-4 | 1e-4 |
| Epochs | 3 | 3 |
| GPU | A100 80 GB | A100 80 GB |

Full hyperparameters in `configs/train_config.yaml` and `configs/train_config_multiclass.yaml`.

---

## Project Structure

```
expert-routing-chest-ct/
├── paper/
│   ├── main.tex                    ← LaTeX source (sn-jnl / Springer Nature)
│   ├── sn-jnl.cls                  ← Springer class file
│   ├── sn-mathphys-num.bst
│   └── figures/
│       └── umap_multiclass.pdf
├── configs/
│   ├── train_config.yaml           ← SC-LoRA hyperparameters
│   └── train_config_multiclass.yaml
├── src/
│   ├── data/
│   │   ├── download_ctrate.py
│   │   ├── download_lidc.py
│   │   ├── prepare_ctrate.py
│   │   ├── prepare_ctrate_multiclass.py
│   │   └── prepare_lidc.py
│   ├── train/
│   │   └── finetune_lora.py
│   ├── eval/
│   │   ├── eval_detection.py
│   │   ├── eval_retrieval.py
│   │   ├── eval_routing_multiclass.py
│   │   ├── analyze_results.py
│   │   ├── nodule_size_stats.py
│   │   └── plot_umap.py
│   └── viz/
│       └── gen_umap.py
├── VLM-Radiology-Agent-Framework/  ← git submodule (MONAI/NVIDIA)
├── requirements.txt
└── setup.sh
```

---

## Citation

```bibtex
@article{rahman2025expertrouting,
  title   = {Multi-Class Expert Routing for Zero-Shot Chest {CT} Analysis
             via Vision--Language Models},
  author  = {Rahman, Nafis Naufal and Saputra, Dionisius Seraf},
  journal = {Visual Computing for Industry, Biomedicine, and Art},
  year    = {2025},
  note    = {under review}
}
```

---

## Acknowledgements

Built on [VILA-M3](https://github.com/Project-MONAI/VLM-Radiology-Agent-Framework) by NVIDIA / MONAI.  
Datasets: [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) · [LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/).  
Training compute: HPC AI-Center, Universitas Brawijaya (NVIDIA A100 80 GB).
