# Battery-JEPA

Official core implementation for:

> **Cross-chemistry battery aging representations for electrochemical analysis and lifetime prediction**  
> Phuong Thuy Vu, Huu Thuan Vo, Hong Ha Thi Vu*, Minhhuy Le*  
> *Phenikaa School of Engineering, Phenikaa University, Hanoi, Vietnam*

---

## Repository Contents

This public repository provides the core Battery-JEPA model, self-supervised pre-training and Input-Adaptive Linear Probing (IALP) scripts, evaluation utilities, and official BatteryLife data-split definitions:

```text
Battery-JEPA/
├── models/
│   └── BatteryJEPA.py                 # Online/EMA encoders, cross-attention predictor, IALP head
├── data_provider/
│   └── data_split_recorder.py         # Official BatteryLife train/val/test cell splits (seeds 2021, 42, 2024)
├── pretrain_jepa.py                   # Self-supervised JEPA pre-training (Li-ion MIX_large)
├── finetune_jepa.py                   # Downstream adaptation (IALP / Standard LP / Full FT)
├── evaluate_model.py                  # Evaluation script (MAPE and Acc@15%)
└── requirements.txt
```

Additional ablation scripts and pre-trained checkpoints will be released upon publication.

---

## Dataset & Splits

This study evaluates on the public **BatteryLife** multi-chemistry benchmark (Tan et al., 2025, available from the [BatteryLife repository](https://github.com/Ruifeng-Tan/BatteryLife)) across four domains using expanding early-cycle prefixes ($k_{\text{obs}} \in [5, 100]$):

| Domain | Train Cells | Val Cells | Test Cells | Test Prefixes | $\tau_{\text{EOL}}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Li-ion** (`MIX_large`) | 500 | 160 | 158 | 15,168 | 0.80 |
| **Zn-ion** (`ZN-coin`) | 59 | 20 | 20 | 1,920 | 0.80 |
| **Na-ion** (`NAion`) | 20 | 6 | 5 | 480 | 0.80 |
| **CALB** (`CALB`) | 17 | 5 | 5 | 477 | 0.90 |

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Self-supervised pre-training on Li-ion MIX_large (Aligned / Unaligned)
python pretrain_jepa.py --dataset MIX_large --align_cycles --epochs 50 --batch_size 32 --lr 1e-4 --lambda_mono 0.1
python pretrain_jepa.py --dataset MIX_large --epochs 50 --batch_size 32 --lr 1e-4 --lambda_mono 0.1

# 2. Downstream IALP adaptation (freeze temporal backbone, adapt input projections and head)
python finetune_jepa.py --dataset MIX_large --align_cycles --freeze_backbone --tune_input_proj --batch_size 64 --lr 3e-4 --lr_factor 1.0
python finetune_jepa.py --dataset ZN-coin --freeze_backbone --tune_input_proj --batch_size 16 --lr 1e-4 --lr_factor 0.5
python finetune_jepa.py --dataset NAion --freeze_backbone --tune_input_proj --batch_size 16 --lr 3e-4 --lr_factor 0.2
python finetune_jepa.py --dataset CALB --align_cycles --freeze_backbone --tune_input_proj --batch_size 16 --lr 1e-4 --lr_factor 0.05
```

---

## Summary of Main Results (`seed 2021` Official Split)

| Domain | BatteryLife SOTA (Tan et al., 2025) | TinyBatteryNet (Le et al., 2027) | Battery-JEPA (IALP, Ours) |
| :--- | :---: | :---: | :---: |
| **Li-ion** (`MIX_large`, Aligned) | 0.1790 / 62.00% | **0.1700** / 62.90% | 0.1731 / **64.24%** |
| **Zn-ion** (`ZN-coin`, Unaligned) | 0.5150 / 20.20% | 0.3680 / 25.10% | **0.3439** / **34.74%** |
| **Na-ion** (`NAion`, Unaligned) | 0.2550 / **40.60%** | **0.2320** / 40.00% | 0.2532 / 40.00% |
| **CALB** (`CALB`, Aligned) | 0.1400 / 70.40% | 0.1330 / 70.20% | **0.1196** / **79.87%** |

---

## Citation

```bibtex
@article{vu2026batteryjepa,
  title   = {Cross-chemistry battery aging representations for electrochemical analysis and lifetime prediction},
  author  = {Vu, Phuong Thuy and Vo, Huu Thuan and Vu, Hong Ha Thi and Le, Minhhuy},
  journal = {Applied Energy (under review)},
  year    = {2026}
}
```
