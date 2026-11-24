# Battery-JEPA

Official core implementation for:

> **Cross-chemistry battery aging representations for electrochemical analysis and lifetime prediction**  
> Phuong Thuy Vu, Huu Thuan Vo, Hong Ha Thi Vu\*, Minhhuy Le\*  
> *Phenikaa School of Engineering, Phenikaa University, Hanoi, Vietnam*

---

## Key Highlights

- **Label-Free Multi-Chemistry Pre-training:** Adapts masked-cycle Joint Embedding Predictive Architecture (JEPA) with a cycle-order latent-distance regularizer ($L_{\text{mono}}$) on early voltage--current--capacity ($V$--$I$--$Q$) histories without using end-of-life (EOL) labels during pre-training.
- **Input-Adaptive Linear Probing (IALP):** Transfers a frozen Li-ion temporal Transformer backbone ($66{,}944$ parameters) to Zn-ion, Na-ion, and industrial CALB LFP cells by adapting only chemistry-specific intra-cycle input projections ($52{,}672$ parameters) and a regression head, preserving a shared 64-dimensional temporal coordinate basis across domains.
- **Cross-Chemistry Lifetime Prediction:** Reduces Zn-ion MAPE by **33.2%** vs. the published BatteryLife supervised SOTA (`0.3439` vs. `0.5150`; Acc@15% `+14.5 pp`: `34.74%` vs. `20.20%`) and reduces full-cohort 5-fold CV MAPE across all 31 Na-ion cells by **21.2%** vs. training from scratch (`0.1983` vs. `0.2516`; cell-level Acc@15% `51.6%` vs. `32.3%`).
- **Competitive with Parameter-Matched PEFT:** Under matched normalization and optimization on Li-ion (`MIX_large`, Aligned), IALP (`0.1871` MAPE / `62.47%` Acc@15%) outperforms From-Scratch (`0.1914`), Random-Frozen IALP (`0.1914`), parameter-matched LoRA (`0.2064`), Bottleneck Adapters (`0.2063`), Standard LP (`0.2068`), and Full Fine-Tuning (`0.2132`).
- **Validated Physical Associations & Independent Internal Resistance Prediction:** Under train-selected, held-out-test-validated within-cell controls, latent coordinates track discharge capacity ($r=-0.846$ in Zn-ion, $+0.818$ in Na-ion, $+0.501$ in Li-ion) and $dQ/dV$ peak height ($r=+0.702$ to $-0.783$). On 24 held-out MATR Li-ion test cells with independently measured internal resistance, train-selected Dim 48 achieves within-cell $r=-0.607$ (**100% same sign across all 24 cells**), and a 64D linear Ridge probe predicts measured internal resistance with **$r=+0.778$** (within-cell **$r=+0.801$**, $R^2=0.581$) — nearly **4$\times$ stronger** than curve-derived resistance proxies ($r=+0.230$).
- **Voltage Scaling & Full-Lifecycle Dynamics:** Global voltage scaling retains a Zn-ion discharge plateau displacement observed in **43.4%** of all 99 cells by EOL ($|\delta V|\ge 20\text{ mV}$) that per-cycle alignment suppresses. Over full lifecycles, a strictly causal online EWMA detector (using only cycles $\le t$) achieves **85.0% detection / 0.0% early FAR** (`130.6`-cycle mean lead) on Zn-ion and **100.0% detection / 0.0% early FAR** (`111.6`-cycle mean lead) on Na-ion.

---

## Highlighted Experimental Results

### 1. Official BatteryLife `seed 2021` Benchmark (Prefix-Level MAPE $\downarrow$ / Acc@15% $\uparrow$)

| Domain | BatteryLife SOTA (Tan et al., 2025) | TinyBatteryNet (Le et al., 2027) | Battery-JEPA (IALP, Ours) | 95% Cell-Level Cluster Bootstrap CI (MAPE) |
| :--- | :---: | :---: | :---: | :---: |
| **Li-ion** (`MIX_large`, $N=158$ cells) | 0.1790 / 62.00% | **0.1700** / 62.90% | 0.1731 / **64.24%** | [0.1389, 0.2118] |
| **Zn-ion** (`ZN-coin`, $N=20$ cells) | 0.5150 / 20.20% | 0.3680 / 25.10% | **0.3439** / **34.74%** | [0.2271, 0.4698] |
| **Na-ion** (`NAion`, $N=5$ cells) | 0.2550 / **40.60%** | **0.2320** / 40.00% | 0.2532 / 40.00% | [0.1043, 0.4201] |
| **CALB** (`CALB`, $N=5$ cells) | 0.1400 / 70.40% | 0.1330 / 70.20% | **0.1196** / **79.87%** | [0.0395, 0.2209] |

### 2. Full-Cohort Stratified 5-Fold Cell-Level Cross-Validation (All 31 Na-ion & 27 CALB Cells)

| Adaptation / Control Mode | Na-ion OOF Cell MAPE [95% CI] $\downarrow$ | Na-ion Acc@15% (Prefix / Cell) $\uparrow$ | CALB OOF Cell MAPE [95% CI] $\downarrow$ | CALB Acc@15% (Prefix / Cell) $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| From-Scratch (No Pre-training) | 0.2516 [0.201, 0.303] | 35.1% / 32.3% | **0.1491** [0.099, 0.209] | 64.3% / 55.6% |
| Random-Frozen IALP | 0.2227 [0.175, 0.274] | 39.0% / 38.7% | 0.1648 [0.106, 0.235] | 63.5% / 63.0% |
| Standard Linear Probing | 0.2039 [0.155, 0.259] | 45.0% / 51.6% | 0.1696 [0.107, 0.241] | 63.0% / 63.0% |
| Full Fine-Tuning | 0.2104 [0.166, 0.259] | 44.4% / 45.2% | 0.1674 [0.110, 0.235] | 62.4% / 59.3% |
| **Battery-JEPA (IALP, Ours)** | **0.1983** [0.149, 0.256] | **47.6% / 51.6%** | 0.1646 [0.107, 0.234] | **64.8% / 66.7%** |

### 3. Matched Adaptation & Parameter-Matched PEFT Comparison (Li-ion `MIX_large`, Aligned, $B=32$)

| Adaptation / Control Mode | Trainable Encoder Params | Prefix MAPE $\downarrow$ | Prefix Acc@15% $\uparrow$ |
| :--- | :---: | :---: | :---: |
| From-Scratch (No Pre-training) | 119,616 | 0.1914 | 58.45% |
| Random-Frozen IALP | 52,672 | 0.1914 | 54.62% |
| Standard Linear Probing | 0 | 0.2068 | 54.42% |
| Full Fine-Tuning | 119,616 | 0.2132 | 49.02% |
| LoRA ($r=28$, Q/K/V/O + FFN) | 50,176 | 0.2064 | 57.67% |
| Bottleneck Adapter ($b=100$) | 51,856 | 0.2063 | 56.36% |
| **Battery-JEPA (IALP, Ours)** | **52,672** | **0.1871** | **62.47%** |

### 4. Held-Out Within-Cell Electrochemical Tracking & Independent MATR Internal Resistance Validation

| Domain / Target | Validation Method | Test Within-Cell $r$ | Partial $r$ (Cell+Cyc) | Partial $r$ (Cell+Cyc+SOH) | Per-Cell Sign Consistency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Li-ion** ($N=158$) — Discharge Capacity | Train-selected Dim 54 | $+0.501$ | $+0.446$ | $+0.175$ | 88% |
| **Zn-ion** ($N=20$) — Discharge Capacity | Train-selected Dim 40 | $-0.846$ | $-0.797$ | $-0.266$ | **100%** |
| **Zn-ion** ($N=20$) — $dQ/dV$ Peak Height | Train-selected Dim 30 | $+0.702$ | $+0.646$ | $+0.479$ | 85% |
| **Na-ion** ($N=5$) — Resistance Proxy ($R_{\text{proxy}}$) | Train-selected Dim 59 | $+0.643$ | $+0.669$ | $+0.668$ | **100%** |
| **CALB** ($N=5$) — $dQ/dV$ Peak Height | Train-selected Dim 63 | $-0.783$ | $-0.627$ | $-0.495$ | **100%** |
| **MATR Independent Measured IR** ($N=24$) | Train-selected Dim 48 | **$-0.607$** | **$-0.397$** | $+0.167$ | **100% (24/24)** |
| **MATR Independent Measured IR** ($N=24$) | 64D Latent Ridge Probe | **$+0.801$** (pooled **$+0.778$**) | — | — | $R^2 = 0.581$ (vs. $R_{\text{proxy}}$ $r=+0.230$) |

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

## Dataset & Quick Start

Download the public **BatteryLife** benchmark (Tan et al., 2025) from the [BatteryLife repository](https://github.com/Ruifeng-Tan/BatteryLife) and place the processed data under `dataset/`.

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

## Citation

```bibtex
@article{vu2026batteryjepa,
  title   = {Cross-chemistry battery aging representations for electrochemical analysis and lifetime prediction},
  author  = {Vu, Phuong Thuy and Vo, Huu Thuan and Vu, Hong Ha Thi and Le, Minhhuy},
  journal = {Applied Energy (under review)},
  year    = {2026}
}
```
