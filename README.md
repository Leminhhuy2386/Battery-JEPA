# Battery-JEPA: Self-Supervised Joint Embedding Predictive Architecture for Universal Battery Degradation Representation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scientific Domain: Battery AI](https://img.shields.io/badge/Domain-Battery%20AI-blue.svg)]()
[![Venue: Nature Energy Review](https://img.shields.io/badge/Manuscript-Nature%20Energy%20Review-red.svg)]()

Welcome to the official repository for **Battery-JEPA**, a self-supervised foundation model architecture designed to extract **universal, chemistry-independent battery degradation coordinates** and project them onto a shared latent degradation manifold. 

Battery-JEPA achieves state-of-the-art lifetime prognosis across diverse battery chemistries—**Li-ion**, **Zn-ion**, **Na-ion**, and **CALB pouch cells**—using self-supervised joint embedding pretraining and a novel **Input-Adaptive Linear Probing (IALP)** calibration framework.

---

## ⚡ Key Scientific Claims & Contributions

1. **Chemistry-Independent Shared Latent Manifold**: Battery-JEPA demonstrates that despite different electrochemical reaction platforms (intercalation vs. metal plating/stripping vs. conversion), battery aging trajectories share a collinear latent degradation coordinate system.
2. **Self-Supervised Pretraining via JEPA**: By predicting masked capacity-voltage sequences in a joint embedding space, the model learns physical degradation signatures (e.g. loss of active material, loss of lithium inventory) without relying on downstream labels.
3. **Rigorous Validation (Nature Energy Peer-Review Standards)**:
   * **Leakage-Corrected Nearest Neighbors**: Proves that local neighborhood similarity is driven by State of Health (SOH) rather than trivial cell-level leakage.
   * **True 64D Latent Geometry**: Demonstrates that the continuous degradation path is shared, as confirmed by DB-index, Silhouette score, and high cosine similarities of aging vectors (0.79 to 0.92) across chemistries.
   * **Bootstrap Statistical Robustness**: 1000 bootstrap trials establish tight confidence intervals for prediction MAPEs, confirming robustness on low-sample regimes (Na-ion, CALB).
   * **Mechanistic Failure Analysis**: Outlines clear boundaries and failure modes related to capacity anomalies and impedance polarization collapse.

---

## 📊 Performance Summary (vs. Baselines)

The table below shows the downstream prediction performance (Mean Absolute Percentage Error - MAPE and Accuracy @ 15% error tolerance) of Battery-JEPA (with IALP calibration) against the domain-best supervised baselines from the **BatteryLife** benchmark:

| Domain / Chemistry | Best Paper Baseline | Baseline MAPE | Battery-JEPA MAPE ↓ | Battery-JEPA Acc@15% ↑ | Improvement (ΔMAPE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Li-ion** (MIX_large) | CPMLP | 0.179 | **0.1732 ± 0.002** | **64.2%** | **+0.006** |
| **Zn-ion** (ZN-coin) | CPTransformer | 0.515 | **0.3437 ± 0.007** | **34.8%** | **+0.171** |
| **Na-ion** (NA-ion) | CPTransformer | 0.255 | **0.2532 ± 0.008** | **40.0%** | **+0.002** |
| **CALB** (CALB pouch) | CPMLP | 0.140 | **0.1193 ± 0.005** | **80.0%** | **+0.021** |

> [!NOTE]
> Battery-JEPA outperforms supervised models on all 4 chemistries, showing the most dramatic improvements on Zn-ion (+0.171 lower MAPE) where direct supervised models struggle with electrochemical noise and complex degradation dynamics.

---

## 🛠️ Repository Structure

* `models/BatteryJEPA.py`: The core core-joint embedding predictive architecture definition.
* `data_provider/`: Standardized data loaders and dataset split helpers.
* `utils/` & `layers/`: Auxiliary masking, losses, metrics, and network layers.
* `pretrain_jepa.py`: Self-supervised pretraining pipeline.
* `finetune_jepa.py`: Downstream adaptation script using Input-Adaptive Linear Probing (IALP) or full fine-tuning.
* `evaluate_all_save_predictions.py`: Script to generate predictions and save latent variables across datasets.
* `universal_manifold_analysis.py`: Main script for cross-chemistry k-NN, raw 64D geometry distance metrics, and bootstrap statistical validation.
* `universal_latent_dynamics.py`: Analyzes transition detection and early-life warning indicators of degradation.
* `figures/`: High-resolution figures generated for the manuscript.
* `paper/`: LaTeX files and bibliography for the scientific manuscript.

---

## 🚀 Quick Start

### 1. Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Pretraining Battery-JEPA

Pretrain the joint-embedding model using self-supervised masked target prediction:
```bash
python pretrain_jepa.py \
  --mask_ratio 0.4 \
  --mask_strategy random \
  --batch_size 128 \
  --train_epochs 20
```

### 3. Downstream Calibration (IALP)

Finetune or calibrate to a target chemistry (e.g., Zn-ion) using frozen backbone weights and a calibrated Input-Adaptive Linear Probing head:
```bash
python finetune_jepa.py \
  --pretrain_checkpoint checkpoints_jepa/pretrain.pth \
  --freeze_backbone \
  --tune_input_projection \
  --task_name long_term_forecast \
  --batch_size 64
```

### 4. Running Validation and Analysis

To run the full Nature Energy reviewer-validation analysis suite and regenerate statistical metrics:
```bash
# Analyze shared manifold geometry, k-NN ratios, distances, and bootstrap CIs
python universal_manifold_analysis.py

# Analyze degradation dynamics, transitions, and early warning horizons
python universal_latent_dynamics.py
```

---

## 📖 Citation

If you find this work helpful in your research, please cite our manuscript:

```bibtex
@article{batteryjepa2026,
  title={Universal Battery Degradation Representation via Self-Supervised Joint Embedding Predictive Architectures},
  author={Le Minh Huy and Ruifeng Tan and Weixiang Hong and et al.},
  journal={arXiv preprint/under review},
  year={2026}
}
```
