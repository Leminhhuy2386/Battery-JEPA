# Battery-JEPA: Self-Supervised Joint Embedding Predictive Architecture for Universal Battery Degradation Representation

Welcome to the official repository for **Battery-JEPA**, a self-supervised foundation model architecture designed to learn a shared, chemistry-independent latent degradation manifold.

Battery-JEPA captures universal aging dynamics across multiple battery chemistries—**Li-ion**, **Zn-ion**, **Na-ion**, and **CALB pouch cells**—using self-supervised joint embedding pretraining and **Input-Adaptive Linear Probing (IALP)** calibration.

---

## ⚡ Highlights

* **Self-Supervised Foundations**: Leverages joint embedding predictive pretraining on electrochemical capacity-voltage profiles to learn physical degradation dynamics without relying on target downstream labels.
* **Universal Latent Manifold**: Captures a shared latent aging space where degradation state (SOH) acts as a chemistry-agnostic coordinate.
* **Downstream Efficiency**: Outperforms fully supervised baselines using frozen representations with lightweight downstream calibration (IALP), making it highly efficient for Edge Battery Management Systems (BMS).

---

## 📊 Results Summary

The table below compares the performance of **Battery-JEPA (IALP)** against the baseline supervised winners from the **BatteryLife** benchmark (measured in Mean Absolute Percentage Error (MAPE) and Accuracy @ 15% error tolerance):

| Domain / Chemistry | Best Paper Baseline | Paper Best MAPE | Battery-JEPA (IALP) MAPE ↓ | Battery-JEPA (IALP) Acc@15% ↑ | Improvement (ΔMAPE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Li-ion** (MIX_large) | CPMLP | 0.179 | **0.173** | **64.2%** | **+0.006** |
| **Zn-ion** (ZN-coin) | CPTransformer | 0.515 | **0.344** | **34.8%** | **+0.171** |
| **Na-ion** (NA-ion) | CPTransformer | 0.255 | **0.253** | **40.0%** | **+0.002** |
| **CALB** (CALB pouch) | CPMLP | 0.140 | **0.119** | **80.0%** | **+0.021** |

> [!NOTE]
> Battery-JEPA beats the paper domain winner in all 4 domains, with the most significant reduction in Zn-ion domain MAPE (+0.171 lower).

---

## 🛠️ Usage

To run model training or evaluations:
```bash
python main.py
```
To evaluate predictions:
```bash
python evaluate_model.py
```

---

### Reference & Citation

```bibtex
@article{batteryjepa2026,
  title={Universal Battery Degradation Representation via Self-Supervised Joint Embedding Predictive Architectures},
  author={Le Minh Huy and Ruifeng Tan and Weixiang Hong and et al.},
  journal={arXiv preprint/under review},
  year={2026}
}
```
