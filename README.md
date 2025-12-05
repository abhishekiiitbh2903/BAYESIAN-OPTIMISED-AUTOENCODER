# 📡 BORCAE: Bayesian Optimized Residual Convolutional AutoEncoder
### Efficient QPC Feedback Compression for RIS-Assisted Time-Varying IoT Networks

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Paper](https://img.shields.io/badge/IEEE-Paper_Link-orange)](https://ieeexplore.ieee.org/document/11271182)
---

## 🧠 Overview

Reconfigurable Intelligent Surfaces (RIS) have the capability to significantly enhance time-varying IoT networks, but their performance is limited by the need for frequent Quantized Phase Configuration (QPC) feedback between Base Station (BS) and RIS controller. As RIS dimensions scale, feedback bandwidth becomes insufficient, making compression essential.

**BORCAE** (Bayesian Optimized Residual Convolutional AutoEncoder) is proposed as a lightweight and noise-resilient framework that compresses QPC bits while maintaining high reconstruction fidelity. The model integrates:

- Residual connections for stable reconstruction,
- Bayesian hyperparameter optimization via Optuna for deployment adaptability,
- LBFGS optimizer in final epochs for accelerated convergence & training stability.

BORCAE consistently outperforms DL-CsiNet and CsiNet under varying SINR environments using NMSE as the primary metric, demonstrating its real-world feasibility for RIS-assisted IoT and 6G systems.
> 📄 [BORCAE: Bayesian Optimized Residual Convolutional Autoencoder for Efficient Feedback Compression in RIS-Assisted Time-Varying IoT Networks](https://ieeexplore.ieee.org/document/11271182)  
> _IEEE Transactions on Artificial Intelligence, 2025._
---

## 🚀 Key Features

- 📉 **Bandwidth-efficient RIS feedback encoding**
- 🧠 **Residual 1D Convolutional Autoencoder for improved stability**
- 🎯 **Bayesian hyperparameter tuning (Optuna)**
- ⚡ **LBFGS optimizer for final convergence**
- 🌐 **Noise-resilient under fluctuating SINR**
- 📊 **State-of-the-art NMSE performance vs existing baselines**

---

## 📊 Benchmark Comparison

| Model      | NMSE ↓ | Compression Ratio |
|------------|--------|-------------------|
| **BORCAE** | Lowest | High              |
| DL-CsiNet  | Higher | Moderate          |
| CsiNet     | Highest| Moderate          |

BORCAE achieves consistent fidelity improvements across operating regimes, making it viable for dense RIS deployments in next-generation networks.

---

## 📁 Project Structure

```bash
BORCAE/
├── dataset/                 # QPC dataset files
├── model/                   # Encoder, decoder, residual blocks
├── config.py                # Hyperparameters and runtime settings
├── main.py                  # Training / evaluation entry point
├── utils.py                 # NMSE metric, loaders, plotting
├── requirements.txt         # Dependencies
└── README.md
