# U-net-vs-Attention-U-net

> **Project:** Beyond Dice — Junction-Level Evaluation of U-Net vs Attention U-Net  
> **Dataset:** DRIVE (Digital Retinal Images for Vessel Extraction)  
> **Framework:** TensorFlow 2.10 + Keras  
> **GPU:** NVIDIA RTX 4050 (6GB VRAM)

---

## 📌 Overview

This project compares **U-Net** and **Attention U-Net** architectures for retinal vessel segmentation on the DRIVE dataset. Beyond standard Dice scoring, we introduce **junction-level evaluation** — measuring model accuracy at vascular branch points.

## 🧠 Models

| Model | Description |
|-------|-------------|
| `unetmodel` | Standard U-Net |
| `attentionunet` | U-Net with Attention Gates |
| `residualunet` | Residual U-Net |
| `attention_residualunet` | Attention + Residual U-Net |

## 📁 Project Structure

```
├── model.py               # All 4 model architectures
├── train.py               # Training script
├── test.py                # Testing & evaluation script
├── evaluation_metrics.py  # IoU, Dice metrics
├── requirements.txt       # Python dependencies
├── SETUP_ANTIGRAVITY.md   # Environment setup guide
└── data/
    └── DRIVE/
        ├── training/
        │   ├── images/      ← 20 retinal .tif images
        │   ├── 1st_manual/  ← 20 vessel masks (.gif)
        │   └── mask/        ← 20 FOV masks (.gif)
        └── test/
            ├── images/
            └── mask/
```

## ⚙️ Setup

```bash
# Activate environment (from Unet_mini folder)
..\retinal_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Training

```python
# Edit train.py — set paths and batch_size = 5
python train.py
```

## 📊 Evaluation

```python
python test.py
```

## 🔭 Phase 5 — Junction-Level Evaluation (Coming Soon)

Custom metric beyond standard Dice:
- Skeletonize predicted vessel masks
- Detect junction pixels (3+ neighbours)
- Compute junction-level Dice, Recall, Precision

---

*Mini Project — AIML | Beyond Dice*
