# 🔬 Retinal Vessel Segmentation — Antigravity Setup Guide

> **Project:** Beyond Dice: Junction-Level Evaluation of U-Net vs Attention U-Net  
> **Dataset:** DRIVE only  
> **Environment:** Antigravity (Local GPU — RTX 4050, 6GB VRAM)

---

## ⚠️ Important Notes Before Starting

- We are using **only the DRIVE dataset** (not HRF or STARE)
- Batch size is set to **5** (original repo used 16 — reduced for RTX 4050 6GB VRAM)
- Framework: **TensorFlow + Keras**
- Python version: **3.9**

---

## 📁 Step 1: Clone the Repository




🐍 Use venv instead of conda (simpler in Antigravity)
bash
# Step 1 — Clone repo
git clone https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET.git
cd Retinal-Vessel-Segmentation-using-variants-of-UNET

# Step 2 — Create virtual environment
python -m venv retinal_env

# Step 3 — Activate it
# On Windows (Antigravity):
retinal_env\Scripts\activate

# Step 4 — Install requirements
pip install -r requirements.txt
✅ Why venv is better here
	venv	conda
Lightweight	✅	❌ heavy
Works in Antigravity	✅	needs setup
Project-isolated	✅	✅
Easy to delete & redo	✅	✅
⚠️ One important thing

Since you’re on Windows + Antigravity, make sure you create the venv inside the project folder so everything stays together:

Retinal-Vessel-Segmentation-using-variants-of-UNET/
├── retinal_env/        ← venv lives here
├── model.py
├── train.py
└── ...


```bash
git clone https://github.com/arkanivasarkar/Retinal-Vessel-Segmentation-using-variants-of-UNET.git
cd Retinal-Vessel-Segmentation-using-variants-of-UNET
```

---

## 🐍 Step 2: Create and Activate Conda Environment

```bash
# Create new environment with Python 3.9
conda create -n retinal_seg python=3.9 -y

# Activate the environment
conda activate retinal_seg
```

---

## 📦 Step 3: Install Requirements

```bash
# Core deep learning
pip install tensorflow==2.10.0

# Image processing
pip install opencv-python==4.7.0.72
pip install scikit-image==0.20.0
pip install Pillow==9.5.0

# Data & utilities
pip install numpy==1.23.5
pip install matplotlib==3.7.1
pip install tqdm==4.65.0
pip install scikit-learn==1.2.2

# Skeletonization (needed for Phase 5 — junction evaluation)
pip install scipy==1.10.1
```

Or install all at once using the requirements below.

### 📄 requirements.txt (save and run)

```
tensorflow==2.10.0
opencv-python==4.7.0.72
scikit-image==0.20.0
Pillow==9.5.0
numpy==1.23.5
matplotlib==3.7.1
tqdm==4.65.0
scikit-learn==1.2.2
scipy==1.10.1
```

```bash
# Install from file
pip install -r requirements.txt
```

---

## ✅ Step 4: Verify GPU is Detected

Run this in Python or a Jupyter cell:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

# Expected output:
# GPUs detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If GPU is not detected, install CUDA-compatible TensorFlow:

```bash
pip install tensorflow-gpu==2.10.0
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
```

---

## 📂 Step 5: Organize DRIVE Dataset

Place the DRIVE dataset in the following structure inside the project folder:

```
Retinal-Vessel-Segmentation-using-variants-of-UNET/
│
├── data/
│   └── DRIVE/
│       ├── training/
│       │   ├── images/        ← 20 retinal images (.tif)
│       │   ├── 1st_manual/    ← 20 vessel masks (.gif)
│       │   └── mask/          ← 20 FOV masks (.gif)
│       └── test/
│           ├── images/        ← 20 retinal images (.tif)
│           └── mask/          ← 20 FOV masks (.gif)
│
├── model.py
├── train.py
├── test.py
└── evaluation_metrics.py
```

> 📝 Your current DRIVE path on PC:
> `D:\MINI_PROJECT\datsets\archive\DRIVE`  
> Copy the `DRIVE` folder into the `data/` directory above.

---

## ✏️ Step 6: Modify `train.py` — DRIVE Only + Batch Size 5

Open `train.py` and make the following changes:

### 6a. Set dataset to DRIVE only

Find the dataset loading section and change to:

```python
# CHANGE THIS — use DRIVE only
DATASET = 'DRIVE'
DATA_PATH = './data/DRIVE'

TRAIN_IMG_PATH  = f'{DATA_PATH}/training/images'
TRAIN_MASK_PATH = f'{DATA_PATH}/training/1st_manual'
TEST_IMG_PATH   = f'{DATA_PATH}/test/images'
```

### 6b. Change batch size from 16 → 5

Find this line:

```python
batch_size = 16   # original
```

Change to:

```python
batch_size = 5    # reduced for RTX 4050 (6GB VRAM)
```

---

## 🚀 Step 7: Run Training

```bash
# Make sure environment is active
conda activate retinal_seg

# Train U-Net on DRIVE
python train.py --model unet --dataset DRIVE --batch_size 5

# Train Attention U-Net on DRIVE
python train.py --model attention_unet --dataset DRIVE --batch_size 5
```

---

## 📊 Step 8: Run Evaluation

```bash
# Test U-Net
python test.py --model unet --dataset DRIVE

# Test Attention U-Net
python test.py --model attention_unet --dataset DRIVE
```

---

## 🔭 What's Next — Phase 5: Junction-Level Evaluation

After training both models, we will add our **custom junction-level evaluation** on top:

1. Skeletonize predicted vessel masks (`skimage.morphology.skeletonize`)
2. Detect junction pixels (pixels with 3+ neighbors in skeleton)
3. Compute junction-level Dice, Recall, Precision
4. Failure analysis — missed branches, broken connections, merged vessels

> This is the core contribution of **"Beyond Dice"** — no existing repo does this step.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `OOM: Out of memory` | Reduce batch size further: `batch_size = 2` |
| `GPU not detected` | Run `nvidia-smi` to check driver, reinstall CUDA toolkit |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` with env active |
| `.gif mask not loading` | Use `PIL.Image.open(path).convert('L')` instead of `cv2.imread` |
| Slow training | Enable mixed precision: `tf.keras.mixed_precision.set_global_policy('mixed_float16')` |

---

*Setup guide for Beyond Dice — Mini Project, AIML*


