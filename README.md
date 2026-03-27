# 🎬 Celebrity Face Recognition App

A full end-to-end face recognition system built with **PyTorch** + **MTCNN** + **Streamlit**.

---

## 📁 Project Structure

```
face_app/
├── model.py            # CNN architecture
├── utils.py            # MTCNN face detection & preprocessing
├── train.py            # Training script
├── setup_dataset.py    # One-time dataset organiser
├── app.py              # Streamlit web app
├── requirements.txt    # Python dependencies
├── face_model.pth      # ← Generated after training
└── metrics.json        # ← Generated after training
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset (Colab / Kaggle)
```python
!pip install kaggle
!kaggle datasets download -d vasukipatel/face-recognition-dataset
!unzip face-recognition-dataset.zip -d face_dataset_raw
```

### 3. Organise the dataset
```bash
python setup_dataset.py
```
This creates `dataset/<Person_Name>/<images>` automatically.

### 4. Train the model
```bash
python train.py
```
- Saves `face_model.pth` (best checkpoint)
- Saves `metrics.json` (loss curves, confusion matrix, per-class report)

### 5. Launch the app
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Upload `face_model.pth` and `metrics.json` to the repo  
   *(or use Git LFS for large files)*
3. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
4. Select your repo, branch, and set **Main file** = `app.py`
5. Click **Deploy** — your app is live! 🎉

---

## 🏗️ Model Architecture

```
Input (3×96×96)
    │
    ▼
Conv Block 1  [32 filters] → MaxPool → 32×48×48
    │
Conv Block 2  [64 filters] → MaxPool → 64×24×24
    │
Conv Block 3 [128 filters] → MaxPool → 128×12×12
    │
Flatten → 18432
    │
FC 512 → BN → ReLU → Dropout(0.5)
    │
FC 256 → ReLU → Dropout(0.3)
    │
FC num_classes → Softmax
```

---

## ⚙️ Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input size | 96 × 96 |
| Batch size | 32 |
| Epochs | 30 |
| Optimiser | Adam (lr=1e-3, wd=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropy + label smoothing 0.1 |
| Augmentation | HFlip, Rotation±15°, ColorJitter |
| Train/Test split | 80% / 20% stratified |

---

## 📋 App Pages

| Page | Description |
|------|-------------|
| 🔍 Identify Celebrity | Upload a photo → MTCNN detects face → CNN classifies |
| 📊 Model Metrics | Training curves, confusion matrix, per-class precision/recall |
| ℹ️ About | Architecture & usage details |
