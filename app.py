"""
app.py  —  Streamlit Celebrity Face Recognition App
Run locally : streamlit run app.py
Deploy      : push to GitHub → connect to Streamlit Community Cloud
"""

import os
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from PIL import Image, ImageDraw

from model import FaceRecognitionCNN
from utils import detect_and_crop_face, preprocess, get_detector

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Celebrity Face Recognition",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH   = "face_model.pth"
METRICS_PATH = "metrics.json"


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    ckpt        = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = ckpt["class_names"]
    model       = FaceRecognitionCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, class_names


@st.cache_data(show_spinner=False)
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return None
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(model, class_names, pil_image):
    """Return (name, confidence_pct, all_probs_array)."""
    tensor = preprocess(pil_image).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]) * 100, probs


def draw_bbox(pil_image: Image.Image) -> Image.Image:
    """Draw MTCNN bounding boxes on the original image."""
    import numpy as np
    detector = get_detector()
    img_np   = np.array(pil_image.convert("RGB"))
    results  = detector.detect_faces(img_np)
    draw_img = pil_image.copy()
    draw     = ImageDraw.Draw(draw_img)
    for r in results:
        x, y, w, h = r["box"]
        x, y = max(0, x), max(0, y)
        conf = r["confidence"]
        draw.rectangle([x, y, x + w, y + h], outline="#00FF88", width=3)
        draw.text((x + 4, max(0, y - 18)), f"{conf:.2f}", fill="#00FF88")
    return draw_img


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/face-id.png", width=80)
st.sidebar.title("🎬 Face Recognition")
st.sidebar.caption(f"Running on **{str(DEVICE).upper()}**")

page = st.sidebar.radio(
    "Navigate",
    ["🔍 Identify Celebrity", "📊 Model Metrics", "ℹ️ About"],
)

model, class_names = load_model()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Identify Celebrity
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Identify Celebrity":
    st.title("🔍 Celebrity Face Identification")
    st.markdown("Upload a photo and the model will identify which celebrity it is.")

    if model is None:
        st.error(
            "⚠️ **Model file not found.**  "
            "Please train the model first with `python train.py`, "
            "then place `face_model.pth` in the same folder as `app.py`."
        )
        st.stop()

    uploaded = st.file_uploader(
        "Drop an image here", type=["jpg", "jpeg", "png", "webp", "jfif"]
    )

    if uploaded:
        raw_img = Image.open(uploaded).convert("RGB")

        col1, col2, col3 = st.columns([1.2, 1.2, 1.6])

        with col1:
            st.subheader("📷 Original")
            st.image(raw_img, use_container_width=True)

        with st.spinner("Detecting & identifying face…"):
            # Face detection
            face_crop = detect_and_crop_face(raw_img)
            no_face   = face_crop is None
            if no_face:
                face_crop = raw_img   # fall back to full image
                st.warning("⚠️ No face detected — using full image for prediction.")

            # Draw bboxes on original
            annotated = draw_bbox(raw_img)

            # Predict
            name, conf, all_probs = predict(model, class_names, face_crop)

        with col2:
            st.subheader("🟢 Detected Face")
            st.image(annotated, use_container_width=True)

        with col3:
            st.subheader("🏆 Prediction")
            st.metric("Identity", name, f"{conf:.1f}% confidence")

            # Confidence gauge
            gauge_color = "#2ecc71" if conf >= 70 else "#f39c12" if conf >= 40 else "#e74c3c"
            st.markdown(
                f"""
                <div style='background:#222;border-radius:8px;padding:4px;margin-bottom:12px'>
                  <div style='background:{gauge_color};width:{conf:.0f}%;height:16px;
                              border-radius:6px;transition:width 0.5s'></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Top-5 bar chart
            top5_idx   = np.argsort(all_probs)[::-1][:5]
            top5_names = [class_names[i] for i in top5_idx]
            top5_probs = [all_probs[i] * 100 for i in top5_idx]

            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            colors = ["#2ecc71" if n == name else "#3498db" for n in top5_names]
            bars = ax.barh(top5_names[::-1], top5_probs[::-1], color=colors[::-1])
            ax.set_xlabel("Probability (%)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444")
            for bar, prob in zip(bars, top5_probs[::-1]):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                        f"{prob:.1f}%", va="center", color="white", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)

        # Cropped face preview
        if not no_face:
            with st.expander("🔎 Cropped face used for classification"):
                st.image(face_crop, width=160)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Metrics
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.title("📊 Model Performance")

    metrics = load_metrics()
    if metrics is None:
        st.error(
            "⚠️ **metrics.json not found.**  "
            "Run `python train.py` to generate it."
        )
        st.stop()

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Val Accuracy", f"{metrics['best_accuracy']}%")
    c2.metric("Total Classes",     len(metrics["class_names"]))
    c3.metric("Epochs Trained",    len(metrics["history"]["val_acc"]))

    st.divider()

    # ── Training curves ───────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("📉 Training Loss")
        fig, ax = plt.subplots()
        ax.plot(metrics["history"]["train_loss"], color="#e74c3c", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training Loss per Epoch")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_b:
        st.subheader("📈 Validation Accuracy")
        fig, ax = plt.subplots()
        ax.plot(metrics["history"]["val_acc"], color="#2ecc71", linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Val Accuracy per Epoch")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.divider()

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.subheader("🔲 Confusion Matrix")
    cm           = np.array(metrics["confusion_matrix"])
    class_names_ = metrics["class_names"]
    # Show only if manageable size
    if len(class_names_) <= 30:
        fig, ax = plt.subplots(figsize=(max(8, len(class_names_) // 2),
                                        max(6, len(class_names_) // 2)))
        sns.heatmap(cm, annot=len(class_names_) <= 15,
                    fmt="d", cmap="Blues",
                    xticklabels=class_names_,
                    yticklabels=class_names_, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0,  fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Confusion matrix hidden (too many classes to display clearly).")

    st.divider()

    # ── Per-class report ──────────────────────────────────────────────────────
    st.subheader("📋 Per-Class Report")
    report = metrics["report"]
    rows   = []
    for cls in class_names_:
        if cls in report:
            r = report[cls]
            rows.append({
                "Class":     cls,
                "Precision": f"{r['precision']*100:.1f}%",
                "Recall":    f"{r['recall']*100:.1f}%",
                "F1-Score":  f"{r['f1-score']*100:.1f}%",
                "Support":   int(r["support"]),
            })
    st.dataframe(rows, use_container_width=True, height=400)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — About
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.title("ℹ️ About This App")
    st.markdown("""
### Celebrity Face Recognition System

This app uses a **custom Convolutional Neural Network (CNN)** to identify celebrity faces.

---

#### 🏗️ Architecture
- **3 convolutional blocks** (32 → 64 → 128 filters) with BatchNorm, ReLU, MaxPool & Dropout
- **Fully connected classifier**: 18432 → 512 → 256 → N classes
- Input size: **96 × 96 RGB**

#### 🔎 Face Detection
- Uses **MTCNN** (Multi-task Cascaded Convolutional Networks) for robust face detection
  before passing the crop to the classifier.

#### 🏋️ Training Details
| Setting | Value |
|---|---|
| Optimiser | Adam (lr=0.001, wd=1e-4) |
| Scheduler | Cosine Annealing |
| Loss | Cross-Entropy + label smoothing 0.1 |
| Augmentation | HFlip, Rotation±15°, ColorJitter |
| Split | 80% train / 20% test (stratified) |
| Epochs | 30 |

#### 📁 File Structure
```
face_app/
├── model.py          # CNN architecture
├── utils.py          # MTCNN detection & preprocessing
├── train.py          # Training script
├── app.py            # This Streamlit app
├── requirements.txt  # Python dependencies
├── face_model.pth    # Trained weights (after training)
└── metrics.json      # Training metrics (after training)
```

#### 🚀 How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organise your dataset
#    dataset/<Person_Name>/<image>.jpg

# 3. Train the model
python train.py

# 4. Launch the app
streamlit run app.py
```
""")
