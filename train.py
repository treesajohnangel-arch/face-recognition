"""
train.py  —  Train the FaceRecognitionCNN on an ImageFolder dataset.

Expected dataset layout:
    dataset/
        Person_A/  img1.jpg  img2.jpg ...
        Person_B/  img1.jpg  img2.jpg ...
        ...

Run:
    python train.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from tqdm import tqdm

from model import FaceRecognitionCNN

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"          # change to your path
MODEL_PATH   = "face_model.pth"
METRICS_PATH = "metrics.json"
IMG_SIZE     = 96
BATCH_SIZE   = 32
EPOCHS       = 30
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_loaders(dataset_dir: str):
    """Split ImageFolder 80/20, apply separate transforms."""
    base    = datasets.ImageFolder(root=dataset_dir)
    indices = np.arange(len(base))

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42,
        stratify=base.targets
    )

    train_ds = Subset(datasets.ImageFolder(dataset_dir, TRAIN_TRANSFORM), train_idx)
    test_ds  = Subset(datasets.ImageFolder(dataset_dir, TEST_TRANSFORM),  test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2)

    return train_loader, test_loader, base.classes


def evaluate(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            preds_all.extend(torch.argmax(outs, 1).cpu().numpy())
            labels_all.extend(lbls.numpy())
    return np.array(labels_all), np.array(preds_all)


# ── Main ──────────────────────────────────────────────────────────────────────
def train():
    print(f"Device: {DEVICE}")

    train_loader, test_loader, class_names = build_loaders(DATASET_DIR)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}  |  "
          f"Train: {len(train_loader.dataset)}  |  "
          f"Test : {len(test_loader.dataset)}")

    model     = FaceRecognitionCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc       = 0.0
    history        = {"train_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        running_loss = 0.0
        for imgs, lbls in tqdm(train_loader,
                               desc=f"Epoch {epoch:02d}/{EPOCHS}", leave=False):
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        # ── Validate ──
        labels, preds = evaluate(model, test_loader, DEVICE)
        acc = accuracy_score(labels, preds)

        history["train_loss"].append(round(avg_loss, 4))
        history["val_acc"].append(round(acc * 100, 2))

        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Val Acc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "num_classes":  num_classes,
            }, MODEL_PATH)
            print(f"  ⭐  New best saved ({acc*100:.2f}%)")

    # ── Final metrics ──
    print(f"\nTraining complete.  Best Val Accuracy: {best_acc*100:.2f}%")

    labels, preds = evaluate(model, test_loader, DEVICE)
    report = classification_report(labels, preds,
                                   target_names=class_names, output_dict=True)
    cm     = confusion_matrix(labels, preds).tolist()

    metrics = {
        "best_accuracy":   round(best_acc * 100, 2),
        "class_names":     class_names,
        "history":         history,
        "report":          report,
        "confusion_matrix": cm,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_PATH}")


if __name__ == "__main__":
    train()
