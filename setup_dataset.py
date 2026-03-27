"""
setup_dataset.py  —  Organise the raw Kaggle face dataset into ImageFolder format.

Usage (run once before training):
    python setup_dataset.py
"""

import os
import shutil

# ── Paths — adjust if yours differ ───────────────────────────────────────────
RAW_DIR     = "face_dataset_raw"        # where you unzipped the Kaggle download
FACES_ROOT  = os.path.join(RAW_DIR, "Faces", "Faces")
DATASET_DIR = "dataset"                 # output ImageFolder structure

# ── Helpers ───────────────────────────────────────────────────────────────────
def find_faces_root(base: str) -> str | None:
    """Walk the raw directory to locate where the images actually live."""
    for root, dirs, files in os.walk(base):
        imgs = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if len(imgs) > 20:          # heuristic: a folder with many images
            return root
    return None


def organise_flat(faces_root: str, out_dir: str):
    """
    Handles datasets where images are flat files named  'PersonName_N.jpg'.
    Moves them into  out_dir/PersonName/PersonName_N.jpg
    """
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for fname in os.listdir(faces_root):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = fname.rsplit("_", 1)
        person = parts[0] if len(parts) > 1 else os.path.splitext(fname)[0]
        dest_dir = os.path.join(out_dir, person)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(os.path.join(faces_root, fname),
                    os.path.join(dest_dir, fname))
        count += 1
    return count


def organise_subfolders(faces_root: str, out_dir: str):
    """
    Handles datasets where images are already in per-person sub-folders.
    """
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for person in os.listdir(faces_root):
        src = os.path.join(faces_root, person)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(out_dir, person)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        count += len(os.listdir(src))
    return count


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if os.path.exists(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        print(f"'{DATASET_DIR}' already exists with "
              f"{len(os.listdir(DATASET_DIR))} classes. Skipping.")
        return

    # Auto-locate image root
    root = FACES_ROOT if os.path.isdir(FACES_ROOT) else find_faces_root(RAW_DIR)
    if root is None:
        print(f"ERROR: Could not find images under '{RAW_DIR}'.")
        print("Please unzip the Kaggle dataset there first.")
        return

    print(f"Found image root: {root}")

    # Decide layout: flat files vs sub-folders
    subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if subdirs:
        print("Detected sub-folder layout — copying…")
        n = organise_subfolders(root, DATASET_DIR)
    else:
        print("Detected flat-file layout — organising by name prefix…")
        n = organise_flat(root, DATASET_DIR)

    classes = os.listdir(DATASET_DIR)
    print(f"\n✅  Done!  {n} images across {len(classes)} classes → '{DATASET_DIR}/'")
    print("Sample classes:", sorted(classes)[:8])


if __name__ == "__main__":
    main()
