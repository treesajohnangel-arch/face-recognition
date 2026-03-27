import numpy as np
from PIL import Image
from torchvision import transforms

# Lazy-load MTCNN so the import doesn't crash if mtcnn isn't installed yet
_detector = None


def get_detector():
    global _detector
    if _detector is None:
        from mtcnn import MTCNN
        _detector = MTCNN()
    return _detector


# ── Inference transform (matches training normalisation) ─────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def detect_and_crop_face(pil_image: Image.Image) -> Image.Image | None:
    """
    Detect the largest face in *pil_image* with MTCNN.
    Returns a cropped PIL image of the face, or None if no face found.
    """
    detector = get_detector()
    img_np   = np.array(pil_image.convert("RGB"))
    results  = detector.detect_faces(img_np)

    if not results:
        return None

    # Pick the detection with the highest confidence
    best = max(results, key=lambda r: r["confidence"])
    x, y, w, h = best["box"]
    x, y = max(0, x), max(0, y)

    face_np = img_np[y : y + h, x : x + w]
    return Image.fromarray(face_np)


def preprocess(pil_image: Image.Image):
    """Return a (1, 3, 96, 96) tensor ready for the model."""
    return INFER_TRANSFORM(pil_image.convert("RGB")).unsqueeze(0)
