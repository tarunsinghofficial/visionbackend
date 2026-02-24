"""Computer Vision service — YOLOv8 + OpenCV image analysis."""

from __future__ import annotations

import base64
import io
import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from models.schemas import DetectedObject

logger = logging.getLogger(__name__)

# Furniture / room-relevant COCO labels to keep
RELEVANT_LABELS: set[str] = {
    "chair", "couch", "bed", "dining table", "tv", "laptop",
    "refrigerator", "oven", "microwave", "sink", "toilet",
    "potted plant", "clock", "vase", "book", "bottle",
}

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

_model: YOLO | None = None


def _get_model(model_path: str = "yolov8n.pt") -> YOLO:
    """Lazily load YOLOv8 model (downloads on first run)."""
    global _model
    if _model is None:
        logger.info("Loading YOLOv8 model from %s …", model_path)
        _model = YOLO(model_path)
        logger.info("YOLOv8 model loaded successfully.")
    return _model


def analyze_image(
    image_bytes: bytes,
    model_path: str = "yolov8n.pt",
) -> Tuple[list[DetectedObject], str]:
    """Run YOLOv8 on an image and return detected objects + annotated base64.

    Returns:
        (detected_objects, annotated_image_base64)
    """
    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError(
            f"Image size ({len(image_bytes) / 1024 / 1024:.1f} MB) exceeds "
            f"maximum allowed size of {MAX_IMAGE_SIZE / 1024 / 1024:.0f} MB."
        )

    # Decode image
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the uploaded image. Please upload a valid image file.")

    # Run inference
    model = _get_model(model_path)
    results = model(img, verbose=False)

    detected_objects: list[DetectedObject] = []
    annotated_img = img.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])

            if label not in RELEVANT_LABELS:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detected_objects.append(
                DetectedObject(
                    label=label,
                    confidence=round(confidence, 3),
                    bbox=[round(v, 1) for v in [x1, y1, x2, y2]],
                )
            )

            # Draw bounding box
            cv2.rectangle(
                annotated_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (59, 130, 246),  # electric blue BGR
                2,
            )
            text = f"{label} {confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                annotated_img,
                (int(x1), int(y1) - th - 8),
                (int(x1) + tw + 4, int(y1)),
                (59, 130, 246),
                -1,
            )
            cv2.putText(
                annotated_img,
                text,
                (int(x1) + 2, int(y1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Encode annotated image to base64
    pil_img = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    logger.info("Detected %d relevant objects in image.", len(detected_objects))
    return detected_objects, annotated_b64
