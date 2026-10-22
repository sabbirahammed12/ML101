"""YOLO object detector running on ONNX Runtime (CPU).

The detector is deliberately decoupled from the tracker: it takes a BGR
frame and returns an (N, 6) array of [x1, y1, x2, y2, score, class_id] in
original-image pixel coordinates.  Anything that honours that contract can
be swapped in -- see `MotionDetector` for a model-free fallback.

Pre-processing is letterbox resize (aspect preserved, grey padding), which
is what the YOLO models were exported with; post-processing is per-class
non-maximum suppression.
"""

from __future__ import annotations

import os
import numpy as np

try:                                            # optional at import time
    import onnxruntime as ort
except ImportError:                             # pragma: no cover
    ort = None

import cv2

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio and pad to a square `size`."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class YoloOnnxDetector:
    """YOLOv8/YOLO11-family detector on ONNX Runtime."""

    def __init__(self, model_path: str, imgsz: int = 416,
                 conf_threshold: float = 0.30, iou_threshold: float = 0.55,
                 classes: list[int] | None = None, num_threads: int = 2,
                 class_names: list[str] | None = None):
        if ort is None:                          # pragma: no cover
            raise RuntimeError("onnxruntime is not installed")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, sess_options=so, providers=["CPUExecutionProvider"])

        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = int(imgsz)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        self.classes = set(classes) if classes else None
        self.names = class_names or COCO_NAMES
        self.model_path = model_path

        # First inference pays for lazy kernel setup; do it now rather than
        # inside the first frame of a live stream.
        self.detect(np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8))

    # --------------------------------------------------------------- detect
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        blob, r, pad_x, pad_y = letterbox(frame, self.imgsz)
        x = blob[:, :, ::-1].transpose(2, 0, 1)[None]        # BGR->RGB, NCHW
        x = np.ascontiguousarray(x, dtype=np.float32) / 255.0

        raw = self.session.run(None, {self.input_name: x})[0]

        pred = np.squeeze(raw)                   # (84, 8400) or (8400, 84)
        if pred.ndim != 2:
            return np.zeros((0, 6), dtype=np.float32)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T                        # -> (num_anchors, 4 + nc)

        boxes_cxcywh = pred[:, :4]
        scores_all = pred[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        scores = scores_all[np.arange(len(class_ids)), class_ids]

        keep = scores >= self.conf
        if self.classes is not None:
            keep &= np.isin(class_ids, list(self.classes))
        if not np.any(keep):
            return np.zeros((0, 6), dtype=np.float32)

        boxes_cxcywh = boxes_cxcywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # letterbox space -> original pixels
        cx, cy, w, h = boxes_cxcywh.T
        x1 = (cx - w / 2 - pad_x) / r
        y1 = (cy - h / 2 - pad_y) / r
        x2 = (cx + w / 2 - pad_x) / r
        y2 = (cy + h / 2 - pad_y) / r

        ih, iw = frame.shape[:2]
        x1 = np.clip(x1, 0, iw - 1)
        y1 = np.clip(y1, 0, ih - 1)
        x2 = np.clip(x2, 0, iw - 1)
        y2 = np.clip(y2, 0, ih - 1)

        out = []
        for cid in np.unique(class_ids):
            m = class_ids == cid
            bx = np.stack([x1[m], y1[m], x2[m] - x1[m], y2[m] - y1[m]], axis=1)
            idx = cv2.dnn.NMSBoxes(bx.tolist(), scores[m].astype(float).tolist(),
                                   self.conf, self.iou)
            if len(idx) == 0:
                continue
            idx = np.asarray(idx).reshape(-1)
            sx1, sy1 = x1[m][idx], y1[m][idx]
            sx2, sy2 = x2[m][idx], y2[m][idx]
            sc = scores[m][idx]
            out.append(np.stack([sx1, sy1, sx2, sy2, sc,
                                 np.full(len(idx), cid)], axis=1))

        if not out:
            return np.zeros((0, 6), dtype=np.float32)
        dets = np.concatenate(out, axis=0).astype(np.float32)
        wh = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        return dets[wh > 16.0]


class MotionDetector:
    """Model-free fallback: MOG2 background subtraction + contour boxes.

    Useful for a fixed camera when no neural detector is available, and as a
    sanity check that the tracker itself is sound.  Everything it finds is
    reported as class 0 ("moving object").
    """

    def __init__(self, min_area: int = 300, history: int = 320,
                 var_threshold: float = 24.0):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.names = ["moving object"]
        self.model_path = "mog2-background-subtraction"
        self.imgsz = 0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.detect(frame)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.dilate(mask, self.kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            dets.append([x, y, x + w, y + h, 0.9, 0])
        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        return np.asarray(dets, dtype=np.float32)
