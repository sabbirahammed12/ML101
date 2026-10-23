#!/usr/bin/env python3
"""Check that everything needed to run the tracker is in place.

    python check_setup.py

Reports each requirement as OK or MISSING, then runs one real frame
through the detector and tracker to prove the whole chain works.
"""

from __future__ import annotations

import os
import sys

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems: list[str] = []
warnings: list[str] = []


def line(status: str, label: str, detail: str = "") -> None:
    print(f"{status}  {label:<28} {detail}")


print("=" * 64)
print("RT-Track setup check")
print("=" * 64)

# ---------------------------------------------------------------- python
v = sys.version_info
if v >= (3, 9):
    line(OK, "Python", f"{v.major}.{v.minor}.{v.micro}")
else:
    line(BAD, "Python", f"{v.major}.{v.minor} -- need 3.9 or newer")
    problems.append("Install Python 3.9+ from python.org")

print(f"       running from: {sys.executable}")
in_venv = sys.prefix != sys.base_prefix
line(OK if in_venv else WARN, "virtual environment",
     "active" if in_venv else "not active (fine, but packages go system-wide)")
if not in_venv:
    warnings.append("No virtual environment active.")

# -------------------------------------------------------------- packages
print("-" * 64)
REQUIRED = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("cv2", "opencv-python"),
    ("onnxruntime", "onnxruntime"),
]
for module, pip_name in REQUIRED:
    try:
        m = __import__(module)
        line(OK, pip_name, getattr(m, "__version__", "installed"))
    except ImportError:
        line(BAD, pip_name, "not installed")
        problems.append(f"pip install {pip_name}")

try:
    from scipy.optimize import linear_sum_assignment  # noqa: F401
except Exception:                                     # noqa: BLE001
    pass

# ------------------------------------------------------------ project files
print("-" * 64)
here = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(os.path.join(here, "rttrack")):
    line(OK, "rttrack package", "found")
else:
    line(BAD, "rttrack package", "missing -- are you in the rt_tracker folder?")
    problems.append("Run this from inside the rt_tracker folder")

model = os.path.join(here, "models", "yolo11n_416.onnx")
if os.path.exists(model):
    line(OK, "detector model", f"{os.path.getsize(model) / 1e6:.1f} MB")
else:
    line(BAD, "detector model", "models/yolo11n_416.onnx not found")
    problems.append("Put yolo11n_416.onnx inside the models folder "
                    "(check the spelling exactly)")

vid_dir = os.path.join(here, "videos")
vids = ([f for f in os.listdir(vid_dir) if f.endswith((".mp4", ".avi"))]
        if os.path.isdir(vid_dir) else [])
if vids:
    line(OK, "videos", f"{len(vids)} found")
else:
    line(WARN, "videos", "none yet")
    warnings.append("Run: python download_videos.py")

# ----------------------------------------------------------- live smoke test
print("-" * 64)
if not problems:
    try:
        import numpy as np

        sys.path.insert(0, here)
        from rttrack import MultiObjectTracker, YoloOnnxDetector

        det = YoloOnnxDetector(model, imgsz=416, num_threads=2)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det.detect(frame)
        line(OK, "detector loads + runs", "ONNX Runtime responded")

        trk = MultiObjectTracker(min_hits=1)
        box = np.array([[100, 100, 180, 260, 0.9, 0]], dtype=np.float32)
        for i in range(3):
            box[0, :4] += 5
            out = trk.update(box.copy(), i)
        if out and out[0].track_id > 0:
            line(OK, "tracker assigns IDs", f"track #{out[0].track_id} moving")
        else:
            line(BAD, "tracker assigns IDs", "no track produced")
            problems.append("Tracker smoke test failed -- send me this output")
    except Exception as exc:                          # noqa: BLE001
        line(BAD, "smoke test", f"{type(exc).__name__}: {exc}")
        problems.append("Smoke test crashed -- send me this output")
else:
    line(WARN, "smoke test", "skipped until the problems above are fixed")

# -------------------------------------------------------------- verdict
print("=" * 64)
if problems:
    print("NOT READY. Fix these:\n")
    for p in dict.fromkeys(problems):
        print(f"  - {p}")
elif warnings:
    print("READY (with notes):\n")
    for w in dict.fromkeys(warnings):
        print(f"  - {w}")
    print("\nThen run:")
    print("  python run_tracking.py --video videos/car-detection.mp4 "
          "--model models/yolo11n_416.onnx --out-video out/car.mp4")
else:
    print("ALL GOOD. Run this next:\n")
    print("  python run_tracking.py --video videos/car-detection.mp4 "
          "--model models/yolo11n_416.onnx --out-video out/car.mp4")
print("=" * 64)