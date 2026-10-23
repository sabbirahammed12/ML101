#!/usr/bin/env python3
"""Track moving objects in a video and export their locations.

Examples
--------
# real-time pass over one clip, writing an annotated video
python run_tracking.py --video videos/car-detection.mp4 \
    --model models/yolo11n_416.onnx --out-video out/car.mp4

# webcam / live stream (source 0), no file output
python run_tracking.py --video 0 --model models/yolo11n_416.onnx --show-fps

Outputs
-------
<stem>_tracks.csv   per-frame rows: frame, time_s, track_id, class, conf,
                    x1, y1, x2, y2, cx, cy, vx, vy
<stem>_summary.json timing, throughput and tracker diagnostics
<stem>.mp4          annotated video (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rttrack import (COCO_NAMES, MotionDetector, MultiObjectTracker,
                     RealTimeTracker, YoloOnnxDetector)

# Classes that actually move from place to place (COCO ids).
MOVING_CLASSES = [0, 1, 2, 3, 5, 6, 7, 8, 14, 15, 16, 17, 18, 19, 21, 22, 23,
                  24, 26, 28, 32, 36, 38, 39, 41, 46, 47, 49, 51, 64, 67, 73]


def colour_for(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-separated colour per ID."""
    h = (int(track_id) * 47) % 180
    hsv = np.uint8([[[h, 210, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def draw(frame, tracks, names, trails, fps_text=None, detector_ran=False):
    out = frame
    for tid, cls, box, score, vel in tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        col = colour_for(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)

        label = f"#{tid} {names[cls] if cls < len(names) else cls}"
        speed = float(np.hypot(*vel))
        if speed > 0.6:
            label += f" {speed:.0f}px/f"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(out, label, (x1 + 3, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        pts = trails.get(tid, [])
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)

    if fps_text:
        cv2.rectangle(out, (0, 0), (235, 26), (0, 0, 0), -1)
        cv2.putText(out, fps_text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if detector_ran:
        cv2.circle(out, (out.shape[1] - 14, 14), 6, (0, 220, 0), -1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path, or a camera index")
    ap.add_argument("--model", default=None, help="YOLO .onnx; omit for MOG2")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.55)
    ap.add_argument("--mode", choices=["async", "sync"], default="sync")
    ap.add_argument("--detect-interval", type=int, default=0,
                    help="run the detector every N frames (0 = auto-tune "
                         "to keep up with the stream)")
    ap.add_argument("--target-fps", type=float, default=0.0,
                    help="frame budget for auto-tuning (0 = the source rate)")
    ap.add_argument("--no-flow", action="store_true")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--track-iou", type=float, default=0.30)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--all-classes", action="store_true",
                    help="keep all 80 COCO classes, not just mobile ones")
    ap.add_argument("--out-video", default=None)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--trail", type=int, default=28, help="trail length, frames")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    source = int(args.video) if str(args.video).isdigit() else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video source: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.model:
        detector = YoloOnnxDetector(
            args.model, imgsz=args.imgsz, conf_threshold=args.conf,
            iou_threshold=args.iou, num_threads=args.threads,
            classes=None if args.all_classes else MOVING_CLASSES)
        names = COCO_NAMES
    else:
        detector = MotionDetector()
        names = detector.names

    MultiObjectTracker.reset_ids()
    tracker = MultiObjectTracker(max_age=args.max_age, min_hits=args.min_hits,
                                 iou_threshold=args.track_iou,
                                 high_conf=max(args.conf, 0.35),
                                 low_conf=min(args.conf, 0.15))
    rt = RealTimeTracker(detector, tracker, mode=args.mode,
                         detect_interval=args.detect_interval,
                         use_flow=not args.no_flow,
                         target_fps=args.target_fps or src_fps)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(str(args.video)))[0]
    csv_path = os.path.join(args.out_dir, f"{stem}_tracks.csv")
    json_path = os.path.join(args.out_dir, f"{stem}_summary.json")

    writer = None
    if args.out_video:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_video)) or ".",
                    exist_ok=True)
        writer = cv2.VideoWriter(args.out_video,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (width, height))

    trails: dict[int, list[tuple[int, int]]] = {}
    rows = []
    seen_ids: set[int] = set()
    frame_idx = 0
    wall_start = time.perf_counter()
    per_frame_ms = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        res = rt.process(frame, frame_idx)
        per_frame_ms.append(res.latency_ms)
        t_s = frame_idx / src_fps

        for tid, cls, box, score, vel in res.tracks:
            seen_ids.add(tid)
            cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            rows.append([frame_idx, round(t_s, 4), tid,
                         names[cls] if cls < len(names) else cls,
                         round(float(score), 4),
                         round(float(box[0]), 2), round(float(box[1]), 2),
                         round(float(box[2]), 2), round(float(box[3]), 2),
                         round(cx, 2), round(cy, 2),
                         round(float(vel[0]), 3), round(float(vel[1]), 3)])
            pts = trails.setdefault(tid, [])
            pts.append((int(round(cx)), int(round(cy))))
            if len(pts) > args.trail:
                del pts[0]
        live = {t[0] for t in res.tracks}
        for gone in [k for k in trails if k not in live]:
            del trails[gone]

        if writer is not None:
            elapsed = time.perf_counter() - wall_start
            fps_now = (frame_idx + 1) / max(elapsed, 1e-6)
            txt = (f"{fps_now:5.1f} FPS | {len(res.tracks)} tracked"
                   if args.show_fps or True else None)
            writer.write(draw(frame.copy(), res.tracks, names, trails, txt,
                              res.detector_ran))

        frame_idx += 1
        if not args.quiet and frame_idx % 200 == 0:
            el = time.perf_counter() - wall_start
            print(f"  {frame_idx}/{n_frames or '?'} frames  "
                  f"{frame_idx / el:5.1f} FPS  {len(res.tracks)} live",
                  flush=True)

    wall = time.perf_counter() - wall_start
    cap.release()
    if writer is not None:
        writer.release()
    rt.close()

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "time_s", "track_id", "class", "conf",
                    "x1", "y1", "x2", "y2", "cx", "cy", "vx", "vy"])
        w.writerows(rows)

    lat = np.array(per_frame_ms) if per_frame_ms else np.zeros(1)
    summary = {
        "video": str(args.video),
        "resolution": [width, height],
        "source_fps": round(float(src_fps), 2),
        "frames_processed": frame_idx,
        "wall_seconds": round(wall, 2),
        "throughput_fps": round(frame_idx / max(wall, 1e-6), 2),
        "realtime_factor": round((frame_idx / max(wall, 1e-6)) / max(src_fps, 1e-6), 2),
        "latency_ms": {
            "mean": round(float(lat.mean()), 2),
            "p50": round(float(np.percentile(lat, 50)), 2),
            "p95": round(float(np.percentile(lat, 95)), 2),
            "max": round(float(lat.max()), 2),
        },
        "unique_tracks": len(seen_ids),
        "track_rows": len(rows),
        "detector": {"model": getattr(detector, "model_path", "n/a"),
                     "imgsz": getattr(detector, "imgsz", 0),
                     "conf": args.conf, "threads": args.threads},
        "config": {"mode": args.mode,
                   "detect_interval": ("auto" if args.detect_interval <= 0
                                       else args.detect_interval),
                   "final_detect_interval": rt.detect_interval,
                   "flow": not args.no_flow, "max_age": args.max_age,
                   "min_hits": args.min_hits, "track_iou": args.track_iou},
        "pipeline_timing": rt.timing,
        "tracker_stats": tracker.stats,
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    if not args.quiet:
        print(f"{stem}: {frame_idx} frames in {wall:.1f}s -> "
              f"{summary['throughput_fps']} FPS "
              f"({summary['realtime_factor']}x source), "
              f"{len(seen_ids)} unique tracks")
        print(f"  tracks  -> {csv_path}")
        print(f"  summary -> {json_path}")
        if args.out_video:
            print(f"  video   -> {args.out_video}")
    return summary


if __name__ == "__main__":
    main()
