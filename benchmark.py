#!/usr/bin/env python3
"""Accuracy benchmarks for the tracker.

Three experiments, because no single one is enough:

1. `synthetic`   -- full pipeline on a clip whose ground truth is known
                    analytically.  Gives honest MOTA / MOTP / IDF1 and ID
                    switch counts, including through a hard occlusion.

2. `noise`       -- feed the tracker ground-truth boxes corrupted with
                    controlled detector noise (jitter, dropped detections,
                    false positives) and sweep the severity.  This isolates
                    the association + Kalman logic from any one detector,
                    and shows where the tracker degrades.

3. `agreement`   -- on the real videos, compare the real-time configuration
                    against a slow, high-quality oracle pass (detector on
                    every frame at 640px).  There is no ground truth here,
                    so this measures how much accuracy the real-time
                    settings give up relative to the best the detector can
                    do on the same footage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rttrack import (MotionDetector, MultiObjectTracker, RealTimeTracker,
                     YoloOnnxDetector)
from rttrack.metrics import centre_error, evaluate
from run_tracking import MOVING_CLASSES


# ----------------------------------------------------------------- utilities
def load_mot_csv(path: str) -> dict[int, list[tuple[int, np.ndarray]]]:
    out: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame"])
            x, y = float(row["x"]), float(row["y"])
            w, h = float(row["w"]), float(row["h"])
            out[f].append((int(row["id"]), np.array([x, y, x + w, y + h])))
    return dict(out)


def run_pipeline(video: str, detector, **kw) -> dict[int, list]:
    """Track a video and return {frame: [(track_id, box)]}."""
    MultiObjectTracker.reset_ids()
    tracker = MultiObjectTracker(**kw.pop("tracker_kw", {}))
    rt = RealTimeTracker(detector, tracker, **kw)
    cap = cv2.VideoCapture(video)
    out: dict[int, list] = {}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = rt.process(frame, f)
        out[f] = [(tid, np.asarray(box)) for tid, _, box, _, _ in res.tracks]
        f += 1
    cap.release()
    rt.close()
    return out


# ------------------------------------------------------------ 1. synthetic
def bench_synthetic(args) -> dict:
    video, gt_csv = args.synthetic_video, args.synthetic_gt
    if not os.path.exists(video):
        import make_synthetic
        os.makedirs(os.path.dirname(os.path.abspath(video)), exist_ok=True)
        make_synthetic.render(video, gt_csv, args.synthetic_frames)

    gt = load_mot_csv(gt_csv)
    results = {}
    for label, kw in [
        ("every_frame", dict(detect_interval=1, use_flow=False)),
        ("interval3_noflow", dict(detect_interval=3, use_flow=False)),
        ("interval3_flow", dict(detect_interval=3, use_flow=True)),
        ("interval5_flow", dict(detect_interval=5, use_flow=True)),
    ]:
        pred = run_pipeline(video, MotionDetector(min_area=250), mode="sync",
                            tracker_kw=dict(min_hits=2, max_age=30), **kw)
        m = evaluate(gt, pred, iou_threshold=0.5)
        m.update({"centre_error": centre_error(gt, pred)})
        results[label] = m
        print(f"  {label:18s} MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f} "
              f"MOTP={m['MOTP_iou']:.3f} IDSW={m['id_switches']:2d} "
              f"FN={m['false_negatives']:4d} FP={m['false_positives']:4d}",
              flush=True)
    return results


# ---------------------------------------------------------------- 2. noise
def bench_noise(args) -> dict:
    """Sweep detector noise, feeding the tracker corrupted GT boxes."""
    gt = load_mot_csv(args.synthetic_gt)
    rng_master = np.random.default_rng(0)
    out = {}

    settings = [
        ("clean",            0.0, 0.00, 0.0),
        ("jitter_5pct",      0.05, 0.00, 0.0),
        ("jitter_10pct",     0.10, 0.00, 0.0),
        ("dropout_20pct",    0.05, 0.20, 0.0),
        ("dropout_40pct",    0.05, 0.40, 0.0),
        ("fp_1_per_frame",   0.05, 0.00, 1.0),
        ("harsh",            0.10, 0.30, 1.0),
    ]
    for label, jitter, dropout, fp_rate in settings:
        rng = np.random.default_rng(rng_master.integers(1 << 30))
        MultiObjectTracker.reset_ids()
        trk = MultiObjectTracker(min_hits=2, max_age=30)
        pred = {}
        for f in sorted(gt):
            dets = []
            for _, box in gt[f]:
                if rng.random() < dropout:
                    continue
                w, h = box[2] - box[0], box[3] - box[1]
                n = rng.normal(0, jitter, 4) * np.array([w, h, w, h])
                b = box + n
                dets.append([b[0], b[1], b[2], b[3],
                             float(np.clip(rng.normal(0.85, 0.07), 0.4, 0.99)), 0])
            if fp_rate and rng.random() < fp_rate:
                x = rng.uniform(0, 900)
                y = rng.uniform(0, 500)
                s = rng.uniform(30, 60)
                dets.append([x, y, x + s, y + s,
                             float(rng.uniform(0.40, 0.62)), 0])
            arr = (np.array(dets, dtype=np.float32) if dets
                   else np.zeros((0, 6), dtype=np.float32))
            trk.update(arr, f)
            pred[f] = [(t.track_id, t.box) for t in trk.live_tracks()]

        m = evaluate(gt, pred, iou_threshold=0.5)
        out[label] = m
        print(f"  {label:16s} MOTA={m['MOTA']:.3f} IDF1={m['IDF1']:.3f} "
              f"IDSW={m['id_switches']:2d} FP={m['false_positives']:4d} "
              f"FN={m['false_negatives']:4d}", flush=True)
    return out


# ------------------------------------------------------------ 3. agreement
def bench_agreement(args) -> dict:
    """Real-time config vs a slow high-quality pass on real footage."""
    out = {}
    videos = sorted([os.path.join(args.video_dir, v)
                     for v in os.listdir(args.video_dir)
                     if v.endswith((".mp4", ".avi"))])[:args.max_videos]

    for v in videos:
        stem = os.path.splitext(os.path.basename(v))[0]
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        limit = min(n, args.agreement_frames)

        oracle_det = YoloOnnxDetector(args.oracle_model, imgsz=640,
                                      conf_threshold=0.25,
                                      classes=MOVING_CLASSES, num_threads=2)
        rt_det = YoloOnnxDetector(args.model, imgsz=416, conf_threshold=0.30,
                                  classes=MOVING_CLASSES, num_threads=2)

        oracle = _run_limited(v, oracle_det, limit, detect_interval=1,
                              use_flow=False)
        realtime = _run_limited(v, rt_det, limit, detect_interval=0,
                                use_flow=True)

        m = evaluate(oracle, realtime, iou_threshold=0.5)
        m["centre_error"] = centre_error(oracle, realtime)

        if m["gt_boxes"] == 0:
            # The oracle found nothing in this window, so there is nothing to
            # agree with -- scoring it would report a meaningless MOTA of 1.
            m["skipped"] = "oracle produced no tracks in this window"
            out[stem] = m
            print(f"  {stem:36s} skipped ({m['skipped']})", flush=True)
            continue

        out[stem] = m
        print(f"  {stem:36s} recall={m['recall']:.3f} prec={m['precision']:.3f} "
              f"MOTP={m['MOTP_iou']:.3f} IDF1={m['IDF1']:.3f} "
              f"IDSW={m['id_switches']} "
              f"centre_err={m['centre_error'].get('median_px', 0):.1f}px",
              flush=True)
    return out


def _run_limited(video: str, detector, limit: int, **kw) -> dict:
    MultiObjectTracker.reset_ids()
    tracker = MultiObjectTracker()
    rt = RealTimeTracker(detector, tracker, mode="sync", **kw)
    cap = cv2.VideoCapture(video)
    out, f = {}, 0
    while f < limit:
        ok, frame = cap.read()
        if not ok:
            break
        res = rt.process(frame, f)
        out[f] = [(tid, np.asarray(box)) for tid, _, box, _, _ in res.tracks]
        f += 1
    cap.release()
    rt.close()
    return out


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "synthetic", "noise", "agreement"])
    ap.add_argument("--model", default="../models/yolo11n_416.onnx")
    ap.add_argument("--oracle-model", default="../models/yolo11n_640.onnx")
    ap.add_argument("--video-dir", default="../videos")
    ap.add_argument("--synthetic-video", default="data/synthetic.mp4")
    ap.add_argument("--synthetic-gt", default="data/synthetic_gt.csv")
    ap.add_argument("--synthetic-frames", type=int, default=420)
    ap.add_argument("--agreement-frames", type=int, default=300)
    ap.add_argument("--max-videos", type=int, default=10)
    ap.add_argument("--out", default="out/benchmark.json")
    args = ap.parse_args()

    report = {}
    if args.which in ("all", "synthetic"):
        print("[1/3] synthetic sequence with exact ground truth", flush=True)
        report["synthetic"] = bench_synthetic(args)
    if args.which in ("all", "noise"):
        print("[2/3] detector-noise robustness sweep", flush=True)
        report["noise"] = bench_noise(args)
    if args.which in ("all", "agreement"):
        print("[3/3] real-time vs oracle agreement on real footage", flush=True)
        report["agreement"] = bench_agreement(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
