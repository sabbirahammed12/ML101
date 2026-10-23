#!/usr/bin/env python3
"""Run the tracker over a folder of videos and collect the results."""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tracking


def main():
    video_dir = sys.argv[1] if len(sys.argv) > 1 else "../videos"
    model = sys.argv[2] if len(sys.argv) > 2 else "../models/yolo11n_416.onnx"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "out"
    os.makedirs(os.path.join(out_dir, "annotated"), exist_ok=True)

    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")) +
                    glob.glob(os.path.join(video_dir, "*.avi")))
    print(f"{len(videos)} videos\n", flush=True)

    results = []
    for i, v in enumerate(videos, 1):
        stem = os.path.splitext(os.path.basename(v))[0]
        print(f"[{i}/{len(videos)}] {stem}", flush=True)
        out_video = os.path.join(out_dir, "annotated", f"{stem}_tracked.mp4")
        summary = run_tracking.main([
            "--video", v, "--model", model, "--out-dir", out_dir,
            "--out-video", out_video, "--quiet",
        ])

        # Nothing found means the objects have no class in the detector's
        # vocabulary (bolts, parcels, luggage on a belt).  Fall back to the
        # model-free motion detector, which does not care what they are.
        if summary["unique_tracks"] == 0:
            print("    no detections from the neural detector -- "
                  "retrying with the motion detector", flush=True)
            summary = run_tracking.main([
                "--video", v, "--out-dir", out_dir,
                "--out-video", out_video, "--quiet",
            ])
            summary["fallback_detector"] = True
        results.append(summary)
        print(f"    {summary['frames_processed']} frames | "
              f"{summary['throughput_fps']} FPS | "
              f"{summary['realtime_factor']}x realtime | "
              f"{summary['unique_tracks']} tracks | "
              f"assoc {summary['tracker_stats']['association_rate']}", flush=True)

    with open(os.path.join(out_dir, "batch_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    total_frames = sum(r["frames_processed"] for r in results)
    total_wall = sum(r["wall_seconds"] for r in results)
    print("\n=== overall ===")
    print(f"videos          : {len(results)}")
    print(f"frames          : {total_frames}")
    print(f"wall seconds    : {total_wall:.1f}")
    print(f"mean throughput : {total_frames / max(total_wall, 1e-6):.1f} FPS")
    print(f"realtime (all)  : "
          f"{sum(r['realtime_factor'] for r in results) / len(results):.2f}x mean")
    print(f"total tracks    : {sum(r['unique_tracks'] for r in results)}")


if __name__ == "__main__":
    main()
