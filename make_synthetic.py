#!/usr/bin/env python3
"""Generate a synthetic tracking sequence with exact ground truth.

Real stock footage has no annotations, so there is nothing to measure
accuracy against.  This builds a clip where every object's position is
known analytically, including the cases that break trackers:

  * straight-line movers at different speeds,
  * a sinusoidal path (constant-velocity models mispredict the turns),
  * two objects that cross -- the classic ID-switch trap,
  * an object that passes behind a static occluder and reappears,
  * an object that stops, waits, then moves off again.

Ground truth is written in MOT format: frame, id, x, y, w, h.
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

W, H = 960, 540


def background(seed: int = 0) -> np.ndarray:
    """Textured background -- flat colour would make this far too easy."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(80, 120, (H // 8, W // 8, 3), dtype=np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    for _ in range(40):
        x, y = rng.integers(0, W), rng.integers(0, H)
        cv2.circle(bg, (int(x), int(y)), int(rng.integers(8, 30)),
                   tuple(int(v) for v in rng.integers(60, 140, 3)), -1)
    return cv2.GaussianBlur(bg, (0, 0), 2)


def trajectories(n_frames: int) -> dict[int, list[tuple[int, np.ndarray]]]:
    """Analytic ground-truth boxes, keyed by object id."""
    out: dict[int, list[tuple[int, np.ndarray]]] = {}

    def add(oid, f, cx, cy, w, h):
        out.setdefault(oid, []).append(
            (f, np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])))

    for f in range(n_frames):
        t = f / n_frames

        # 1: steady left-to-right
        add(1, f, 40 + t * (W - 80), 90, 46, 46)
        # 2: faster, right-to-left, different size
        add(2, f, W - 40 - t * (W - 80) * 1.0, 150, 62, 34)
        # 3: sinusoidal -- punishes a constant-velocity assumption
        add(3, f, 60 + t * (W - 120), 300 + 70 * np.sin(t * 4 * np.pi), 44, 44)
        # 4 and 5: crossing paths, meeting in the middle
        add(4, f, 80 + t * (W - 160), 380 + t * 90, 40, 40)
        add(5, f, W - 80 - t * (W - 160), 470 - t * 90, 40, 40)
        # 6: stop-and-go
        if t < 0.35:
            cx = 60 + (t / 0.35) * 300
        elif t < 0.6:
            cx = 360
        else:
            cx = 360 + ((t - 0.6) / 0.4) * 400
        add(6, f, cx, 230, 48, 48)
        # 7: passes behind the occluder bar and comes out the other side
        add(7, f, 50 + t * (W - 100), 500, 42, 38)
    return out


def render(path: str, csv_path: str, n_frames: int = 420, fps: int = 30,
           occluder: bool = True) -> None:
    gt = trajectories(n_frames)
    bg = background()
    rng = np.random.default_rng(7)
    colours = {i: tuple(int(v) for v in rng.integers(40, 235, 3))
               for i in gt}

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    rows = []
    occ = (int(W * 0.45), 440, int(W * 0.09), 110)      # x, y, w, h

    for f in range(n_frames):
        frame = bg.copy()
        frame = cv2.add(frame, rng.integers(0, 9, frame.shape, dtype=np.uint8))

        for oid, seq in gt.items():
            _, box = seq[f]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            col = colours[oid]
            if oid % 3 == 0:
                cv2.circle(frame, (cx, cy), (x2 - x1) // 2, col, -1)
            elif oid % 3 == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            else:
                pts = np.array([[cx, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                cv2.fillPoly(frame, [pts], col)
            # a little internal texture so optical flow has something to grip
            cv2.line(frame, (x1 + 4, cy), (x2 - 4, cy),
                     tuple(255 - c for c in col), 2)

            visible = True
            if occluder and oid == 7:
                ox, oy, ow, oh = occ
                visible = not (x1 > ox - 10 and x2 < ox + ow + 10)
            if visible:
                rows.append([f, oid, round(x1, 2), round(y1, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)])

        if occluder:
            ox, oy, ow, oh = occ
            cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), (35, 35, 40), -1)

        writer.write(frame)
    writer.release()

    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "id", "x", "y", "w", "h"])
        w.writerows(rows)
    print(f"wrote {path} ({n_frames} frames) and {csv_path} ({len(rows)} boxes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.mp4")
    ap.add_argument("--gt", default="data/synthetic_gt.csv")
    ap.add_argument("--frames", type=int, default=420)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    render(a.out, a.gt, a.frames)
