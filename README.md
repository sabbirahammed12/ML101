# RT-Track — real-time tracking of moving objects

**CSE445 · Section 7 · Project Group 7**

| Member | ID |
| --- | --- |
| Sabbir Ahammed | 2131846042 |
| Anika Afrin Toma | 2132041642 |
| Tahsina Mamun Niha | 2131302642 |
| Md. Maruf Parvez | 2211865042 |

## Project topic

> Find 10 videos online showing objects moving from one place to another. Then,
> create a machine learning model that can accurately track the locations of
> these moving things in real time.

This repository is **Phase 1** (the first half of the project, 14 Oct – 15 Dec
2026): the class-agnostic core that finds anything that visibly moves in a video
and keeps a **stable ID** on each object, so its trajectory can be read out
afterwards. It needs no GPU, and it does not care what the object *is*.

### How Phase 1 answers the topic

| The topic asks for | What Phase 1 does |
| --- | --- |
| **10 videos online** of objects moving from one place to another | `python download_videos.py` fetches ten public clips (traffic, pedestrians, a work zone, three conveyor belts) from permissively licensed repositories. Table below; licences in [`SOURCES.md`](SOURCES.md). |
| **A model** that **locates** the moving things | A **motion detector** (MOG2: a Gaussian-mixture background model learned from the video itself) finds the moving objects. A **Kalman filter** per object estimates its position and velocity. |
| **Tracks** them over time | Detections are matched to tracks with the Hungarian algorithm on box overlap (IoU), so each object keeps one ID. |
| **Accurately** | Measured against a synthetic clip with exact ground truth (MOTA, MOTP, IDF1, ID switches), and against ground truth corrupted with controlled detector noise. See [Results](#results). |
| **In real time** | 9 of the 10 clips run at or above real-time speed on a CPU (mean 50.4 FPS). The 1080p clip does not: 0.35×. See [Results](#results). |

A note on the word "model": Phase 1 uses statistical models that learn from the
video as it plays (the background model and the Kalman filter). The
pretrained neural detector (YOLO11n) is planned for Phase 2.

## The ten videos

| # | File | What moves |
| --- | --- | --- |
| 1 | `person-bicycle-car-detection.mp4` | people, cyclists and cars crossing |
| 2 | `car-detection.mp4` | vehicles along a road |
| 3 | `people-detection.mp4` | pedestrians walking through frame |
| 4 | `store-aisle-detection.mp4` | shoppers moving down a shop aisle |
| 5 | `worker-zone-detection.mp4` | workers and vehicles in a work zone (1080p) |
| 6 | `bottle-detection.mp4` | bottles travelling on a conveyor |
| 7 | `bolt-detection.mp4` | bolts travelling on a conveyor |
| 8 | `one-by-one-person-detection.mp4` | people entering one at a time |
| 9 | `fruit-and-vegetable-detection.mp4` | produce travelling on a conveyor |
| 10 | `street-pedestrians-768x576.avi` | pedestrians and vehicles on a campus street |

Clips 1–9 come from `intel-iot-devkit/sample-videos` (CC BY 4.0); clip 10 comes
from `opencv/opencv_extra` (Apache 2.0). The videos are not stored in this
repository; the download script fetches them into `videos/`.

## How it works

```
video ──► motion detector ──► Kalman filter ──► association ──► track IDs
          (MOG2 background    (one per track,   (Hungarian on     │
           subtraction,        predicts where    IoU, gated)      ▼
           class-agnostic)     each box goes)               tracks.csv
                                                            + annotated video
```

**Why motion first.** A pretrained detector can only find what it was trained
on. A moving bolt, a parcel or an animal it has never seen simply does not exist
for it. A background model has no vocabulary: if a pixel region changes against
the static scene, it is reported. Phase 1 builds that class-agnostic core and
the tracker that gives its detections identities.

## Results

Everything below was produced by the scripts in this repository on one CPU-only
machine. Full tables, the method and the caveats are in [`RESULTS.md`](RESULTS.md).

**Accuracy** — synthetic clip, 7 objects, 420 frames, exact ground truth
(detector run on every frame):

| MOTA | IDF1 | MOTP (IoU) | ID switches |
| --- | --- | --- | --- |
| 0.614 | 0.711 | 0.792 | 7 |

The tracker on its own, fed ground truth with simulated detector noise, keeps
MOTA between 0.957 and 0.996, so the main limit is the **detector's recall
(0.738)**: an object that stops is absorbed into the background and is no longer
reported.

**Speed** — ten real clips, 17,306 frames: mean **50.4 FPS**; nine clips run at
1.4× to 11.3× real time. The 1080p, 60 fps clip runs at 0.35× because the motion
detector takes about 37 ms per frame at that size (the tracker takes about 1 ms).

The real clips have no ground truth, so no accuracy figure is claimed for them.

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py                                  # verifies the whole chain
```

## Use

```bash
python download_videos.py                              # ten sample clips -> videos/

# one video, annotated output
python run_tracking.py --video videos/car-detection.mp4 --out-video out/car.mp4

# a whole folder
python run_batch.py videos out

# see only what the detector sees (left: boxes, right: foreground mask)
python view_motion.py --video videos/bolt-detection.mp4 --out-video out/motion.mp4
```

Useful switches for `run_tracking.py`: `--min-area` (smallest moving blob that
counts as an object), `--detect-interval N` (detector every Nth frame, tracks
coast in between), `--max-age`, `--min-hits`, `--track-iou`.

## Output

`<video>_tracks.csv` — one row per object per frame, which is the actual
deliverable: `frame, time_s, track_id, class, conf, x1, y1, x2, y2, cx, cy,
vx, vy`. `cx, cy` is the object's position and `vx, vy` its velocity in
pixels per frame, both from the Kalman state. (`class` is always `object` in
Phase 1.)

`<video>_summary.json` — throughput, per-frame latency percentiles, detector
duty cycle and tracker diagnostics.

## Evaluating it

Stock footage has no annotations, so accuracy is measured against a synthetic
clip whose ground truth is exact (crossing pair, sinusoidal path, stop-and-go
object, full occlusion), and against ground truth corrupted with controlled
detector noise:

```bash
python make_synthetic.py            # clip + ground truth -> data/
python benchmark.py --which all     # MOTA / MOTP / IDF1 / ID switches
python analyze_synthetic.py         # is the error the detector's or the tracker's?
python tests/test_tracker.py        # behaviour tests (also: python -m pytest tests)
python tests/test_metrics.py
```

## Status: Phase 1 and Phase 2

| | Phase 1 (this repository) | Phase 2 (planned) |
| --- | --- | --- |
| Detector | MOG2 motion detector, any moving object | + YOLO11n (ONNX) for named classes, hybrid with motion |
| Between detections | Kalman prediction (coasting) | + Lucas-Kanade optical-flow correction |
| Association | single stage: Hungarian on IoU | two stage (high/low confidence), Mahalanobis gating |
| Pipeline | synchronous, fixed detection interval | auto-tuned duty cycle, async mode |
| Evaluation | synthetic ground truth + detector-noise sweep | + real-footage agreement, throughput study |
| Interface | command line, CSV + annotated video | + live web dashboard |

## Repository layout and who owns what

| Part | Area | Files |
| --- | --- | --- |
| 1 | Motion detection and data | `rttrack/detector.py`, `view_motion.py`, `download_videos.py`, `check_setup.py`, `SOURCES.md` |
| 2 | Tracking core | `rttrack/kalman.py`, `rttrack/tracker.py`, `tests/test_tracker.py` |
| 3 | Pipeline and command line | `rttrack/pipeline.py`, `run_tracking.py`, `run_batch.py` |
| 4 | Evaluation | `rttrack/metrics.py`, `make_synthetic.py`, `benchmark.py`, `analyze_synthetic.py`, `per_object_recall.py`, `tests/test_metrics.py`, `RESULTS.md` |

Each part has a write-up in `docs/` (`part1_…` to `part4_…`) describing what it
does, how it works, and what comes next.

## Known limits of Phase 1

* The camera must be fixed — a moving camera makes the whole scene "move".
* An object that stops is slowly absorbed into the background and disappears
  from the detector (visible in `docs/img/motion_frame200.png`).
* Objects that touch can merge into a single box (same image).
* No object class — everything is `object`.
* Some real clips report many tracks (for example 184 on
  `one-by-one-person-detection.mp4`); the cause has not been investigated yet.

These are the gaps Phase 2 is meant to close.
