# Results

Hardware: 2 vCPU, 7 GB RAM, **no GPU**. Detector: YOLO11n exported to ONNX,
416 px input, running on ONNX Runtime CPU with 2 threads.

## Throughput

Tracking only — decode, detect, track, emit coordinates. This is what a
deployment does.

| video | source | resolution | throughput | vs real time | p95 latency | tracks |
| --- | --- | --- | --- | --- | --- | --- |
| bolt-detection * | 30 fps | 1024×576 | 65.9 FPS | **2.20×** | 19.5 ms | 13 |
| bottle-detection | 30 fps | 640×360 | 38.1 FPS | **1.28×** | 33.9 ms | 18 |
| car-detection | 12 fps | 768×432 | 37.4 FPS | **2.99×** | 33.1 ms | 6 |
| fruit-and-vegetable | 60 fps | 960×540 | 75.1 FPS | **1.25×** | 32.1 ms | 22 |
| one-by-one-person | 10 fps | 768×432 | 37.0 FPS | **3.70×** | 34.8 ms | 8 |
| people-detection | 12 fps | 768×432 | 33.1 FPS | **2.76×** | 38.9 ms | 12 |
| person-bicycle-car | 12 fps | 768×432 | 37.1 FPS | **3.10×** | 33.7 ms | 9 |
| store-aisle | 60 fps | 720×404 | 75.6 FPS | **1.26×** | 31.8 ms | 18 |
| street-pedestrians | 25 fps | 768×576 | 33.9 FPS | **1.36×** | 34.7 ms | 12 |
| worker-zone | 60 fps | 1920×1080 | 76.0 FPS | **1.27×** | 32.4 ms | 6 |

\* tracked with the model-free motion detector — bolts have no COCO class.

**17,306 frames, 59.7 FPS mean, 2.16× real time, 10/10 videos at or above
real time, worst p95 latency 35.4 ms.**

Writing an annotated MP4 on top costs real time: with the overlay video
being encoded, the same run averages 52.4 FPS / 1.93×, and the 1080p clip
drops to 0.86× — encoding 1080p is more expensive than tracking it.

## Accuracy

Stock footage has no annotations, so accuracy is measured three ways.

### 1. Tracker in isolation (ground truth + controlled detector noise)

Ground-truth boxes corrupted with jitter, dropped detections and false
positives, fed straight to the tracker. This isolates the association and
motion model from any particular detector.

| detector noise | MOTA | IDF1 | ID switches |
| --- | --- | --- | --- |
| clean | **0.996** | 0.998 | 0 |
| 5% box jitter | 0.996 | 0.894 | 2 |
| 10% box jitter | 0.922 | 0.837 | 30 |
| 20% of detections dropped | 0.994 | 0.894 | 2 |
| 40% of detections dropped | **0.986** | 0.891 | 2 |
| 1 false positive per frame | 0.965 | 0.983 | 0 |
| all three at once | 0.912 | 0.814 | 27 |

Losing 40% of detections costs one point of MOTA: the Kalman filter plus
flow carries tracks through the gaps, and the low-confidence second
association stage recovers them. Heavy box jitter is what hurts — 10%
jitter is larger than the gating tolerance, so identities start to swap.

### 2. End to end on a synthetic clip with exact ground truth

7 objects, 420 frames, including a crossing pair, a sinusoidal path, a
stop-and-go object and a full occlusion. Detector here is the model-free
MOG2 one, since the shapes have no COCO class.

| stage | recall | precision | MOTA | MOTP |
| --- | --- | --- | --- | --- |
| detector alone | 0.728 | 0.830 | 0.562 | 0.667 |
| detector + tracker | 0.735 | 0.809 | 0.558 | 0.670 |

The end-to-end ceiling here is the crude detector, not the tracker: the
tracker reaches the detector's own accuracy while adding stable identities
(8–9 ID switches across 420 frames, through the crossing and the
occlusion). Running the detector every 3rd or 5th frame instead of every
frame changes MOTA by less than 0.01.

### 3. Real-time settings vs a slow high-quality pass, on the real videos

The oracle runs the detector on every frame at 640 px; the real-time
configuration runs at 416 px with an auto-tuned duty cycle. This measures
what the speed is costing on real footage.

| video | recall vs oracle | precision | MOTP | IDF1 | ID switches | median centre error |
| --- | --- | --- | --- | --- | --- | --- |
| bottle-detection | 1.000 | 0.996 | 0.969 | 0.998 | 0 | 0.9 px |
| worker-zone | 1.000 | 0.984 | 0.979 | 0.992 | 0 | 2.6 px |
| people-detection | 0.979 | 0.995 | 0.957 | 0.987 | 0 | 1.2 px |
| person-bicycle-car | 0.936 | 0.850 | 0.951 | 0.891 | 0 | 0.8 px |
| car-detection | 0.909 | 0.865 | 0.918 | 0.887 | 0 | 1.3 px |
| street-pedestrians | 0.899 | 0.946 | 0.913 | 0.837 | 2 | 1.0 px |
| one-by-one-person | 0.776 | 0.950 | 0.945 | 0.855 | 0 | 2.2 px |
| fruit-and-vegetable | 0.364 | 1.000 | 0.741 | 0.533 | 0 | 1.4 px |

Where the objects are a reasonable size, the fast configuration recovers
90–100% of what the slow one finds, localises within 1–3 px of it, and
almost never swaps an identity. The produce conveyor is the honest
failure: at 416 px the small items at the far end of the belt fall below
the detector's threshold, so recall drops to 0.36 — everything it does
report is correct (precision 1.000), it simply sees fewer of them.
`--imgsz 640` recovers them at roughly half the frame rate.

Two videos were excluded from this comparison because the oracle found no
objects in the measured window (bolt-detection has no COCO class at all;
store-aisle is empty for its first 300 frames).

## Known limitations

* Small or distant objects need `--imgsz 640`, which roughly halves speed.
* Box jitter above ~10% of object size starts causing identity swaps.
* The motion-detector fallback assumes a fixed camera, and its precision
  (0.83 on the synthetic clip) is well below the neural detector's.
* COCO classes only; anything outside that vocabulary needs the fallback
  detector or a fine-tuned model.
