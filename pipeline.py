"""Real-time tracking pipeline.

The detector is the bottleneck on a CPU, so running it on every frame caps
the whole system at the detector's rate.  This pipeline decouples the two:

    capture ──► detector worker (own thread, newest-frame-only queue)
            └─► tracker loop (every frame: Kalman predict + optical flow)

The tracker loop never waits for the detector.  Each frame it propagates
every live box with optical flow, so every frame gets a fresh position.
Whenever a detection result becomes available it is folded in as a
correction -- associated against the tracks' states, back-dated to the
frame it was computed on.  Throughput is then set by the cheap per-frame
work, while accuracy is still anchored by the detector.

`mode="sync"` runs the detector inline every `detect_interval` frames and
is used as the reference/oracle configuration for accuracy measurement.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .flow import FlowPropagator
from .tracker import MultiObjectTracker


@dataclass
class FrameResult:
    """Everything the pipeline knows about one frame."""
    frame_idx: int
    timestamp: float
    tracks: list[tuple] = field(default_factory=list)  # (id, cls, box, score, vel)
    detector_ran: bool = False
    latency_ms: float = 0.0


class _DetectorWorker(threading.Thread):
    """Runs the detector on the most recent frame handed to it, forever."""

    def __init__(self, detector, downscale: float = 1.0):
        super().__init__(daemon=True)
        self.detector = detector
        self.downscale = float(downscale)
        self._in: queue.Queue = queue.Queue(maxsize=1)
        self._out: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self.n_runs = 0
        self.total_ms = 0.0

    def submit(self, frame_idx: int, frame: np.ndarray) -> bool:
        """Offer a frame, replacing any frame still waiting.

        The queue holds at most one frame and it is always the newest one:
        detecting on a frame that is already several frames stale only adds
        lag, and the tracker has to undo it afterwards.
        """
        while True:
            try:
                self._in.put_nowait((frame_idx, frame))
                return True
            except queue.Full:
                try:
                    self._in.get_nowait()        # drop the stale frame
                except queue.Empty:
                    pass

    def poll(self):
        try:
            return self._out.get_nowait()
        except queue.Empty:
            return None

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                frame_idx, frame = self._in.get(timeout=0.05)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            if self.downscale != 1.0:
                small = cv2.resize(frame, None, fx=self.downscale,
                                   fy=self.downscale, interpolation=cv2.INTER_AREA)
                dets = self.detector.detect(small)
                if len(dets):
                    dets = dets.copy()
                    dets[:, :4] /= self.downscale
            else:
                dets = self.detector.detect(frame)
            dt = (time.perf_counter() - t0) * 1000.0
            self.n_runs += 1
            self.total_ms += dt
            self._out.put((frame_idx, dets, dt))

    def stop(self) -> None:
        self._stop.set()


class RealTimeTracker:
    """Detector + tracker + flow propagation, wired for real-time use."""

    def __init__(self, detector, tracker: MultiObjectTracker | None = None,
                 mode: str = "sync", detect_interval: int = 0,
                 use_flow: bool = True, detector_downscale: float = 1.0,
                 target_fps: float = 30.0, max_interval: int = 8):
        """`detect_interval=0` auto-tunes the detector duty cycle.

        The interval is chosen from the measured cost of the detector and
        the frame budget implied by `target_fps`, so the same code keeps up
        with the stream on a slow CPU (detect every few frames, track with
        flow in between) and detects on every frame where there is headroom.
        """
        if mode not in ("async", "sync"):
            raise ValueError("mode must be 'async' or 'sync'")
        self.detector = detector
        self.tracker = tracker or MultiObjectTracker()
        self.mode = mode
        self.auto_interval = int(detect_interval) <= 0
        self.detect_interval = 1 if self.auto_interval else int(detect_interval)
        self.target_fps = float(target_fps)
        self.max_interval = int(max_interval)
        self._lat_ema: float | None = None
        self._since_tune = 0
        self.tune_every = 12
        self._next_detect_frame = 0
        self._last_entry: float | None = None
        self._sync_coast_window()
        self.use_flow = bool(use_flow)
        self.flow = FlowPropagator() if use_flow else None
        self.worker = (_DetectorWorker(detector, detector_downscale)
                       if mode == "async" else None)
        if self.worker:
            self.worker.start()

        self.frames_seen = 0
        self.detector_runs = 0
        self.detector_ms = 0.0
        self.tracker_ms = 0.0
        self.flow_ms = 0.0
        self.detector_lag: list[int] = []

    # ------------------------------------------------------------ per frame
    def process(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        t_start = time.perf_counter()
        if self._last_entry is not None:
            # Time since the previous call: the true per-frame cost, including
            # the caller's own decode and output work.
            self._tune_interval((t_start - self._last_entry) * 1000.0)
        self._last_entry = t_start
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if self.use_flow else None
        detector_ran = False

        if self.mode == "sync":
            if frame_idx >= self._next_detect_frame:
                t0 = time.perf_counter()
                dets = self.detector.detect(frame)
                det_ms = (time.perf_counter() - t0) * 1000.0
                self.detector_ms += det_ms
                self.detector_runs += 1
                detector_ran = True
                self._next_detect_frame = frame_idx + self.detect_interval
                if self.use_flow:
                    self.flow.set_frame(gray)
                t0 = time.perf_counter()
                self.tracker.update(dets, frame_idx,
                                    dt_scale=float(self.detect_interval))
                self.tracker_ms += (time.perf_counter() - t0) * 1000.0
            else:
                self._flow_step(gray, frame_idx)
        else:
            # Hand the newest frame to the worker if it is free.
            self.worker.submit(frame_idx, frame)

            result = self.worker.poll()
            if result is not None:
                src_idx, dets, dt = result
                self.detector_ms += dt
                self.detector_runs += 1
                detector_ran = True
                lag = max(0, frame_idx - src_idx)
                self.detector_lag.append(lag)
                t0 = time.perf_counter()
                # Correct with the detection, accounting for how many frames
                # elapsed while the detector was thinking.
                self.tracker.update(dets, frame_idx, dt_scale=1.0, lag=lag)
                self.tracker_ms += (time.perf_counter() - t0) * 1000.0
                if self.use_flow:
                    self.flow.set_frame(gray)
                self._last_lag = lag
            else:
                self._flow_step(gray, frame_idx)

        self.frames_seen += 1
        tracks = [(t.track_id, t.class_id, t.box, t.score, t.velocity)
                  for t in self.tracker.live_tracks()]
        for t in self.tracker.live_tracks():
            t.record(frame_idx)
        return FrameResult(frame_idx=frame_idx, timestamp=time.time(),
                           tracks=tracks, detector_ran=detector_ran,
                           latency_ms=(time.perf_counter() - t_start) * 1000.0)

    # --------------------------------------------------------------- helpers
    def _tune_interval(self, loop_ms: float) -> None:
        """Adjust the detector duty cycle from measured end-to-end frame time.

        This is a closed loop on what the caller actually experiences --
        decode, detection, tracking and whatever it does with the result --
        rather than an estimate built from the detector cost alone, which
        misses everything outside this class.
        """
        if not self.auto_interval:
            return
        self._lat_ema = (loop_ms if self._lat_ema is None
                         else 0.85 * self._lat_ema + 0.15 * loop_ms)
        self._since_tune += 1
        if self._since_tune < self.tune_every:
            return

        budget_ms = 1000.0 / max(self.target_fps, 1e-6)
        if self._lat_ema > budget_ms and self.detect_interval < self.max_interval:
            self.detect_interval += 1
        elif (self._lat_ema < 0.60 * budget_ms and self.detect_interval > 1):
            self.detect_interval -= 1
        else:
            return
        self._since_tune = 0
        self._lat_ema = None
        self._sync_coast_window()

    def _sync_coast_window(self) -> None:
        """Keep the reporting window in step with the detector duty cycle.

        `time_since_update` counts frames, not detector runs, so the window
        has to scale with the interval or every flow-only frame would look
        like an unsupported coast.
        """
        self.tracker.max_coast_report = max(4, 2 * self.detect_interval + 2)

    def _flow_step(self, gray, frame_idx: int) -> None:
        live = self.tracker.tracks
        if self.use_flow and live:
            t0 = time.perf_counter()
            boxes = [t.box for t in live]
            propagated = self.flow.propagate(gray, boxes)
            # Keyed by position in the track list, not by track_id: tracks
            # that are still tentative have no public ID yet.
            flow_boxes = {i: b for i, b in enumerate(propagated) if b is not None}
            self.flow_ms += (time.perf_counter() - t0) * 1000.0
        else:
            flow_boxes = {}
            if self.use_flow:
                self.flow.set_frame(gray)
        t0 = time.perf_counter()
        self.tracker.update_with_flow(flow_boxes, frame_idx)
        self.tracker_ms += (time.perf_counter() - t0) * 1000.0

    def close(self) -> None:
        if self.worker:
            self.worker.stop()

    # ----------------------------------------------------------- diagnostics
    @property
    def timing(self) -> dict:
        n = max(self.frames_seen, 1)
        return {
            "frames": self.frames_seen,
            "detector_runs": self.detector_runs,
            "detector_duty_cycle": round(self.detector_runs / n, 4),
            "mean_detector_ms": round(self.detector_ms / max(self.detector_runs, 1), 2),
            "mean_tracker_ms_per_frame": round(self.tracker_ms / n, 3),
            "mean_flow_ms_per_frame": round(self.flow_ms / n, 3),
            "mean_detector_lag_frames": (
                round(float(np.mean(self.detector_lag)), 2)
                if self.detector_lag else 0.0),
        }
