"""Multi-object tracker: Kalman motion model + two-stage association.

This is a tracking-by-detection tracker in the SORT/ByteTrack family, with
the pieces that matter for keeping identities stable written out explicitly:

  1. every live track predicts its next box with a Kalman filter;
  2. high-confidence detections are matched to confirmed tracks first,
     using an IoU cost gated by Mahalanobis distance, solved optimally with
     the Hungarian algorithm;
  3. tracks left unmatched get a second chance against the low-confidence
     detections that stage 1 ignored -- this is what recovers partially
     occluded objects instead of killing their ID (the ByteTrack idea);
  4. still-unmatched detections start tentative tracks, which must be seen
     `min_hits` times before they are reported, so detector flicker does
     not create ghost IDs;
  5. unmatched tracks coast on prediction for up to `max_age` frames before
     being deleted, which bridges short occlusions.

Between detector runs the pipeline feeds flow-measured boxes in through
`update_with_flow`, so reported positions stay current at full frame rate.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import CHI2_INV95, KalmanBoxFilter, xyah_to_xyxy, xyxy_to_xyah

# Flow pseudo-measurements are trusted for position but not for size.
FLOW_NOISE_SCALE = 2.5       # inflate measurement noise vs. a real detection
FLOW_MAX_SCALE_STEP = 0.02   # at most 2% box-size change per flow-only frame
BOX_HISTORY = 64             # frames of box history kept for back-dating


# ------------------------------------------------------------------- helpers
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _hungarian(cost: np.ndarray, max_cost: float):
    """Optimal assignment, dropping pairs whose cost exceeds `max_cost`."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    rows, cols = linear_sum_assignment(cost)
    matches, un_rows, un_cols = [], [], []
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    un_rows = [r for r in range(cost.shape[0]) if r not in matched_r]
    un_cols = [c for c in range(cost.shape[1]) if c not in matched_c]
    return matches, un_rows, un_cols


# --------------------------------------------------------------------- track
class Track:
    """One tracked object: Kalman state, identity, class vote and history."""

    _next_id = 1

    __slots__ = ("mean", "cov", "track_id", "class_id", "score", "hits",
                 "age", "time_since_update", "state", "history",
                 "_class_votes", "start_frame", "last_frame", "_box_hist")

    def __init__(self, kf: KalmanBoxFilter, box: np.ndarray, score: float,
                 class_id: int, frame_idx: int, min_hits: int):
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        # A public ID is only minted once the track is confirmed, so that
        # detector flicker never burns through the ID space.
        self.track_id = -1
        self.class_id = int(class_id)
        self.score = float(score)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = "tentative"
        if min_hits <= 1:
            self.confirm()
        self.history: list[tuple[int, np.ndarray]] = []
        self._class_votes = {int(class_id): 1.0}
        self.start_frame = int(frame_idx)
        self.last_frame = int(frame_idx)
        self._box_hist: deque = deque(maxlen=BOX_HISTORY)
        self._box_hist.append((int(frame_idx), self.box))

    # ------------------------------------------------------------ properties
    @property
    def box(self) -> np.ndarray:
        return xyah_to_xyxy(self.mean)

    @property
    def centre(self) -> tuple[float, float]:
        return float(self.mean[0]), float(self.mean[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.mean[4]), float(self.mean[5])

    @property
    def is_confirmed(self) -> bool:
        return self.state == "confirmed"

    # ---------------------------------------------------------------- steps
    def predict(self, kf: KalmanBoxFilter, dt_scale: float = 1.0) -> None:
        self.mean, self.cov = kf.predict(self.mean, self.cov, dt_scale)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanBoxFilter, box: np.ndarray, score: float,
               class_id: int, frame_idx: int, min_hits: int,
               is_detection: bool = True) -> None:
        meas = xyxy_to_xyah(box)
        if is_detection:
            noise_scale = 1.0
        else:
            # Flow gives a reliable translation but a noisy size estimate, so
            # take its centre and let the box size follow only gently.
            noise_scale = FLOW_NOISE_SCALE
            scale = float(np.clip(meas[3] / max(self.mean[3], 1e-6),
                                  1.0 - FLOW_MAX_SCALE_STEP,
                                  1.0 + FLOW_MAX_SCALE_STEP))
            meas = np.array([meas[0], meas[1], self.mean[2],
                             self.mean[3] * scale])
        self.mean, self.cov = kf.update(self.mean, self.cov, meas, noise_scale)
        self.last_frame = int(frame_idx)
        self._box_hist.append((int(frame_idx), self.box))
        if is_detection:
            self.hits += 1
            self.time_since_update = 0
            self.score = float(score)
            self._class_votes[int(class_id)] = (
                self._class_votes.get(int(class_id), 0.0) + float(score))
            self.class_id = max(self._class_votes, key=self._class_votes.get)
            if self.state == "tentative" and self.hits >= min_hits:
                self.confirm()
        else:
            # Flow measurement keeps the position fresh but is not evidence
            # that the object is still really there, so `time_since_update`
            # (which drives deletion) is deliberately left alone.
            self.score = float(self.score)

    def record(self, frame_idx: int) -> None:
        self.history.append((int(frame_idx), self.box.copy()))

    def note_state(self, frame_idx: int) -> None:
        """Remember where this track was at `frame_idx` (for back-dating)."""
        self._box_hist.append((int(frame_idx), self.box))

    def box_at(self, frame_idx: int) -> np.ndarray:
        """The box this track held at `frame_idx`, or its current box."""
        for f, b in reversed(self._box_hist):
            if f <= frame_idx:
                return b
        return self.box

    def confirm(self) -> None:
        """Promote to a reported track and mint its public ID."""
        self.state = "confirmed"
        if self.track_id < 0:
            self.track_id = Track._next_id
            Track._next_id += 1

    def mark_missed(self, tentative_max_age: int = 10) -> None:
        """A tentative track survives a few misses before being dropped.

        The detector does not run on every frame, so insisting that a new
        object be hit on consecutive detector steps would delete and
        recreate it repeatedly -- which is exactly how ID churn starts.
        """
        if self.state == "tentative" and self.time_since_update > tentative_max_age:
            self.state = "deleted"


# ------------------------------------------------------------------- tracker
class MultiObjectTracker:
    """Tracking-by-detection multi-object tracker with persistent IDs."""

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.30, low_conf: float = 0.10,
                 high_conf: float = 0.40, gate_mahalanobis: bool = True,
                 max_centre_jump: float = 3.5, tentative_max_age: int = 10,
                 duplicate_iou: float = 0.45, max_coast_report: int = 4):
        self.kf = KalmanBoxFilter()
        self.tracks: list[Track] = []
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.low_conf = float(low_conf)
        self.high_conf = float(high_conf)
        self.gate = bool(gate_mahalanobis)
        self.max_centre_jump = float(max_centre_jump)
        self.tentative_max_age = int(tentative_max_age)
        self.duplicate_iou = float(duplicate_iou)
        self.max_coast_report = int(max_coast_report)
        self.frame_count = 0
        self._lag = 0
        self._assoc_frame = 0
        # diagnostics
        self.n_matches = 0
        self.n_det_total = 0
        self.n_new_tracks = 0
        self.n_recovered_low_conf = 0
        self.n_duplicates_suppressed = 0

    # ------------------------------------------------------------ main entry
    def update(self, detections: np.ndarray, frame_idx: int | None = None,
               dt_scale: float = 1.0, lag: int = 0) -> list[Track]:
        """Advance the tracker one detector step and return live tracks.

        `lag` is how many frames old the detections are (non-zero when the
        detector runs asynchronously).  Association is then done against
        where each track *was* when the detection was computed, and the
        detection is carried forward by the track's velocity before it is
        used to correct the current state.
        """
        self.frame_count += 1
        if frame_idx is None:
            frame_idx = self.frame_count

        dets = (np.zeros((0, 6), dtype=np.float32) if detections is None
                or len(detections) == 0 else np.asarray(detections, dtype=np.float64))
        self.n_det_total += len(dets)

        for t in self.tracks:
            t.predict(self.kf, dt_scale)
            t.note_state(frame_idx)
        self._lag = max(0, int(lag))
        self._assoc_frame = int(frame_idx) - self._lag

        high = dets[dets[:, 4] >= self.high_conf] if len(dets) else dets
        low = (dets[(dets[:, 4] < self.high_conf) & (dets[:, 4] >= self.low_conf)]
               if len(dets) else dets)

        confirmed = [i for i, t in enumerate(self.tracks) if t.is_confirmed]
        tentative = [i for i, t in enumerate(self.tracks) if not t.is_confirmed]

        # ---- stage 1: confirmed + tentative tracks vs high-conf detections
        pool = confirmed + tentative
        m1, un_tracks, un_dets = self._associate(pool, high)
        for ti, di in m1:
            trk = self.tracks[pool[ti]]
            trk.update(self.kf, self._compensate(trk, high[di, :4]),
                       high[di, 4], int(high[di, 5]), frame_idx, self.min_hits)
            self.n_matches += 1

        # ---- stage 2: leftover confirmed tracks vs low-conf detections
        left = [pool[i] for i in un_tracks
                if self.tracks[pool[i]].is_confirmed]
        if len(left) and len(low):
            m2, un2, _ = self._associate(left, low, iou_threshold=0.20,
                                         use_gate=False)
            for ti, di in m2:
                trk = self.tracks[left[ti]]
                trk.update(self.kf, self._compensate(trk, low[di, :4]),
                           low[di, 4], int(low[di, 5]), frame_idx,
                           self.min_hits)
                self.n_matches += 1
                self.n_recovered_low_conf += 1
            still_missing = [left[i] for i in un2]
        else:
            still_missing = left

        tentative_missing = [pool[i] for i in un_tracks
                             if not self.tracks[pool[i]].is_confirmed]
        for idx in still_missing + tentative_missing:
            self.tracks[idx].mark_missed(self.tentative_max_age)

        # ---- new tracks from unmatched high-confidence detections
        #
        # A detection that still overlaps an existing track is almost always
        # that same object with a failed association, not a new one.  Giving
        # it an ID of its own would spawn a duplicate, which then competes
        # for future detections and makes the next association worse -- so
        # suppress it and let the existing track recover instead.
        live_boxes = np.array([t.box for t in self.tracks]) if self.tracks else None
        for di in un_dets:
            if high[di, 4] < self.high_conf:
                continue
            if live_boxes is not None and len(live_boxes):
                overlap = iou_matrix(high[di:di + 1, :4], live_boxes)
                if overlap.size and overlap.max() > self.duplicate_iou:
                    self.n_duplicates_suppressed += 1
                    continue
            self.tracks.append(Track(self.kf, high[di, :4], high[di, 4],
                                     int(high[di, 5]), frame_idx,
                                     self.min_hits))
            self.n_new_tracks += 1

        self.tracks = [t for t in self.tracks
                       if t.state != "deleted" and t.time_since_update <= self.max_age]
        return self.live_tracks()

    # ----------------------------------------------------- flow-only update
    def update_with_flow(self, flow_boxes: dict[int, np.ndarray],
                         frame_idx: int) -> list[Track]:
        """Advance one frame using flow pseudo-measurements (no detector).

        `flow_boxes` maps an index into `self.tracks` to its propagated box.
        """
        self.frame_count += 1
        for i, t in enumerate(self.tracks):
            t.predict(self.kf, 1.0)
            fb = flow_boxes.get(i)
            if fb is not None:
                t.update(self.kf, fb, t.score, t.class_id, frame_idx,
                         self.min_hits, is_detection=False)
            else:
                t.note_state(frame_idx)
        self.tracks = [t for t in self.tracks
                       if t.state != "deleted" and t.time_since_update <= self.max_age]
        return self.live_tracks()

    # ------------------------------------------------------------- internals
    def _associate(self, track_indices: list[int], dets: np.ndarray,
                   iou_threshold: float | None = None, use_gate: bool = True):
        if len(track_indices) == 0 or len(dets) == 0:
            return [], list(range(len(track_indices))), list(range(len(dets)))

        thr = self.iou_threshold if iou_threshold is None else iou_threshold
        # Compare against where each track was when these pixels were seen.
        tboxes = np.array([self.tracks[i].box_at(self._assoc_frame)
                           for i in track_indices])
        cost = 1.0 - iou_matrix(tboxes, dets[:, :4])

        meas = np.array([xyxy_to_xyah(d) for d in dets[:, :4]])
        for r, ti in enumerate(track_indices):
            t = self.tracks[ti]
            tc = ((tboxes[r][0] + tboxes[r][2]) / 2.0,
                  (tboxes[r][1] + tboxes[r][3]) / 2.0)
            # class consistency: switching an object's class mid-track is
            # usually a detector error, so make such pairings expensive.
            cost[r][dets[:, 5].astype(int) != t.class_id] += 0.25

            if use_gate and self.gate:
                d2 = self.kf.gating_distance(t.mean, t.cov, meas)
                cost[r][d2 > CHI2_INV95[4] * 3.0] = 1e5

            # A confirmed track should not teleport: cap the plausible jump
            # at a few box-heights, scaled by how long it has been unseen.
            h = max(t.mean[3], 1.0)
            limit = self.max_centre_jump * h * (1 + 0.5 * t.time_since_update)
            dist = np.linalg.norm(meas[:, :2] - np.array(tc), axis=1)
            cost[r][dist > limit] = 1e5

        return _hungarian(cost, max_cost=1.0 - thr)

    def _compensate(self, track: Track, box: np.ndarray) -> np.ndarray:
        """Carry a stale detection forward to the current frame."""
        if self._lag <= 0:
            return box
        vx, vy = track.velocity
        shift = np.array([vx, vy, vx, vy], dtype=np.float64) * self._lag
        return np.asarray(box, dtype=np.float64) + shift

    def live_tracks(self) -> list[Track]:
        """Confirmed tracks that still have recent detector support.

        A track is kept alive internally for `max_age` frames so that it can
        reclaim its ID after an occlusion, but it is only *reported* while a
        detection has backed it recently.  Reporting a long unsupported coast
        means emitting a box for an object that has left -- a false positive
        for every frame it persists.
        """
        return [t for t in self.tracks
                if t.is_confirmed and t.time_since_update <= self.max_coast_report]

    @property
    def stats(self) -> dict:
        return {
            "detections_seen": int(self.n_det_total),
            "associations": int(self.n_matches),
            "tracks_created": int(self.n_new_tracks),
            "low_conf_recoveries": int(self.n_recovered_low_conf),
            "duplicates_suppressed": int(self.n_duplicates_suppressed),
            "association_rate": round(self.n_matches / max(self.n_det_total, 1), 4),
        }

    @staticmethod
    def reset_ids() -> None:
        Track._next_id = 1
