"""CLEAR-MOT and ID metrics for evaluating tracking output.

Implements the standard measures used on the MOTChallenge benchmarks:

    MOTA  = 1 - (FN + FP + IDSW) / |GT|      -- overall accuracy
    MOTP  = mean IoU over matched pairs      -- localisation quality
    IDF1  = 2 IDTP / (2 IDTP + IDFP + IDFN)  -- identity consistency

Per-frame matching is an optimal (Hungarian) assignment on IoU, accepting
pairs at IoU >= `iou_threshold`, with the usual preference for keeping an
existing GT-to-prediction pairing so that an ID switch is only counted when
the tracker really does swap identities.

IDF1 is computed globally: the best one-to-one mapping between GT IDs and
predicted IDs over the whole sequence, which is what distinguishes a
tracker that holds one ID throughout from one that re-labels constantly.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker import iou_matrix

Frames = dict[int, list[tuple[int, np.ndarray]]]   # frame -> [(id, box)]


def _match(gt: list, pred: list, iou_threshold: float,
           previous: dict[int, int]):
    """Optimal IoU matching for one frame, biased toward existing pairs."""
    if not gt or not pred:
        return [], list(range(len(gt))), list(range(len(pred)))

    gb = np.array([b for _, b in gt], dtype=np.float64)
    pb = np.array([b for _, b in pred], dtype=np.float64)
    iou = iou_matrix(gb, pb)

    cost = 1.0 - iou
    # Small bonus for a pairing that already held last frame: with equal
    # overlap, continuing an identity beats swapping it.
    for i, (gid, _) in enumerate(gt):
        if gid in previous:
            for j, (pid, _) in enumerate(pred):
                if pid == previous[gid]:
                    cost[i, j] -= 0.02

    rows, cols = linear_sum_assignment(cost)
    matches = [(int(r), int(c)) for r, c in zip(rows, cols)
               if iou[r, c] >= iou_threshold]
    mr = {r for r, _ in matches}
    mc = {c for _, c in matches}
    return (matches,
            [i for i in range(len(gt)) if i not in mr],
            [j for j in range(len(pred)) if j not in mc])


def evaluate(gt_frames: Frames, pred_frames: Frames,
             iou_threshold: float = 0.5) -> dict:
    """Compare predicted tracks against ground truth."""
    frames = sorted(set(gt_frames) | set(pred_frames))

    n_gt = n_fp = n_fn = n_idsw = 0
    iou_sum = 0.0
    n_matched = 0
    last_pairing: dict[int, int] = {}
    co_occurrence: dict[tuple[int, int], int] = defaultdict(int)
    gt_counts: dict[int, int] = defaultdict(int)
    pred_counts: dict[int, int] = defaultdict(int)

    for f in frames:
        gt = gt_frames.get(f, [])
        pred = pred_frames.get(f, [])
        n_gt += len(gt)
        for gid, _ in gt:
            gt_counts[gid] += 1
        for pid, _ in pred:
            pred_counts[pid] += 1

        matches, un_gt, un_pred = _match(gt, pred, iou_threshold, last_pairing)

        for gi, pj in matches:
            gid, gbox = gt[gi]
            pid, pbox = pred[pj]
            iou_sum += float(iou_matrix(gbox[None], pbox[None])[0, 0])
            n_matched += 1
            co_occurrence[(gid, pid)] += 1
            if gid in last_pairing and last_pairing[gid] != pid:
                n_idsw += 1
            last_pairing[gid] = pid

        n_fn += len(un_gt)
        n_fp += len(un_pred)

    # ---- IDF1: best global one-to-one GT-ID <-> pred-ID assignment
    gids = sorted(gt_counts)
    pids = sorted(pred_counts)
    idtp = 0
    if gids and pids:
        overlap = np.zeros((len(gids), len(pids)))
        gi = {g: i for i, g in enumerate(gids)}
        pi = {p: i for i, p in enumerate(pids)}
        for (g, p), c in co_occurrence.items():
            overlap[gi[g], pi[p]] = c
        r, c = linear_sum_assignment(-overlap)
        idtp = int(overlap[r, c].sum())
    idfn = sum(gt_counts.values()) - idtp
    idfp = sum(pred_counts.values()) - idtp
    idf1 = 2 * idtp / max(2 * idtp + idfp + idfn, 1)

    mota = 1.0 - (n_fn + n_fp + n_idsw) / max(n_gt, 1)
    motp = iou_sum / max(n_matched, 1)
    recall = n_matched / max(n_gt, 1)
    precision = n_matched / max(n_matched + n_fp, 1)

    return {
        "frames": len(frames),
        "gt_boxes": int(n_gt),
        "matched": int(n_matched),
        "false_negatives": int(n_fn),
        "false_positives": int(n_fp),
        "id_switches": int(n_idsw),
        "MOTA": round(float(mota), 4),
        "MOTP_iou": round(float(motp), 4),
        "IDF1": round(float(idf1), 4),
        "recall": round(float(recall), 4),
        "precision": round(float(precision), 4),
        "gt_ids": len(gt_counts),
        "pred_ids": len(pred_counts),
    }


def centre_error(gt_frames: Frames, pred_frames: Frames,
                 iou_threshold: float = 0.3) -> dict:
    """Localisation error in pixels over matched pairs."""
    errs, diags = [], []
    for f in sorted(set(gt_frames) & set(pred_frames)):
        gt, pred = gt_frames[f], pred_frames[f]
        matches, _, _ = _match(gt, pred, iou_threshold, {})
        for gi, pj in matches:
            gb, pb = gt[gi][1], pred[pj][1]
            gc = np.array([(gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2])
            pc = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
            errs.append(float(np.linalg.norm(gc - pc)))
            diags.append(float(np.hypot(gb[2] - gb[0], gb[3] - gb[1])))
    if not errs:
        return {"n": 0}
    errs = np.array(errs)
    return {
        "n": int(len(errs)),
        "mean_px": round(float(errs.mean()), 2),
        "median_px": round(float(np.median(errs)), 2),
        "p95_px": round(float(np.percentile(errs, 95)), 2),
        "mean_normalised": round(float((errs / np.array(diags)).mean()), 4),
    }
