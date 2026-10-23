"""Kalman filter for 2-D bounding-box motion.

State (8-dim):  [cx, cy, a, h, vcx, vcy, va, vh]
    cx, cy : box centre in pixels
    a      : aspect ratio (w / h), assumed slowly varying
    h      : box height in pixels
    v*     : per-frame velocity of each of the above

Measurement (4-dim): [cx, cy, a, h]

Constant-velocity process model.  Process and measurement noise are scaled
by the current box height, so that large (near) objects are allowed to move
further per frame than small (far) ones -- the standard trick from SORT /
DeepSORT, which keeps a single parameter set working across scales.
"""

from __future__ import annotations

import numpy as np

# chi-square 0.95 quantiles for N degrees of freedom (gating on Mahalanobis d^2)
CHI2_INV95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877}


class KalmanBoxFilter:
    """Constant-velocity Kalman filter over [cx, cy, a, h]."""

    ndim = 4

    def __init__(self, std_weight_position: float = 1.0 / 20,
                 std_weight_velocity: float = 1.0 / 160):
        ndim, dt = self.ndim, 1.0

        # x_{k+1} = F x_k ;  z_k = H x_k
        self._F = np.eye(2 * ndim, dtype=np.float64)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float64)

        self._std_pos = std_weight_position
        self._std_vel = std_weight_velocity

    # ------------------------------------------------------------------ init
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a track state from an unassociated measurement."""
        mean_pos = np.asarray(measurement, dtype=np.float64)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = measurement[3]
        std = np.array([
            2 * self._std_pos * h,      # cx
            2 * self._std_pos * h,      # cy
            1e-2,                       # a
            2 * self._std_pos * h,      # h
            10 * self._std_vel * h,     # vcx
            10 * self._std_vel * h,     # vcy
            1e-5,                       # va
            10 * self._std_vel * h,     # vh
        ])
        covariance = np.diag(np.square(std))
        return mean, covariance

    # --------------------------------------------------------------- predict
    def predict(self, mean: np.ndarray, covariance: np.ndarray,
                dt_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the state one step (or `dt_scale` steps) forward."""
        h = max(mean[3], 1.0)
        std_pos = np.array([self._std_pos * h, self._std_pos * h, 1e-2,
                            self._std_pos * h])
        std_vel = np.array([self._std_vel * h, self._std_vel * h, 1e-5,
                            self._std_vel * h])
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]) * dt_scale)

        F = self._F
        if dt_scale != 1.0:
            F = self._F.copy()
            for i in range(self.ndim):
                F[i, self.ndim + i] = dt_scale

        mean = F @ mean
        covariance = F @ covariance @ F.T + motion_cov
        return mean, covariance

    # -------------------------------------------------------------- project
    def project(self, mean: np.ndarray, covariance: np.ndarray,
                noise_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Map a state distribution into measurement space.

        `noise_scale` inflates the measurement noise: flow pseudo-measurements
        are less trustworthy than detections and are fed in with a larger
        value, so they nudge the state instead of overwriting it.
        """
        h = max(mean[3], 1.0)
        std = np.array([self._std_pos * h, self._std_pos * h, 1e-1,
                        self._std_pos * h]) * float(noise_scale)
        innovation_cov = np.diag(np.square(std))
        mean_proj = self._H @ mean
        cov_proj = self._H @ covariance @ self._H.T + innovation_cov
        return mean_proj, cov_proj

    # --------------------------------------------------------------- update
    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray,
               noise_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Correct the state with an associated measurement."""
        proj_mean, proj_cov = self.project(mean, covariance, noise_scale)

        # Kalman gain via a solve rather than an explicit inverse.
        kalman_gain = np.linalg.solve(
            proj_cov.T, (covariance @ self._H.T).T).T
        innovation = np.asarray(measurement, dtype=np.float64) - proj_mean

        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov

    # ------------------------------------------------------------- distance
    def gating_distance(self, mean: np.ndarray, covariance: np.ndarray,
                        measurements: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance from this state to each measurement."""
        proj_mean, proj_cov = self.project(mean, covariance)
        if len(measurements) == 0:
            return np.zeros(0)
        d = np.asarray(measurements, dtype=np.float64) - proj_mean
        try:
            chol = np.linalg.cholesky(proj_cov)
            z = np.linalg.solve(chol, d.T)
            return np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:          # pragma: no cover - ill-conditioned
            inv = np.linalg.pinv(proj_cov)
            return np.einsum("ij,jk,ik->i", d, inv, d)


# ---------------------------------------------------------------- conversions
def xyxy_to_xyah(box: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, aspect, height]."""
    x1, y1, x2, y2 = box[:4]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    return np.array([x1 + w / 2.0, y1 + h / 2.0, w / h, h], dtype=np.float64)


def xyah_to_xyxy(state: np.ndarray) -> np.ndarray:
    """[cx, cy, aspect, height] -> [x1, y1, x2, y2]."""
    cx, cy, a, h = state[:4]
    h = max(float(h), 1e-6)
    w = max(float(a) * h, 1e-6)
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                    dtype=np.float64)
