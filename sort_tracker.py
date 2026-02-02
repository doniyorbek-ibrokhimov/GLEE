"""SORT-style IoU tracker for stabilizing GLEE detections across video frames.

Implements a lightweight tracking-by-detection approach with:
- Kalman filter for bounding box state estimation and smoothing
- Per-class Hungarian matching using IoU cost matrix
- Track lifecycle management (birth, confirmation, death)

Only depends on numpy and scipy (no filterpy).
"""

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# 20 distinct colors for track visualization (BGR order for OpenCV)
TRACK_COLORS: List[Tuple[int, int, int]] = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # blue
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0, 128, 128),    # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),   # brown
    (255, 250, 200),  # beige
    (128, 0, 0),      # maroon
    (170, 255, 195),  # mint
    (128, 128, 0),    # olive
    (255, 215, 180),  # coral
    (0, 0, 128),      # navy
    (128, 128, 128),  # grey
]


def get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Return a deterministic RGB color for a given track ID."""
    return TRACK_COLORS[int(track_id) % len(TRACK_COLORS)]


def iou_batch(bb_test: np.ndarray, bb_truth: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of bounding boxes.

    Args:
        bb_test: (N, 4) array of boxes in xyxy format.
        bb_truth: (M, 4) array of boxes in xyxy format.

    Returns:
        (N, M) IoU matrix.
    """
    if len(bb_test) == 0 or len(bb_truth) == 0:
        return np.empty((len(bb_test), len(bb_truth)))

    bb_test = bb_test[:, np.newaxis, :]   # (N, 1, 4)
    bb_truth = bb_truth[np.newaxis, :, :]  # (1, M, 4)

    xx1 = np.maximum(bb_test[..., 0], bb_truth[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_truth[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_truth[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_truth[..., 3])

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_truth = (bb_truth[..., 2] - bb_truth[..., 0]) * (bb_truth[..., 3] - bb_truth[..., 1])
    union = area_test + area_truth - intersection

    iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def _convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert xyxy bounding box to Kalman measurement [cx, cy, scale, aspect_ratio]."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h  # scale = area
    r = w / max(h, 1e-6)  # aspect ratio
    return np.array([cx, cy, s, r]).reshape((4, 1))


def _convert_x_to_bbox(x: np.ndarray) -> np.ndarray:
    """Convert Kalman state [cx, cy, scale, aspect_ratio, ...] to xyxy bounding box."""
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    s = max(s, 1e-6)
    r = max(r, 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([
        cx - w / 2.0,
        cy - h / 2.0,
        cx + w / 2.0,
        cy + h / 2.0,
    ])


class KalmanBoxTracker:
    """Kalman filter tracker for a single bounding box.

    State vector: [cx, cy, scale, aspect_ratio, dx, dy, ds, 0]
    Measurement:  [cx, cy, scale, aspect_ratio]
    """

    _count: int = 0

    def __init__(self, bbox: np.ndarray, score: float = 0.0, label: int = 0) -> None:
        KalmanBoxTracker._count += 1
        self.id: int = KalmanBoxTracker._count

        self.label: int = label
        self.score: float = score

        # State dimension and measurement dimension
        dim_x = 7
        dim_z = 4

        # State: [cx, cy, s, r, dx, dy, ds]
        self.x = np.zeros((dim_x, 1))
        z = _convert_bbox_to_z(bbox)
        self.x[:dim_z] = z

        # State transition matrix F
        self.F = np.eye(dim_x)
        self.F[0, 4] = 1.0  # cx += dx
        self.F[1, 5] = 1.0  # cy += dy
        self.F[2, 6] = 1.0  # s  += ds

        # Measurement matrix H
        self.H = np.zeros((dim_z, dim_x))
        self.H[:dim_z, :dim_z] = np.eye(dim_z)

        # Measurement noise R
        self.R = np.eye(dim_z)
        self.R[2, 2] *= 10.0
        self.R[3, 3] *= 10.0

        # Covariance P
        self.P = np.eye(dim_x)
        self.P[4, 4] *= 1000.0
        self.P[5, 5] *= 1000.0
        self.P[6, 6] *= 1000.0
        self.P *= 10.0

        # Process noise Q
        self.Q = np.eye(dim_x)
        self.Q[4, 4] *= 0.01
        self.Q[5, 5] *= 0.01
        self.Q[6, 6] *= 0.0001
        self.Q[-1, -1] *= 0  # unused dim

        # Track management
        self.hits: int = 1
        self.hit_streak: int = 1
        self.age: int = 1
        self.time_since_update: int = 0

    def predict(self) -> np.ndarray:
        """Advance state and return predicted xyxy box."""
        # Prevent negative area
        if self.x[2] + self.x[6] <= 0:
            self.x[6] *= 0.0

        # Predict: x = F @ x, P = F @ P @ F^T + Q
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1
        return _convert_x_to_bbox(self.x.flatten())

    def update(self, bbox: np.ndarray, score: float) -> None:
        """Update state with observed measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.score = score

        z = _convert_bbox_to_z(bbox)

        # Kalman gain: K = P @ H^T @ (H @ P @ H^T + R)^-1
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update: x = x + K @ (z - H @ x)
        y = z - self.H @ self.x
        self.x = self.x + K @ y

        # Update covariance: P = (I - K @ H) @ P
        I_KH = np.eye(self.P.shape[0]) - K @ self.H
        self.P = I_KH @ self.P

    def get_state(self) -> np.ndarray:
        """Return current xyxy bounding box estimate."""
        return _convert_x_to_bbox(self.x.flatten())


class SortTracker:
    """SORT-style multi-object tracker with per-class matching.

    Args:
        max_age: Maximum frames a track survives without updates before deletion.
        min_hits: Minimum hit streak for a track to be considered confirmed.
        iou_threshold: Minimum IoU for a detection-track match to be valid.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count: int = 0

    def update(
        self,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Run one tracking step.

        Args:
            boxes_xyxy: (N, 4) detection boxes in xyxy format.
            scores: (N,) confidence scores.
            labels: (N,) integer class labels.

        Returns:
            (M, 7) array where each row is [x1, y1, x2, y2, track_id, score, label].
            Only confirmed tracks (or young tracks in early frames) are returned.
        """
        self.frame_count += 1

        # Predict new locations for all existing trackers
        predicted_boxes = np.zeros((len(self.trackers), 4))
        to_delete: List[int] = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            predicted_boxes[i] = pos
            if np.any(np.isnan(pos)):
                to_delete.append(i)

        for i in reversed(to_delete):
            self.trackers.pop(i)
            predicted_boxes = np.delete(predicted_boxes, i, axis=0)

        # Per-class matching
        matched, unmatched_dets, unmatched_trks = self._match_per_class(
            boxes_xyxy, scores, labels, predicted_boxes,
        )

        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(boxes_xyxy[det_idx], float(scores[det_idx]))

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(
                boxes_xyxy[det_idx],
                score=float(scores[det_idx]),
                label=int(labels[det_idx]),
            )
            self.trackers.append(trk)

        # Build output and prune dead tracks
        results: List[np.ndarray] = []
        active_trackers: List[KalmanBoxTracker] = []
        for trk in self.trackers:
            if trk.time_since_update > self.max_age:
                continue
            active_trackers.append(trk)

            # Return confirmed tracks or young tracks (early frame fallback)
            if trk.hit_streak >= self.min_hits or trk.age <= self.min_hits:
                if trk.time_since_update == 0:  # only return tracks updated this frame
                    state = trk.get_state()
                    results.append(np.array([
                        state[0], state[1], state[2], state[3],
                        float(trk.id),
                        trk.score,
                        float(trk.label),
                    ]))

        self.trackers = active_trackers

        if len(results) == 0:
            return np.empty((0, 7))
        return np.stack(results)

    def _match_per_class(
        self,
        det_boxes: np.ndarray,
        det_scores: np.ndarray,
        det_labels: np.ndarray,
        trk_boxes: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to trackers per class using Hungarian algorithm."""
        if len(det_boxes) == 0 and len(trk_boxes) == 0:
            return [], [], []
        if len(det_boxes) == 0:
            return [], [], list(range(len(self.trackers)))
        if len(trk_boxes) == 0:
            return [], list(range(len(det_boxes))), []

        # Get unique classes from both detections and trackers
        det_classes = set(det_labels.astype(int))
        trk_classes = set(trk.label for trk in self.trackers)
        all_classes = det_classes | trk_classes

        all_matched: List[Tuple[int, int]] = []
        matched_det_set: set = set()
        matched_trk_set: set = set()

        for cls in all_classes:
            # Indices of detections and trackers for this class
            d_idxs = [i for i in range(len(det_labels)) if int(det_labels[i]) == cls]
            t_idxs = [i for i in range(len(self.trackers)) if self.trackers[i].label == cls]

            if len(d_idxs) == 0 or len(t_idxs) == 0:
                continue

            cls_det_boxes = det_boxes[d_idxs]
            cls_trk_boxes = trk_boxes[t_idxs]

            iou_matrix = iou_batch(cls_det_boxes, cls_trk_boxes)
            # Cost = 1 - IoU (Hungarian minimizes cost)
            cost_matrix = 1.0 - iou_matrix

            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                for row, col in zip(row_indices, col_indices):
                    if iou_matrix[row, col] >= self.iou_threshold:
                        det_global = d_idxs[row]
                        trk_global = t_idxs[col]
                        all_matched.append((det_global, trk_global))
                        matched_det_set.add(det_global)
                        matched_trk_set.add(trk_global)

        unmatched_dets = [i for i in range(len(det_boxes)) if i not in matched_det_set]
        unmatched_trks = [i for i in range(len(self.trackers)) if i not in matched_trk_set]

        return all_matched, unmatched_dets, unmatched_trks
