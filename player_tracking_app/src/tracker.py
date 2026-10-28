"""
DeepSORT-based multi-object tracker for assigning persistent IDs to players
across frames.
"""
from typing import Dict, List

from deep_sort_realtime.deepsort_tracker import DeepSort


class PlayerTracker:
    """Thin wrapper around deep-sort-realtime's DeepSort tracker."""

    def __init__(self, max_age: int = 30, n_init: int = 3, max_iou_distance: float = 0.7):
        """
        Args:
            max_age: frames to keep a track alive without a matching detection
                (handles brief occlusions).
            n_init: number of consecutive detections required before a track
                is confirmed (reduces false-positive IDs).
            max_iou_distance: gating threshold for the IOU matching cascade.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=max_iou_distance,
        )

    def update(self, detections: list, frame) -> List[Dict]:
        """
        Update tracks with the latest detections.

        Args:
            detections: output of PlayerDetector.detect() for the current frame.
            frame: the current BGR frame (used internally for the appearance
                embedder).

        Returns:
            List of dicts: {"id": track_id, "bbox": (left, top, right, bottom)}
            for every *confirmed* track.
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        confirmed = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            left, top, right, bottom = t.to_ltrb()
            confirmed.append(
                {
                    "id": t.track_id,
                    "bbox": (int(left), int(top), int(right), int(bottom)),
                }
            )
        return confirmed
