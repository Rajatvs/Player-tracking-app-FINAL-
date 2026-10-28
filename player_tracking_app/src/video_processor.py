"""
Orchestrates the detector -> tracker -> heatmap pipeline for a single frame,
and exposes a helper to process a whole video file to disk (used by the CLI
and, optionally, by the Streamlit app for full-video export).
"""
import time
from typing import Dict, Tuple

import cv2
import numpy as np

from .heatmap import HeatmapGenerator


def _color_for_id(track_id) -> Tuple[int, int, int]:
    """Deterministic BGR color per track id, so a player keeps the same
    color across frames."""
    rng = np.random.default_rng(abs(hash(str(track_id))) % (2**32))
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


class VideoProcessor:
    """Runs detection + tracking + heatmap accumulation for a video stream."""

    def __init__(self, detector, tracker, frame_width: int, frame_height: int, heatmap_decay: float = 0.0):
        self.detector = detector
        self.tracker = tracker
        self.heatmap = HeatmapGenerator(frame_width, frame_height, decay=heatmap_decay)
        self.unique_ids = set()
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray, heatmap_alpha: float = 0.55):
        """
        Process a single BGR frame.

        Args:
            frame: BGR frame to process.
            heatmap_alpha: opacity of the heatmap overlay (0-1).

        Returns:
            annotated: frame with bounding boxes + player IDs drawn
            heatmap_overlay: frame blended with the current heatmap
            stats: dict with active_players, total_unique_players, fps
        """
        start = time.time()
        self.frame_count += 1

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections, frame)

        annotated = frame.copy()
        for t in tracks:
            self.unique_ids.add(t["id"])
            left, top, right, bottom = t["bbox"]
            color = _color_for_id(t["id"])
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

            label = f"Player {t['id']}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (left, max(top - text_h - 8, 0)), (left + text_w + 6, top), color, -1)
            cv2.putText(
                annotated, label, (left + 3, max(top - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

        self.heatmap.update(tracks)
        heatmap_overlay = self.heatmap.as_color_overlay(frame, alpha=heatmap_alpha)

        elapsed = max(time.time() - start, 1e-6)
        stats = {
            "frame": self.frame_count,
            "active_players": len(tracks),
            "total_unique_players": len(self.unique_ids),
            "fps": round(1.0 / elapsed, 1),
        }
        return annotated, heatmap_overlay, stats


def process_video_file(
    input_path: str,
    output_path: str,
    heatmap_output_path: str,
    detector,
    tracker,
    progress_callback=None,
) -> Dict:
    """
    Process an entire video file end-to-end and write:
      - output_path: annotated video (bounding boxes + IDs) as .mp4
      - heatmap_output_path: final cumulative heatmap as a .png

    progress_callback(frame_index, total_frames, stats) is called after each
    frame if provided (used by the Streamlit app to update a progress bar).

    Returns final stats dict.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processor = VideoProcessor(detector, tracker, width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stats = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        annotated, _, stats = processor.process_frame(frame)
        writer.write(annotated)
        if progress_callback:
            progress_callback(idx, total_frames, stats)

    cap.release()
    writer.release()

    cv2.imwrite(heatmap_output_path, processor.heatmap.as_standalone_image())
    return stats
