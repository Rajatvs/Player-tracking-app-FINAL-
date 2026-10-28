"""
Movement heatmap generation from tracked player positions.
"""
from typing import List, Dict

import cv2
import numpy as np


class HeatmapGenerator:
    """Accumulates player foot-positions over time into a density heatmap."""

    def __init__(self, frame_width: int, frame_height: int, decay: float = 0.0, point_radius: int = 15):
        """
        Args:
            frame_width / frame_height: dimensions of the source video.
            decay: if > 0, fraction of accumulated heat removed each frame,
                producing a "recent activity" heatmap instead of an
                all-time-cumulative one. 0 = never decays.
            point_radius: radius (px) of the "footprint" added per detection.
        """
        self.width = frame_width
        self.height = frame_height
        self.decay = decay
        self.point_radius = point_radius
        self.accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    def update(self, tracks: List[Dict]) -> None:
        """Add the current frame's player positions to the accumulator."""
        if self.decay:
            self.accumulator *= (1.0 - self.decay)

        for t in tracks:
            left, top, right, bottom = t["bbox"]
            cx = int((left + right) / 2)
            cy = int(bottom)  # feet position, more meaningful than box center
            cx = min(max(cx, 0), self.width - 1)
            cy = min(max(cy, 0), self.height - 1)
            cv2.circle(self.accumulator, (cx, cy), self.point_radius, 1.0, thickness=-1)

    def _normalized_blurred(self) -> np.ndarray:
        max_val = self.accumulator.max()
        if max_val > 0:
            norm = (self.accumulator / max_val * 255).astype(np.uint8)
        else:
            norm = self.accumulator.astype(np.uint8)
        return cv2.GaussianBlur(norm, (0, 0), sigmaX=15, sigmaY=15)

    def as_color_overlay(self, base_frame: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        """Return base_frame blended with a JET-colormapped heatmap."""
        blurred = self._normalized_blurred()
        colored = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
        return cv2.addWeighted(base_frame, 1 - alpha, colored, alpha, 0)

    def as_standalone_image(self) -> np.ndarray:
        """Return just the colormapped heatmap (e.g. for saving to disk)."""
        blurred = self._normalized_blurred()
        return cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

    def reset(self) -> None:
        self.accumulator[:] = 0
