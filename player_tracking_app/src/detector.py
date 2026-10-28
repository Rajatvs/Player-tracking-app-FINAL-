"""
YOLOv8-based player (person) detector.

Wraps an Ultralytics YOLOv8 model and returns detections in the format
expected by deep-sort-realtime: ([left, top, w, h], confidence, class_name)
"""
from typing import List, Optional, Tuple

from ultralytics import YOLO


class PlayerDetector:
    """Detects people ("players") in a video frame using YOLOv8."""

    # COCO class id for 'person'
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: path/name of the YOLOv8 weights. Ultralytics will
                auto-download standard weights (yolov8n.pt, yolov8s.pt, ...)
                the first time they are used.
            conf_threshold: minimum confidence to keep a detection.
            device: 'cpu', 'cuda', or None to let Ultralytics decide.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        if device:
            self.model.to(device)

    def detect(self, frame) -> List[Tuple[list, float, str]]:
        """
        Run person detection on a single BGR frame (as returned by cv2.VideoCapture).

        Returns:
            List of (bbox, confidence, class_name) tuples where bbox is
            [left, top, width, height] in pixel coordinates.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=[self.PERSON_CLASS_ID],
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, "player"))
        return detections
