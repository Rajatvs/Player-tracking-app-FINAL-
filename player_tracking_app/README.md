# 🏟️ AI Player Tracking & Analysis

Real-time player detection and tracking in sports video, built with
**YOLOv8**, **OpenCV**, **DeepSORT**, and a **Streamlit** web app for
interactive analysis and movement heatmaps.

## What it does

1. **Detection** — YOLOv8 (Ultralytics) detects people in each video frame.
2. **Tracking** — DeepSORT assigns a persistent ID to each detected player
   and follows them across frames, even through brief occlusions.
3. **Heatmap** — every player's foot position is accumulated into a density
   map, rendered as a live and final movement heatmap.
4. **Web app** — a Streamlit UI lets you upload a video (or use a webcam),
   tune detection confidence and model size, watch live annotated video +
   heatmap side by side, and download the results.

## Project structure

```
player_tracking_app/
├── app.py                 # Streamlit web app (main entry point)
├── run_cli.py              # Headless CLI runner (no browser needed)
├── src/
│   ├── detector.py          # YOLOv8 wrapper -> person detections
│   ├── tracker.py            # DeepSORT wrapper -> persistent player IDs
│   ├── heatmap.py            # Heatmap accumulation + rendering
│   └── video_processor.py    # Wires detector + tracker + heatmap together
├── outputs/                 # Processed videos & heatmaps land here
├── sample_videos/           # Put your own test clips here (see below)
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> The first time you run the app, Ultralytics will automatically download
> the chosen YOLOv8 weights (e.g. `yolov8n.pt`, a few MB). This requires an
> internet connection on the machine you run it on.

## Running the web app

```bash
streamlit run app.py
```

Then in the browser tab that opens:

1. Pick a video source — upload a clip, or select **Webcam** to try it live
   on yourself (great for a quick demo without needing sports footage on
   hand — YOLOv8 detects any person, not just athletes).
2. Choose a model size (`yolov8n` is fastest, `yolov8m` is most accurate).
3. Click **Start / Process** and watch:
   - the annotated video with bounding boxes + player IDs,
   - the live movement heatmap,
   - live metrics (active players, unique players tracked, processing FPS).
4. When it finishes, download the annotated video and the final heatmap
   image.

## Running from the command line (no UI)

```bash
python run_cli.py --input path/to/match.mp4 --model yolov8n.pt --conf 0.35
# or, using your webcam:
python run_cli.py --input 0
```

Outputs are written to `outputs/processed_output.mp4` and
`outputs/final_heatmap.png`.

## Getting test footage

If you don't have sports footage handy, two easy options:

- Use **Webcam mode** in the app — walk around in front of your camera to
  simulate player movement and generate a live heatmap.
- Download a short, free sports clip from a stock site (e.g. Pexels,
  Pixabay) and drop it in `sample_videos/`, then upload it in the app.

## How it works (architecture)

```
Video frame ─▶ YOLOv8 (PlayerDetector) ─▶ person bounding boxes
                                              │
                                              ▼
                              DeepSORT (PlayerTracker) ─▶ persistent player IDs
                                              │
                          ┌───────────────────┼───────────────────┐
                          ▼                                       ▼
                 Draw boxes + ID labels                 Update heatmap accumulator
                          │                                       │
                          ▼                                       ▼
                  Annotated video frame                 Heatmap overlay / image
                          │                                       │
                          └───────────────┬───────────────────────┘
                                          ▼
                                  Streamlit UI (app.py)
```

- **`PlayerDetector`** (`src/detector.py`) runs YOLOv8 restricted to the
  COCO `person` class and returns detections in the `([l, t, w, h], conf,
  class_name)` format DeepSORT expects.
- **`PlayerTracker`** (`src/tracker.py`) wraps `deep-sort-realtime`, which
  combines motion (Kalman filter) and appearance (embedding) cues to keep
  IDs stable across frames.
- **`HeatmapGenerator`** (`src/heatmap.py`) accumulates each player's foot
  position (bottom-center of their bounding box) into a 2D density map,
  blurs it, and applies a JET colormap for visualization.
- **`VideoProcessor`** (`src/video_processor.py`) ties the three together
  per-frame and is shared by both the Streamlit app and the CLI, so the
  processing logic only lives in one place.

## Notes & possible extensions

- Currently tracks all detected people generically as "players" — for a
  specific sport you could add team classification (e.g. by jersey color
  clustering) to color-code by team.
- Speed/distance-covered estimates could be added by converting pixel
  displacement to real-world units via a homography/calibration step.
- For longer videos, consider running on a machine with a CUDA GPU
  (`device="cuda"` in `PlayerDetector`) for real-time performance with
  larger models.
