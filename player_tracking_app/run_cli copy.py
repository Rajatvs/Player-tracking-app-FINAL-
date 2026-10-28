"""
Headless CLI runner - useful for quickly testing the pipeline from a
terminal (no Streamlit UI), e.g. during a live demo or for batch processing.

Usage:
    python run_cli.py --input path/to/video.mp4 --model yolov8n.pt --conf 0.35
    python run_cli.py --input 0                     # use webcam
"""
import argparse
import os

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.video_processor import process_video_file


def parse_args():
    parser = argparse.ArgumentParser(description="AI Player Tracking & Analysis - CLI")
    parser.add_argument("--input", required=True, help="Path to a video file, or 0 for webcam")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 weights, e.g. yolov8n.pt / yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--output-dir", default="outputs", help="Where to write results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    input_source = int(args.input) if args.input.isdigit() else args.input
    output_video = os.path.join(args.output_dir, "processed_output.mp4")
    output_heatmap = os.path.join(args.output_dir, "final_heatmap.png")

    detector = PlayerDetector(model_path=args.model, conf_threshold=args.conf)
    tracker = PlayerTracker()

    def progress(idx, total, stats):
        if total:
            print(f"\rFrame {idx}/{total} | active={stats['active_players']} "
                  f"unique={stats['total_unique_players']} fps={stats['fps']}", end="")
        else:
            print(f"\rFrame {idx} | active={stats['active_players']} "
                  f"unique={stats['total_unique_players']} fps={stats['fps']}", end="")

    final_stats = process_video_file(
        input_path=input_source,
        output_path=output_video,
        heatmap_output_path=output_heatmap,
        detector=detector,
        tracker=tracker,
        progress_callback=progress,
    )

    print()
    print("Done.")
    print(f"Annotated video saved to: {output_video}")
    print(f"Heatmap saved to: {output_heatmap}")
    print(f"Final stats: {final_stats}")


if __name__ == "__main__":
    main()
