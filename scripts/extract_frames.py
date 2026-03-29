"""
Extract frames from match videos for YOLO training dataset annotation.

Usage:
    python scripts/extract_frames.py [options] <video_file_or_directory>

Options:
    --interval N        Extract every N-th frame (default: 30)
    --output DIR        Output directory for extracted frames (default: dataset/images)
    --max-frames N      Max frames to extract per video (default: 200)
    --start-sec N       Start extracting from N seconds into the video (default: 0)
    --end-sec N         Stop extracting at N seconds into the video (default: end)
    --blur-threshold F  Skip frames with blur score below this (default: 50.0)

The extracted frames are saved as JPEG images ready for annotation in Roboflow
or other annotation tools. A metadata CSV is also written alongside the images
for traceability.

Recommended workflow:
    1. Run this script to extract diverse frames from your match videos.
    2. Upload the resulting images to Roboflow (https://roboflow.com).
    3. Annotate bounding boxes around each FUEL ball (class: "fuel").
    4. Export the dataset in "YOLOv8" format.
    5. Place the exported dataset in the `dataset/` directory.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np


def compute_blur_score(frame):
    """Compute Laplacian variance as a measure of image sharpness.
    Higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(
    video_path,
    output_dir,
    interval=30,
    max_frames=200,
    start_sec=0,
    end_sec=None,
    blur_threshold=50.0,
):
    """Extract frames from a single video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec is not None else total_frames

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    # Sanitize for filenames
    video_basename = video_basename.replace(" ", "_").replace("/", "_")

    print(f"  Video: {video_path}")
    print(f"    FPS: {fps:.1f} | Duration: {duration:.1f}s | Total frames: {total_frames}")
    print(f"    Extracting every {interval}-th frame from frame {start_frame} to {end_frame}")
    print(f"    Blur threshold: {blur_threshold} | Max frames: {max_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted = []
    frame_idx = start_frame
    skipped_blur = 0
    saved_count = 0

    while frame_idx < end_frame and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % interval == 0:
            blur = compute_blur_score(frame)
            if blur < blur_threshold:
                skipped_blur += 1
            else:
                filename = f"{video_basename}_f{frame_idx:06d}.jpg"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted.append({
                    "filename": filename,
                    "video": os.path.basename(video_path),
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "blur_score": round(blur, 2),
                })
                saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"    Extracted: {saved_count} frames | Skipped (blur): {skipped_blur}")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from match videos for YOLO annotation."
    )
    parser.add_argument(
        "source",
        help="Path to a video file or a directory containing video files.",
    )
    parser.add_argument("--interval", type=int, default=30, help="Extract every N-th frame")
    parser.add_argument("--output", default="dataset/images", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=200, help="Max frames per video")
    parser.add_argument("--start-sec", type=float, default=0, help="Start time (seconds)")
    parser.add_argument("--end-sec", type=float, default=None, help="End time (seconds)")
    parser.add_argument("--blur-threshold", type=float, default=50.0, help="Min blur score")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect video files
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    if os.path.isfile(args.source):
        video_files = [args.source]
    elif os.path.isdir(args.source):
        video_files = sorted([
            os.path.join(args.source, f)
            for f in os.listdir(args.source)
            if os.path.splitext(f)[1].lower() in video_exts
        ])
    else:
        print(f"Error: {args.source} is not a valid file or directory.")
        sys.exit(1)

    if not video_files:
        print(f"No video files found in {args.source}")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s) to process.\n")

    all_extracted = []
    for vf in video_files:
        frames = extract_frames(
            vf,
            args.output,
            interval=args.interval,
            max_frames=args.max_frames,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            blur_threshold=args.blur_threshold,
        )
        all_extracted.extend(frames)
        print()

    # Write metadata CSV
    csv_path = os.path.join(args.output, "_metadata.csv")
    if all_extracted:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_extracted[0].keys())
            writer.writeheader()
            writer.writerows(all_extracted)

    print(f"=== DONE ===")
    print(f"Total frames extracted: {len(all_extracted)}")
    print(f"Output directory: {os.path.abspath(args.output)}")
    print(f"Metadata CSV: {os.path.abspath(csv_path)}")
    print()
    print("Next steps:")
    print("  1. Upload images to Roboflow (https://app.roboflow.com)")
    print("  2. Create a project with class 'fuel' (bounding box)")
    print("  3. Annotate all FUEL balls in each frame")
    print("  4. Export in 'YOLOv8' format")
    print("  5. Place exported dataset in the 'dataset/' directory")


if __name__ == "__main__":
    main()
