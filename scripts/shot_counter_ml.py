"""
ML-based shot counter for FRC REBUILT match videos.

Uses YOLOv11 for FUEL ball detection and ByteTrack for multi-object tracking.
Counts successful shots when tracked balls enter a user-defined basket ROI polygon.

Usage:
    python scripts/shot_counter_ml.py [options] <video_file>

Options:
    --model PATH        Path to trained YOLO model weights (default: models/fuel_detector/weights/best.pt)
    --conf FLOAT        Detection confidence threshold (default: 0.25)
    --iou FLOAT         NMS IoU threshold (default: 0.5)
    --imgsz N           Inference image size (default: 640)
    --no-display        Run without GUI display (headless)
    --output PATH       Save annotated output video to file
    --reset-roi         Force re-draw of ROI polygons (ignore cache)

Controls (during playback):
    p           Pause / resume
    d           Toggle debug overlays
    ESC         Quit
    +/-         Adjust confidence threshold
    s           Skip forward 5 seconds
"""

import json
import os
import sys
import time
from collections import defaultdict
from enum import Enum, auto

# Fix for macOS OpenMP library conflicts and GUI backend crashes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# CRITICAL: Import ultralytics/torch BEFORE cv2 on macOS to avoid NSException/Segfaults
from ultralytics import YOLO
import torch

import cv2
import numpy as np

from utils import get_screen_size


# ============================
# Ball State Machine
# ============================
class BallState(Enum):
    """State of a tracked ball relative to the basket ROI."""
    OUTSIDE = auto()     # Ball is outside the basket ROI
    INSIDE = auto()      # Ball is currently inside the basket ROI
    SCORED = auto()       # Ball has been counted as a score


class TrackedBall:
    """Maintains scoring state for a single tracked ball."""

    def __init__(self, track_id):
        self.track_id = track_id
        self.state = BallState.OUTSIDE
        self.positions = []  # History of (x, y) centroids
        self.entry_frame = None
        self.score_frame = None
        self.frames_inside = 0
        self.last_seen_frame = 0
        self.confidence_history = []

    def update_position(self, cx, cy, frame_idx, confidence):
        """Record a new position for this ball."""
        self.positions.append((cx, cy))
        self.last_seen_frame = frame_idx
        self.confidence_history.append(confidence)
        # Keep only last 60 positions for trail drawing
        if len(self.positions) > 60:
            self.positions.pop(0)
        if len(self.confidence_history) > 30:
            self.confidence_history.pop(0)

    @property
    def avg_confidence(self):
        if not self.confidence_history:
            return 0.0
        return sum(self.confidence_history) / len(self.confidence_history)


# ============================
# ROI Utilities (reused from shot_counter.py)
# ============================
def point_in_poly(pt, poly):
    """Check if point is inside polygon."""
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def get_roi_cache_path():
    return os.path.join(os.path.dirname(__file__), "roi_cache.json")


def load_roi_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_roi_cache(path, cache):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to save ROI cache: {e}")


def poly_from_cache(raw):
    if not isinstance(raw, list) or len(raw) < 3:
        return np.array([], dtype=np.int32)
    try:
        poly = np.array(raw, dtype=np.int32)
    except Exception:
        return np.array([], dtype=np.int32)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        return np.array([], dtype=np.int32)
    return poly


def center_window(win_name, w, h, screen_w, screen_h):
    x = max(0, (screen_w - w) // 2)
    y = max(0, (screen_h - h) // 2)
    cv2.moveWindow(win_name, x, y)


class PolyDrawer:
    """Interactive polygon ROI drawer."""

    def __init__(self, win_name, help_text, display_size=None):
        self.win = win_name
        self.help_text = help_text
        self.pts = []
        self.done = False
        self.img = None
        self.display_size = display_size
        self.scale = 1.0
        self.pad_x = 0
        self.pad_y = 0
        self.draw_w = 0
        self.draw_h = 0

    def _mouse(self, event, x, y, flags, param):
        if self.done:
            return
        in_bounds = (
            self.pad_x <= x < self.pad_x + self.draw_w
            and self.pad_y <= y < self.pad_y + self.draw_h
        )
        if not in_bounds:
            return
        x_img = int(np.clip(round((x - self.pad_x) / self.scale), 0, self.img.shape[1] - 1))
        y_img = int(np.clip(round((y - self.pad_y) / self.scale), 0, self.img.shape[0] - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x_img, y_img))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pts:
                self.pts.pop()

    def draw(self, frame):
        self.img = frame.copy()
        img_h, img_w = self.img.shape[:2]
        if self.display_size is None:
            target_w, target_h = img_w, img_h
        else:
            target_w, target_h = self.display_size
        self.scale = min(target_w / img_w, target_h / img_h)
        self.draw_w = max(1, int(round(img_w * self.scale)))
        self.draw_h = max(1, int(round(img_h * self.scale)))
        self.pad_x = max(0, (target_w - self.draw_w) // 2)
        self.pad_y = max(0, (target_h - self.draw_h) // 2)

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, target_w, target_h)
        screen_w, screen_h = get_screen_size()
        center_window(self.win, target_w, target_h, screen_w, screen_h)
        cv2.setMouseCallback(self.win, self._mouse)

        while True:
            vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            resized = cv2.resize(self.img, (self.draw_w, self.draw_h), interpolation=cv2.INTER_LINEAR)
            vis[self.pad_y:self.pad_y + self.draw_h, self.pad_x:self.pad_x + self.draw_w] = resized
            for p in self.pts:
                sp = (int(round(p[0] * self.scale + self.pad_x)),
                      int(round(p[1] * self.scale + self.pad_y)))
                cv2.circle(vis, sp, 4, (0, 255, 0), -1)
            if len(self.pts) >= 2:
                disp_pts = np.array(
                    [[int(round(px * self.scale + self.pad_x)),
                      int(round(py * self.scale + self.pad_y))]
                     for px, py in self.pts],
                    dtype=np.int32,
                )
                cv2.polylines(vis, [disp_pts], False, (0, 255, 0), 2)
            y0 = 20
            for line in self.help_text.split("\n"):
                cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 255), 2, cv2.LINE_AA)
                y0 += 22
            cv2.imshow(self.win, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 13 and len(self.pts) >= 3:
                self.done = True
                break
            elif k == 27:
                self.pts = []
                self.done = True
                break
        cv2.destroyWindow(self.win)
        return np.array(self.pts, dtype=np.int32)


def confirm_cached_poly(frame, poly, win_name, title, prompt, display_size):
    """Show cached ROI and ask user to confirm, redraw, or skip."""
    img_h, img_w = frame.shape[:2]
    target_w, target_h = display_size
    scale = min(target_w / img_w, target_h / img_h)
    draw_w = max(1, int(round(img_w * scale)))
    draw_h = max(1, int(round(img_h * scale)))
    pad_x = max(0, (target_w - draw_w) // 2)
    pad_y = max(0, (target_h - draw_h) // 2)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, target_w, target_h)
    screen_w, screen_h = get_screen_size()
    center_window(win_name, target_w, target_h, screen_w, screen_h)

    disp_pts = np.array(
        [[int(round(px * scale + pad_x)), int(round(py * scale + pad_y))]
         for px, py in poly],
        dtype=np.int32,
    )

    while True:
        vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
        vis[pad_y:pad_y + draw_h, pad_x:pad_x + draw_w] = resized
        cv2.polylines(vis, [disp_pts], True, (0, 255, 255), 3)
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, prompt, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(win_name, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("y"), ord("Y"), 13):
            cv2.destroyWindow(win_name)
            return "use"
        if k in (ord("n"), ord("N")):
            cv2.destroyWindow(win_name)
            return "redraw"
        if k in (27, ord("s"), ord("S")):
            cv2.destroyWindow(win_name)
            return "skip"


# ============================
# ROI Setup
# ============================
def setup_basket_roi(frame, cache, video_key, screen_size, reset_roi=False):
    """Get basket ROI polygon — from cache or by drawing."""
    w, h = frame.shape[1], frame.shape[0]
    cached = cache.get(video_key, {})
    cached_poly = poly_from_cache(cached.get("basket_poly", []))
    cached_size = cached.get("frame_size", [])
    size_match = (
        isinstance(cached_size, list)
        and len(cached_size) == 2
        and int(cached_size[0]) == w
        and int(cached_size[1]) == h
    )

    if not reset_roi and cached_poly.size != 0 and size_match:
        choice = confirm_cached_poly(
            frame, cached_poly,
            "Cached Basket ROI", "Cached Basket ROI Found",
            "Y/Enter: use | N: redraw | ESC: cancel",
            display_size=screen_size,
        )
        if choice == "use":
            return cached_poly
        elif choice == "skip":
            return None

    poly = PolyDrawer(
        "Draw Basket ROI",
        "LEFT click: add point | RIGHT click: undo\n"
        "ENTER: finish (>=3 points) | ESC: cancel\n"
        "Tip: outline the basket opening / net area tightly.",
        display_size=screen_size,
    ).draw(frame)

    return poly if poly.size != 0 else None


# ============================
# Scoring Logic
# ============================
class ScoringEngine:
    """Manages ball tracking state and scoring decisions."""

    # Minimum consecutive frames a ball must be inside ROI to count as a score.
    # This prevents false triggers from detection jitter at the ROI boundary.
    MIN_FRAMES_INSIDE = 2

    # Maximum frames a ball can be unseen before we stop tracking it for scoring.
    MAX_UNSEEN_FRAMES = 15

    def __init__(self, basket_poly, fps):
        self.basket_poly = basket_poly
        self.fps = fps
        self.balls = {}  # track_id -> TrackedBall
        self.score_events = []  # list of (frame_idx, timestamp_sec, track_id)

    def update(self, track_id, cx, cy, frame_idx, confidence):
        """Update a tracked ball's position and check for scoring."""
        if track_id not in self.balls:
            self.balls[track_id] = TrackedBall(track_id)

        ball = self.balls[track_id]
        ball.update_position(cx, cy, frame_idx, confidence)

        # Already scored — don't re-score
        if ball.state == BallState.SCORED:
            return False

        inside = point_in_poly((cx, cy), self.basket_poly)

        if ball.state == BallState.OUTSIDE:
            if inside:
                ball.state = BallState.INSIDE
                ball.entry_frame = frame_idx
                ball.frames_inside = 1
        elif ball.state == BallState.INSIDE:
            if inside:
                ball.frames_inside += 1
                if ball.frames_inside >= self.MIN_FRAMES_INSIDE:
                    # Score!
                    ball.state = BallState.SCORED
                    ball.score_frame = frame_idx
                    t = frame_idx / self.fps
                    self.score_events.append((frame_idx, t, track_id))
                    return True
            else:
                # Exited without enough frames inside — reset
                ball.state = BallState.OUTSIDE
                ball.frames_inside = 0

        return False

    def cleanup_stale(self, current_frame):
        """Remove balls that haven't been seen for a while."""
        stale_ids = [
            tid for tid, ball in self.balls.items()
            if (current_frame - ball.last_seen_frame) > self.MAX_UNSEEN_FRAMES
            and ball.state != BallState.SCORED
        ]
        for tid in stale_ids:
            del self.balls[tid]


# ============================
# Visualization
# ============================
def draw_detections(vis, boxes, track_ids, confidences, scored_this_frame, balls_state):
    """Draw bounding boxes and labels for detected balls."""
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        x1, y1, x2, y2 = map(int, box)
        tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else None

        # Color based on state
        if tid is not None and tid == scored_this_frame:
            color = (0, 255, 0)  # Green flash for score
            thickness = 3
        elif tid is not None and tid in balls_state and balls_state[tid].state == BallState.SCORED:
            color = (0, 200, 0)  # Scored ball (dimmer green)
            thickness = 2
        elif tid is not None and tid in balls_state and balls_state[tid].state == BallState.INSIDE:
            color = (0, 255, 255)  # Yellow — inside ROI
            thickness = 2
        else:
            color = (255, 100, 50)  # Blue — normal tracking
            thickness = 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # Centroid dot
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(vis, (cx, cy), 4, color, -1)

        # Label
        label = f"ID:{tid}" if tid is not None else "?"
        label += f" {conf:.2f}"
        if tid is not None and tid == scored_this_frame:
            label += " SCORED!"
        cv2.putText(vis, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_trails(vis, balls_state, fade=True):
    """Draw position trails for tracked balls."""
    for tid, ball in balls_state.items():
        if len(ball.positions) < 2:
            continue
        pts = ball.positions
        for i in range(1, len(pts)):
            if fade:
                alpha = i / len(pts)
                color = (int(255 * alpha), int(100 * alpha), int(50 * alpha))
            else:
                color = (255, 100, 50)
            cv2.line(vis, (int(pts[i-1][0]), int(pts[i-1][1])),
                     (int(pts[i][0]), int(pts[i][1])), color, 1, cv2.LINE_AA)


def draw_hud(vis, frame_idx, fps, score_count, conf_threshold, debug_on,
             basket_poly, detection_count, track_count):
    """Draw heads-up display with score and status info."""
    h, w = vis.shape[:2]

    # Basket ROI
    cv2.polylines(vis, [basket_poly], True, (0, 255, 255), 2)

    # Score counter (bottom left)
    cv2.putText(vis, f"Score: {score_count}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

    # Frame / time info (top left)
    current_time = frame_idx / fps
    cv2.putText(vis, f"Frame: {frame_idx} | Time: {current_time:.1f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)

    # Detection stats (top left, below frame info)
    if debug_on:
        cv2.putText(vis, f"Detections: {detection_count} | Tracks: {track_count} | Conf: {conf_threshold:.2f}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(vis, "Debug[d]: ON | +/-: conf | s: skip 5s | p: pause | ESC: quit",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (150, 150, 150), 1, cv2.LINE_AA)


# ============================
# Main
# ============================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ML-based shot counter for FRC REBUILT")
    parser.add_argument("video", help="Path to match video file")
    parser.add_argument("--model", default="models/fuel_detector/weights/best.pt",
                        help="Path to trained YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--no-display", action="store_true", help="Run headless")
    parser.add_argument("--output", default=None, help="Save annotated video to file")
    parser.add_argument("--reset-roi", action="store_true", help="Force re-draw ROI")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: Model weights not found: {args.model}")
        print()
        print("To train a model:")
        print("  1. python scripts/extract_frames.py assets/")
        print("  2. Annotate frames in Roboflow, export as YOLOv8")
        print("  3. python scripts/train_yolo.py")
        sys.exit(1)

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(args.model)
    print(f"  Model loaded: {args.model}")
    print(f"  Classes: {model.names}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    print(f"  Video: {args.video}")
    print(f"  Resolution: {w}x{h} | FPS: {fps:.1f} | Duration: {duration:.1f}s")

    # Read first frame for ROI setup
    ret, frame0 = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        sys.exit(1)

    # Setup basket ROI
    if not args.no_display:
        screen_w, screen_h = get_screen_size()
        screen_size = (screen_w, screen_h)
    else:
        screen_size = (1920, 1080)

    cache_path = get_roi_cache_path()
    roi_cache = load_roi_cache(cache_path)
    video_key = os.path.basename(args.video)

    basket_poly = setup_basket_roi(frame0, roi_cache, video_key, screen_size, args.reset_roi)
    if basket_poly is None:
        print("No basket ROI provided. Exiting.")
        sys.exit(0)

    # Save ROI to cache
    roi_cache[video_key] = {
        "frame_size": [w, h],
        "basket_poly": basket_poly.tolist(),
    }
    save_roi_cache(cache_path, roi_cache)

    # Initialize scoring engine
    scorer = ScoringEngine(basket_poly, fps)

    # Setup output video writer
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # Setup display window
    if not args.no_display:
        main_win = "Shot Counter (ML)"
        cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(main_win, min(w, screen_w - 100), min(h, screen_h - 100))
        center_window(main_win, min(w, screen_w - 100), min(h, screen_h - 100), screen_w, screen_h)

    # Rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # State
    conf_threshold = args.conf
    debug_on = True
    frame_idx = 0
    t_start = time.time()

    print(f"\nRunning inference (conf={conf_threshold}, iou={args.iou}, imgsz={args.imgsz})...")
    print("Controls: p=pause, d=debug, +/-=conf, s=skip 5s, ESC=quit\n")

    # Run tracking with Ultralytics built-in tracker
    results = model.track(
        source=args.video,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        conf=conf_threshold,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img

        # Extract detections
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.array([])
        confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])
        track_ids = result.boxes.id.cpu().numpy() if result.boxes is not None and result.boxes.id is not None else None

        detection_count = len(boxes)
        track_count = len(track_ids) if track_ids is not None else 0

        # Update scoring for each tracked ball
        scored_this_frame = None
        if track_ids is not None:
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                tid = int(track_ids[i])
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                is_score = scorer.update(tid, cx, cy, frame_idx, float(conf))
                if is_score:
                    scored_this_frame = tid
                    t = frame_idx / fps
                    print(f"  [SCORE] Frame {frame_idx} ({t:.2f}s) — Track ID {tid}")

        scorer.cleanup_stale(frame_idx)

        # Visualization
        if not args.no_display or video_writer:
            vis = frame.copy()

            if debug_on:
                draw_trails(vis, scorer.balls)

            draw_detections(vis, boxes, track_ids, confidences,
                           scored_this_frame, scorer.balls)

            draw_hud(vis, frame_idx, fps, len(scorer.score_events),
                     conf_threshold, debug_on, basket_poly,
                     detection_count, track_count)

            # Flash green border on score
            if scored_this_frame is not None:
                cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 255, 0), 6)

            if video_writer:
                video_writer.write(vis)

            if not args.no_display:
                cv2.imshow(main_win, vis)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('p'):
                    while True:
                        k2 = cv2.waitKey(0) & 0xFF
                        if k2 == ord('p') or k2 == 27:
                            break
                    if k2 == 27:
                        break
                elif key == ord('d'):
                    debug_on = not debug_on
                elif key == ord('+') or key == ord('='):
                    conf_threshold = min(0.95, conf_threshold + 0.05)
                    print(f"  Confidence threshold: {conf_threshold:.2f}")
                elif key == ord('-'):
                    conf_threshold = max(0.05, conf_threshold - 0.05)
                    print(f"  Confidence threshold: {conf_threshold:.2f}")
                elif key == ord('s'):
                    skip_frames = int(5 * fps)
                    for _ in range(skip_frames):
                        # Advance the result iterator
                        try:
                            next(results)
                            frame_idx += 1
                        except StopIteration:
                            break

        frame_idx += 1

        # Progress every 300 frames
        if frame_idx % 300 == 0:
            elapsed = time.time() - t_start
            processing_fps = frame_idx / elapsed if elapsed > 0 else 0
            progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            print(f"  Progress: {progress:.1f}% ({frame_idx}/{total_frames}) | "
                  f"{processing_fps:.1f} FPS | Scores: {len(scorer.score_events)}")

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start

    # Results
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Video: {args.video}")
    print(f"Model: {args.model}")
    print(f"Processed: {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} FPS)")
    print()

    if scorer.score_events:
        print(f"Scored shots ({len(scorer.score_events)} total):")
        for fi, t, tid in scorer.score_events:
            print(f"  Track {tid:3d}  Frame {fi:6d}  ->  {t:8.3f}s")
    else:
        print("No scored shots detected.")

    print(f"\nTotal score: {len(scorer.score_events)}")

    if args.output:
        print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
