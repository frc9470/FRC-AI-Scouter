"""
Semi-automatic shot counter for a robotics match video.

What it does:
- Lets you draw the basket opening as a polygon ROI (mouse clicks).
- Lets you tune an HSV color range for the ball (trackbars).
- Tracks ball centroids via simple color segmentation + contour filtering.
- Counts a "made" when a ball centroid enters the basket ROI and then disappears
  (or exits downward) within a short time window.

Notes:
- This does NOT automatically identify robot 9470. To filter only 9470's shots,
  you can optionally define an additional "robot shooting zone" ROI near where 9470
  stands and only count makes when a ball originated from that zone. (Included.)
- Works best when balls have a distinct color (often orange) and the basket is stationary.

Dependencies:
  pip install opencv-python numpy
Run:
  python shot_counter.py /path/to/video.mov
"""

import sys
import cv2
import numpy as np

# ----------------------------
# Utility: polygon ROI drawing
# ----------------------------
class PolyDrawer:
    def __init__(self, win_name, help_text):
        self.win = win_name
        self.help_text = help_text
        self.pts = []
        self.done = False
        self.img = None

    def _mouse(self, event, x, y, flags, param):
        if self.done:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # undo last point
            if self.pts:
                self.pts.pop()

    def draw(self, frame):
        self.img = frame.copy()
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win, self._mouse)

        while True:
            vis = self.img.copy()
            # Draw points + edges
            for p in self.pts:
                cv2.circle(vis, p, 4, (0, 255, 0), -1)
            if len(self.pts) >= 2:
                cv2.polylines(vis, [np.array(self.pts, dtype=np.int32)], False, (0, 255, 0), 2)

            # Overlay instructions
            y0 = 20
            for line in self.help_text.split("\n"):
                cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                y0 += 22

            cv2.imshow(self.win, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 13:  # Enter finishes
                if len(self.pts) >= 3:
                    self.done = True
                    break
            elif k == 27:  # Esc cancels
                self.pts = []
                self.done = True
                break

        cv2.destroyWindow(self.win)
        return np.array(self.pts, dtype=np.int32)


# ----------------------------
# HSV trackbar calibration
# ----------------------------
def make_hsv_tuner(win="HSV Tuner"):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def nothing(x):
        pass

    # Reasonable defaults for "orange ball" — tune as needed
    cv2.createTrackbar("H min", win, 5, 179, nothing)
    cv2.createTrackbar("H max", win, 25, 179, nothing)
    cv2.createTrackbar("S min", win, 120, 255, nothing)
    cv2.createTrackbar("S max", win, 255, 255, nothing)
    cv2.createTrackbar("V min", win, 120, 255, nothing)
    cv2.createTrackbar("V max", win, 255, 255, nothing)

    return win


def get_hsv_bounds(win):
    hmin = cv2.getTrackbarPos("H min", win)
    hmax = cv2.getTrackbarPos("H max", win)
    smin = cv2.getTrackbarPos("S min", win)
    smax = cv2.getTrackbarPos("S max", win)
    vmin = cv2.getTrackbarPos("V min", win)
    vmax = cv2.getTrackbarPos("V max", win)
    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)
    return lower, upper


# ----------------------------
# Geometry helper
# ----------------------------
def point_in_poly(pt, poly):
    # poly: Nx2 int32
    return cv2.pointPolygonTest(poly, pt, False) >= 0


# ----------------------------
# Main analysis
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python shot_counter.py /path/to/video")
        sys.exit(1)

    video_path = sys.argv[1]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Grab a reference frame for ROIs
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")
    h, w = frame0.shape[:2]

    # Ask user to define basket opening ROI (polygon)
    basket_poly = PolyDrawer(
        "Draw Basket ROI",
        "LEFT click: add point | RIGHT click: undo\n"
        "ENTER: finish (>=3 points) | ESC: cancel (empty)\n"
        "Tip: outline the basket opening / net area tightly."
    ).draw(frame0)

    if basket_poly.size == 0:
        print("No basket ROI provided. Exiting.")
        sys.exit(0)

    # Optional: define a "9470 shooting zone" ROI to approximate "shots by 9470"
    # (Define where 9470 typically shoots from; count makes only if the ball was last seen in this zone.)
    zone_poly = PolyDrawer(
        "Optional: Draw 9470 Shooting Zone ROI",
        "OPTIONAL\n"
        "LEFT click: add point | RIGHT click: undo\n"
        "ENTER: finish (>=3 points) | ESC: skip\n"
        "Tip: roughly outline the floor area near robot 9470's shooter position."
    ).draw(frame0)

    use_zone = zone_poly.size != 0

    # HSV tuner
    tuner_win = make_hsv_tuner()

    # Tracking state
    # We'll keep a simple nearest-centroid tracker for a single most-likely ball.
    last_centroid = None
    last_seen_frame = -10_000
    last_seen_in_zone_frame = -10_000

    # "Made" event logic
    in_basket = False
    entered_basket_frame = None
    prev_inside_basket = False
    last_inside_centroid = None

    # Tunables (adjust as needed)
    MAX_MISSED_FRAMES_AFTER_ENTRY = int(0.30 * fps)  # if ball disappears soon after entry, count it
    MIN_CONTOUR_AREA = 80    # reject noise (increase if too many false detections)
    MAX_CONTOUR_AREA = 20_000  # reject giant blobs
    MAX_TRACK_DIST = 80      # px; tracker association radius
    MIN_DOWNWARD_EXIT_PX = 4  # minimum downward motion between inside and outside centroid

    basket_y_mid = (int(np.min(basket_poly[:, 1])) + int(np.max(basket_poly[:, 1]))) / 2.0

    made_events = []  # list of (frame_idx, seconds)
    frame_idx = 0
    debug_overlays = True

    # Rewind to beginning (we consumed 1 frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = get_hsv_bounds(tuner_win)

        mask = cv2.inRange(hsv, lower, upper)

        # Clean mask
        mask = cv2.medianBlur(mask, 5)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

        # Find candidate ball contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
                continue
            (x, y, ww, hh) = cv2.boundingRect(c)
            cx = x + ww / 2.0
            cy = y + hh / 2.0
            candidates.append((area, (cx, cy), (x, y, ww, hh)))

        # Pick best candidate to track:
        # - if we already have a centroid, choose nearest
        # - else choose largest area
        centroid = None
        bbox = None
        if candidates:
            if last_centroid is not None:
                best = None
                best_d = 1e9
                for area, cxy, bb in candidates:
                    d = np.hypot(cxy[0] - last_centroid[0], cxy[1] - last_centroid[1])
                    if d < best_d:
                        best_d = d
                        best = (area, cxy, bb)
                if best is not None and best_d <= MAX_TRACK_DIST:
                    centroid = best[1]
                    bbox = best[2]
                else:
                    # fall back to largest
                    best = max(candidates, key=lambda t: t[0])
                    centroid = best[1]
                    bbox = best[2]
            else:
                best = max(candidates, key=lambda t: t[0])
                centroid = best[1]
                bbox = best[2]

        # Update tracker state
        if centroid is not None:
            last_centroid = centroid
            last_seen_frame = frame_idx

            if use_zone and point_in_poly((centroid[0], centroid[1]), zone_poly):
                last_seen_in_zone_frame = frame_idx

        # Determine if we should count based on zone constraint
        zone_ok = True
        if use_zone:
            # Require that ball was seen in 9470 zone within last 1.0s
            zone_ok = (frame_idx - last_seen_in_zone_frame) <= int(1.0 * fps)

        # Basket entry / make logic
        downward_exit = False
        ball_missing = False
        timed_out = False
        tracked_centroid = None
        if last_centroid is not None and (frame_idx - last_seen_frame) <= 1:
            tracked_centroid = (last_centroid[0], last_centroid[1])
            inside_basket = point_in_poly(tracked_centroid, basket_poly)
        else:
            inside_basket = False

        if inside_basket and tracked_centroid is not None:
            last_inside_centroid = tracked_centroid

        if not in_basket:
            if inside_basket and zone_ok:
                in_basket = True
                entered_basket_frame = frame_idx
        else:
            # We were in basket; count a make when the tracked centroid exits the ROI downward.
            if (
                prev_inside_basket
                and not inside_basket
                and tracked_centroid is not None
                and last_inside_centroid is not None
            ):
                dy = tracked_centroid[1] - last_inside_centroid[1]
                downward_exit = (
                    dy >= MIN_DOWNWARD_EXIT_PX
                    and tracked_centroid[1] >= basket_y_mid
                    and last_inside_centroid[1] >= basket_y_mid
                )

            if downward_exit:
                t = frame_idx / fps
                made_events.append((entered_basket_frame, t))
                in_basket = False
                entered_basket_frame = None
                last_inside_centroid = None
            # Fallback: if the ball disappears soon after entry, still count as make.
            ball_missing = (frame_idx - last_seen_frame) > 2
            timed_out = (frame_idx - entered_basket_frame) > MAX_MISSED_FRAMES_AFTER_ENTRY

            if in_basket and ball_missing and not timed_out:
                t = frame_idx / fps
                made_events.append((entered_basket_frame, t))
                in_basket = False
                entered_basket_frame = None
                last_inside_centroid = None
            elif timed_out:
                # Give up (likely bounce/false positive)
                in_basket = False
                entered_basket_frame = None
                last_inside_centroid = None

        # ----------------------------
        # Debug visualization
        # ----------------------------
        vis = frame.copy()

        # Draw basket poly
        cv2.polylines(vis, [basket_poly], True, (0, 255, 255), 2)
        cv2.putText(vis, "Basket ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw optional zone
        if use_zone:
            cv2.polylines(vis, [zone_poly], True, (255, 255, 0), 2)
            cv2.putText(vis, "9470 Zone ROI", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # Draw all current contour candidates for debugging.
        if debug_overlays:
            for idx, (area, cxy, bb) in enumerate(candidates):
                x, y, ww, hh = bb
                cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 255, 0), 1)
                cv2.circle(vis, (int(cxy[0]), int(cxy[1])), 3, (0, 255, 0), -1)
                cv2.putText(
                    vis,
                    f"c{idx} a={int(area)}",
                    (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Draw tracked ball
        if last_centroid is not None and (frame_idx - last_seen_frame) <= 2:
            cv2.circle(vis, (int(last_centroid[0]), int(last_centroid[1])), 8, (0, 0, 255), -1)
            if bbox is not None:
                x, y, ww, hh = bbox
                cv2.rectangle(vis, (x, y), (x + ww, y + hh), (0, 0, 255), 2)

        if debug_overlays:
            frames_since_seen = frame_idx - last_seen_frame
            frames_since_zone = frame_idx - last_seen_in_zone_frame if use_zone else 0
            debug_lines = [
                f"Debug[d]: ON  candidates={len(candidates)}",
                f"inside_basket={inside_basket} prev_inside={prev_inside_basket} in_basket={in_basket}",
                f"zone_ok={zone_ok} use_zone={use_zone} frames_since_zone={frames_since_zone}",
                f"frames_since_seen={frames_since_seen} downward_exit={downward_exit}",
                f"ball_missing={ball_missing} timed_out={timed_out}",
            ]
            if tracked_centroid is not None:
                debug_lines.append(f"tracked=({tracked_centroid[0]:.1f}, {tracked_centroid[1]:.1f})")
            if last_inside_centroid is not None:
                debug_lines.append(f"last_inside=({last_inside_centroid[0]:.1f}, {last_inside_centroid[1]:.1f})")

            y0 = 90 if use_zone else 60
            for line in debug_lines:
                cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
                y0 += 18
        else:
            cv2.putText(vis, "Debug[d]: OFF", (10, 90 if use_zone else 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Overlay counts
        cv2.putText(vis, f"Makes: {len(made_events)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Show side-by-side mask for tuning
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combo = np.hstack([
            cv2.resize(vis, (w // 2, h // 2)),
            cv2.resize(mask_bgr, (w // 2, h // 2))
        ])

        cv2.imshow("Shot Counter (Left=Video, Right=Mask)", combo)
        cv2.imshow(tuner_win, np.zeros((1, 600, 3), dtype=np.uint8))  # just to keep trackbars visible

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('p'):
            # pause
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('p') or k2 == 27:
                    break
            if k2 == 27:
                break
        elif key == ord('d'):
            debug_overlays = not debug_overlays

        prev_inside_basket = inside_basket
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== RESULTS ===")
    print(f"Total makes counted: {len(made_events)}")
    if made_events:
        print("Make timestamps (seconds):")
        for fi, t in made_events:
            print(f"  frame {fi:6d}  ->  {t:8.3f}s")


if __name__ == "__main__":
    main()
