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

import json
import os
import sys

import cv2
import numpy as np

from utils import get_screen_size

def resize_to_fit(image, max_w, max_h):
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    scale = min(max_w / w, max_h / h)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


# ----------------------------
# Utility: polygon ROI drawing
# ----------------------------
class PolyDrawer:
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
        x_img = (x - self.pad_x) / self.scale
        y_img = (y - self.pad_y) / self.scale
        x_img = int(np.clip(round(x_img), 0, self.img.shape[1] - 1))
        y_img = int(np.clip(round(y_img), 0, self.img.shape[0] - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x_img, y_img))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # undo last point
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
        cv2.setMouseCallback(self.win, self._mouse)

        while True:
            vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            resized = cv2.resize(self.img, (self.draw_w, self.draw_h), interpolation=cv2.INTER_LINEAR)
            vis[self.pad_y:self.pad_y + self.draw_h, self.pad_x:self.pad_x + self.draw_w] = resized
            # Draw points + edges
            for p in self.pts:
                sp = (
                    int(round(p[0] * self.scale + self.pad_x)),
                    int(round(p[1] * self.scale + self.pad_y)),
                )
                cv2.circle(vis, sp, 4, (0, 255, 0), -1)
            if len(self.pts) >= 2:
                disp_pts = np.array(
                    [
                        [
                            int(round(px * self.scale + self.pad_x)),
                            int(round(py * self.scale + self.pad_y)),
                        ]
                        for px, py in self.pts
                    ],
                    dtype=np.int32,
                )
                cv2.polylines(vis, [disp_pts], False, (0, 255, 0), 2)

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
def make_hsv_tuner(win="HSV Tuner", window_size=(1000, 260)):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, window_size[0], window_size[1])

    def nothing(x):
        pass

    # Reasonable defaults for "orange ball" — tune as needed
    # OLD HSV -- H (5, 25); S (120, 255); V (120, 255)
    cv2.createTrackbar("H min", win, 20, 179, nothing)
    cv2.createTrackbar("H max", win, 32, 179, nothing)
    cv2.createTrackbar("S min", win, 150, 255, nothing)
    cv2.createTrackbar("S max", win, 255, 255, nothing)
    cv2.createTrackbar("V min", win, 150, 255, nothing)
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
        print(f"Warning: failed to save ROI cache to {path}: {e}")


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


def confirm_cached_poly(frame, poly, win_name, title, prompt, display_size):
    img_h, img_w = frame.shape[:2]
    target_w, target_h = display_size
    scale = min(target_w / img_w, target_h / img_h)
    draw_w = max(1, int(round(img_w * scale)))
    draw_h = max(1, int(round(img_h * scale)))
    pad_x = max(0, (target_w - draw_w) // 2)
    pad_y = max(0, (target_h - draw_h) // 2)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, target_w, target_h)

    disp_pts = np.array(
        [[int(round(px * scale + pad_x)), int(round(py * scale + pad_y))] for px, py in poly],
        dtype=np.int32,
    )

    while True:
        vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        resized = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
        vis[pad_y:pad_y + draw_h, pad_x:pad_x + draw_w] = resized
        cv2.polylines(vis, [disp_pts], True, (0, 255, 255), 3)
        cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, prompt, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
    screen_w, screen_h = get_screen_size()
    cache_path = get_roi_cache_path()
    roi_cache = load_roi_cache(cache_path)
    video_key = os.path.basename(video_path)
    cached = roi_cache.get(video_key, {})

    cached_basket_poly = poly_from_cache(cached.get("basket_poly", []))
    cached_zone_poly = poly_from_cache(cached.get("zone_poly", []))
    cached_size = cached.get("frame_size", [])
    cache_size_match = (
        isinstance(cached_size, list)
        and len(cached_size) == 2
        and int(cached_size[0]) == int(w)
        and int(cached_size[1]) == int(h)
    )
    if not cache_size_match:
        cached_basket_poly = np.array([], dtype=np.int32)
        cached_zone_poly = np.array([], dtype=np.int32)

    # Ask user to define basket opening ROI (polygon)
    if cached_basket_poly.size != 0:
        choice = confirm_cached_poly(
            frame0,
            cached_basket_poly,
            "Cached Basket ROI",
            "Cached Basket ROI Found",
            "Y/Enter: use this ROI | N: redraw | ESC: cancel",
            display_size=(screen_w, screen_h),
        )
        if choice == "use":
            basket_poly = cached_basket_poly
        elif choice == "redraw":
            basket_poly = PolyDrawer(
                "Draw Basket ROI",
                "LEFT click: add point | RIGHT click: undo\n"
                "ENTER: finish (>=3 points) | ESC: cancel (empty)\n"
                "Tip: outline the basket opening / net area tightly.",
                display_size=(screen_w, screen_h),
            ).draw(frame0)
        else:
            print("Basket ROI selection canceled. Exiting.")
            sys.exit(0)
    else:
        basket_poly = PolyDrawer(
            "Draw Basket ROI",
            "LEFT click: add point | RIGHT click: undo\n"
            "ENTER: finish (>=3 points) | ESC: cancel (empty)\n"
            "Tip: outline the basket opening / net area tightly.",
            display_size=(screen_w, screen_h),
        ).draw(frame0)

    if basket_poly.size == 0:
        print("No basket ROI provided. Exiting.")
        sys.exit(0)

    # Optional: define a "9470 shooting zone" ROI to approximate "shots by 9470"
    # (Define where 9470 typically shoots from; count makes only if the ball was last seen in this zone.)
    if cached_zone_poly.size != 0:
        zone_choice = confirm_cached_poly(
            frame0,
            cached_zone_poly,
            "Cached 9470 Zone ROI",
            "Cached 9470 Zone ROI Found",
            "Y/Enter: use this ROI | N: redraw | S/ESC: skip zone",
            display_size=(screen_w, screen_h),
        )
        if zone_choice == "use":
            zone_poly = cached_zone_poly
        elif zone_choice == "redraw":
            zone_poly = PolyDrawer(
                "Optional: Draw 9470 Shooting Zone ROI",
                "OPTIONAL\n"
                "LEFT click: add point | RIGHT click: undo\n"
                "ENTER: finish (>=3 points) | ESC: skip\n"
                "Tip: roughly outline the floor area near robot 9470's shooter position.",
                display_size=(screen_w, screen_h),
            ).draw(frame0)
        else:
            zone_poly = np.array([], dtype=np.int32)
    else:
        zone_poly = PolyDrawer(
            "Optional: Draw 9470 Shooting Zone ROI",
            "OPTIONAL\n"
            "LEFT click: add point | RIGHT click: undo\n"
            "ENTER: finish (>=3 points) | ESC: skip\n"
            "Tip: roughly outline the floor area near robot 9470's shooter position.",
            display_size=(screen_w, screen_h),
        ).draw(frame0)

    use_zone = zone_poly.size != 0

    # Persist ROI selections for this video filename so future runs can reuse them.
    roi_cache[video_key] = {
        "frame_size": [int(w), int(h)],
        "basket_poly": basket_poly.tolist(),
        "zone_poly": zone_poly.tolist() if use_zone else [],
    }
    save_roi_cache(cache_path, roi_cache)

    # HSV tuner
    hsv_win_w = min(1200, max(900, int(screen_w * 0.6)))
    hsv_win_h = max(220, int(screen_h * 0.25))
    tuner_win = make_hsv_tuner(window_size=(hsv_win_w, hsv_win_h))
    main_win = "Shot Counter (Left=Video, Right=Mask)"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_win, screen_w, screen_h)

    # Tunables (adjust as needed)
    MIN_CONTOUR_AREA = 80    # reject noise (increase if too many false detections)
    MAX_CONTOUR_AREA = 20_000  # reject giant blobs
    MAX_TRACKABLE_AREA = 8_000  # prefer smaller blobs over giant fuel piles
    MIN_TRACKABLE_CIRCULARITY = 0.20
    MAX_TRACK_DIST = 90      # px; association radius
    TRACK_MAX_MISSED = 8
    MIN_TRACK_MOTION_PX = 2.5
    MOTION_MEMORY_FRAMES = int(0.50 * fps)
    TRACK_COOLDOWN_FRAMES = int(0.40 * fps)
    MAKE_DEBUG_PRINT = True
    EVENT_LOG_MAX = 12
    STRICT_ZONE_GATE = False  # if True, require zone_ok for basket entry when zone ROI is enabled
    AUTO_PAUSE_ON_MAKE = False  # toggle with 'w'

    # Tracks are keyed by integer ID and updated with nearest-neighbor matching.
    tracks = {}
    next_track_id = 1

    zone_window_frames = int(1.0 * fps)
    airborne_y_max = int(0.86 * h)

    made_events = []  # list of (frame_idx, seconds, track_id)
    frame_idx = 0
    debug_overlays = True
    last_event_text = "none"
    recent_events = []
    auto_pause_pending = False

    def push_event(msg):
        nonlocal last_event_text
        last_event_text = msg
        recent_events.append(msg)
        if len(recent_events) > EVENT_LOG_MAX:
            recent_events.pop(0)
        if MAKE_DEBUG_PRINT:
            print(msg)

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
            perimeter = cv2.arcLength(c, True)
            circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 1e-6 else 0.0
            aspect = (ww / float(hh)) if hh > 0 else 0.0
            trackable = (
                area <= MAX_TRACKABLE_AREA
                and circularity >= MIN_TRACKABLE_CIRCULARITY
                and 0.45 <= aspect <= 2.2
            )
            candidates.append(
                {
                    "area": area,
                    "centroid": (cx, cy),
                    "bbox": (x, y, ww, hh),
                    "circularity": circularity,
                    "trackable": trackable,
                }
            )

        # Multi-object nearest-neighbor association
        active_track_ids = [
            tid for tid, tr in tracks.items()
            if (frame_idx - tr["last_seen_frame"]) <= TRACK_MAX_MISSED
        ]
        trackable_indices = [i for i, cand in enumerate(candidates) if cand["trackable"]]
        pairs = []
        for tid in active_track_ids:
            tx, ty = tracks[tid]["centroid"]
            for ci in trackable_indices:
                cx, cy = candidates[ci]["centroid"]
                d = float(np.hypot(cx - tx, cy - ty))
                if d <= MAX_TRACK_DIST:
                    pairs.append((d, tid, ci))
        pairs.sort(key=lambda t: t[0])

        assigned_tracks = set()
        assigned_candidates = set()
        for _, tid, ci in pairs:
            if tid in assigned_tracks or ci in assigned_candidates:
                continue
            track = tracks[tid]
            cand = candidates[ci]
            prev_centroid = track["centroid"]
            new_centroid = cand["centroid"]
            motion = float(np.hypot(new_centroid[0] - prev_centroid[0], new_centroid[1] - prev_centroid[1]))

            track["centroid"] = new_centroid
            track["bbox"] = cand["bbox"]
            track["area"] = cand["area"]
            track["last_seen_frame"] = frame_idx
            track["motion_ema"] = 0.65 * track["motion_ema"] + 0.35 * motion
            if motion >= MIN_TRACK_MOTION_PX:
                track["last_motion_frame"] = frame_idx
            if use_zone and point_in_poly(new_centroid, zone_poly):
                track["last_seen_in_zone_frame"] = frame_idx

            assigned_tracks.add(tid)
            assigned_candidates.add(ci)

        # Spawn new tracks for unmatched trackable candidates.
        for ci in trackable_indices:
            if ci in assigned_candidates:
                continue
            cand = candidates[ci]
            zone_seen_frame = -10_000
            if use_zone and point_in_poly(cand["centroid"], zone_poly):
                zone_seen_frame = frame_idx
            tracks[next_track_id] = {
                "id": next_track_id,
                "centroid": cand["centroid"],
                "bbox": cand["bbox"],
                "area": cand["area"],
                "first_seen_frame": frame_idx,
                "last_seen_frame": frame_idx,
                "last_seen_in_zone_frame": zone_seen_frame,
                "motion_ema": 0.0,
                "last_motion_frame": -10_000,
                "inside_basket": False,
                "prev_inside_basket": False,
                "cooldown_until": -1,
                "visible": True,
                "preferred": False,
                "zone_ok": True,
                "recently_moving": False,
                "airborne": False,
            }
            next_track_id += 1

        # Per-track make logic: count a make immediately on ROI entry.
        made_track_id_this_frame = None
        for tid, track in tracks.items():
            visible = (frame_idx - track["last_seen_frame"]) <= 1
            zone_ok = True
            if use_zone:
                zone_ok = (frame_idx - track["last_seen_in_zone_frame"]) <= zone_window_frames
            just_spawned = (frame_idx - track["first_seen_frame"]) <= 2
            recently_moving = ((frame_idx - track["last_motion_frame"]) <= MOTION_MEMORY_FRAMES) or just_spawned
            airborne = track["centroid"][1] <= airborne_y_max

            inside_basket = visible and point_in_poly(track["centroid"], basket_poly)
            entering_basket = inside_basket and not track["prev_inside_basket"]
            if frame_idx >= track["cooldown_until"] and entering_basket:
                entry_zone_ok = (not use_zone) or (not STRICT_ZONE_GATE) or zone_ok
                entry_ok = entry_zone_ok and (recently_moving or airborne)
                if entry_ok:
                    t = frame_idx / fps
                    made_events.append((frame_idx, t, tid))
                    track["cooldown_until"] = frame_idx + TRACK_COOLDOWN_FRAMES
                    made_track_id_this_frame = tid
                    if AUTO_PAUSE_ON_MAKE:
                        auto_pause_pending = True
                    push_event(
                        f"[MAKEDBG] f{frame_idx} T{tid} MAKE (entry) "
                        f"(zone_ok={zone_ok}, strict_zone={STRICT_ZONE_GATE}, moving={recently_moving}, airborne={airborne})"
                    )
                else:
                    push_event(
                        f"[MAKEDBG] f{frame_idx} T{tid} REJECT enter "
                        f"(zone_ok={zone_ok}, strict_zone={STRICT_ZONE_GATE}, moving={recently_moving}, airborne={airborne})"
                    )

            track["visible"] = visible
            track["inside_basket"] = inside_basket
            track["zone_ok"] = zone_ok
            track["recently_moving"] = recently_moving
            track["airborne"] = airborne
            track["preferred"] = visible and recently_moving and airborne
            track["prev_inside_basket"] = inside_basket

        # Drop very stale tracks to keep state compact.
        stale_track_ids = [
            tid for tid, tr in tracks.items()
            if (frame_idx - tr["last_seen_frame"]) > TRACK_MAX_MISSED
        ]
        for tid in stale_track_ids:
            del tracks[tid]

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
            for idx, cand in enumerate(candidates):
                area = cand["area"]
                cxy = cand["centroid"]
                bb = cand["bbox"]
                circ = cand["circularity"]
                trackable = cand["trackable"]
                x, y, ww, hh = bb
                color = (0, 255, 0) if trackable else (0, 180, 180)
                cv2.rectangle(vis, (x, y), (x + ww, y + hh), color, 1)
                cv2.circle(vis, (int(cxy[0]), int(cxy[1])), 3, color, -1)
                cv2.putText(
                    vis,
                    f"c{idx} a={int(area)} cir={circ:.2f}",
                    (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Draw all tracked objects (red = preferred moving airborne tracks)
        for tid, track in sorted(tracks.items()):
            age = frame_idx - track["last_seen_frame"]
            x, y, ww, hh = track["bbox"]
            cxy = track["centroid"]

            if made_track_id_this_frame is not None and tid == made_track_id_this_frame:
                color = (255, 0, 0)  # blue: counted as make recently
                thickness = 3
            elif age <= 1 and track["preferred"]:
                color = (0, 0, 255)
                thickness = 2
            elif age <= 1:
                color = (0, 165, 255)
                thickness = 2
            else:
                color = (140, 140, 140)
                thickness = 1

            cv2.rectangle(vis, (x, y), (x + ww, y + hh), color, thickness)
            cv2.circle(vis, (int(cxy[0]), int(cxy[1])), 5, color, -1)
            label = f"T{tid} m={track['motion_ema']:.1f}"
            if track["inside_basket"]:
                label += " IN"
            if made_track_id_this_frame is not None and tid == made_track_id_this_frame:
                label += " MADE"
            if STRICT_ZONE_GATE and not track["zone_ok"]:
                label += " Z0"
            if track["preferred"]:
                label += " P"
            cv2.putText(
                vis,
                label,
                (x, y + hh + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        if debug_overlays:
            visible_tracks = [tr for tr in tracks.values() if tr["visible"]]
            preferred_tracks = [tr for tr in visible_tracks if tr["preferred"]]
            inside_ids = [tid for tid, tr in tracks.items() if tr["inside_basket"]]
            debug_lines = [
                f"Debug[d]: ON  candidates={len(candidates)} trackable={len(trackable_indices)}",
                f"tracks={len(tracks)} visible={len(visible_tracks)} preferred={len(preferred_tracks)}",
                f"inside_roi_tracks={inside_ids[:6]} use_zone={use_zone}",
                f"airborne_y_max={airborne_y_max} motion_px>={MIN_TRACK_MOTION_PX}",
                f"strict_zone_gate[z]={STRICT_ZONE_GATE}",
                f"auto_pause_on_make[w]={AUTO_PAUSE_ON_MAKE}",
                f"make_debug_print[m]={MAKE_DEBUG_PRINT}",
                f"last_event={last_event_text}",
            ]
            y0 = 90 if use_zone else 60
            for line in debug_lines:
                cv2.putText(vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
                y0 += 18
            event_lines = recent_events[-4:]
            for ev in event_lines:
                cv2.putText(vis, ev[:140], (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)
                y0 += 16
        else:
            cv2.putText(vis, "Debug[d]: OFF", (10, 90 if use_zone else 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Overlay counts
        cv2.putText(vis, f"Makes: {len(made_events)}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # Show side-by-side mask for tuning
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combo = np.hstack([vis, mask_bgr])
        combo = resize_to_fit(combo, screen_w, screen_h)

        cv2.imshow(main_win, combo)

        # Keep trackbars visible on a larger canvas and show current HSV bounds.
        tuner_canvas = np.zeros((hsv_win_h, hsv_win_w, 3), dtype=np.uint8)
        cv2.putText(
            tuner_canvas,
            f"Lower HSV: ({int(lower[0])}, {int(lower[1])}, {int(lower[2])})",
            (20, max(24, hsv_win_h - 56)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tuner_canvas,
            f"Upper HSV: ({int(upper[0])}, {int(upper[1])}, {int(upper[2])})",
            (20, max(24, hsv_win_h - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(tuner_win, tuner_canvas)

        if auto_pause_pending:
            auto_pause_pending = False
            push_event(f"[MAKEDBG] f{frame_idx} auto-paused on make; press 'p' to resume")
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('p') or k2 == 27:
                    break
            if k2 == 27:
                break

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
        elif key == ord('m'):
            MAKE_DEBUG_PRINT = not MAKE_DEBUG_PRINT
            push_event(f"[MAKEDBG] f{frame_idx} console logging {'ON' if MAKE_DEBUG_PRINT else 'OFF'}")
        elif key == ord('z'):
            STRICT_ZONE_GATE = not STRICT_ZONE_GATE
            push_event(f"[MAKEDBG] f{frame_idx} strict zone gate {'ON' if STRICT_ZONE_GATE else 'OFF'}")
        elif key == ord('w'):
            AUTO_PAUSE_ON_MAKE = not AUTO_PAUSE_ON_MAKE
            push_event(f"[MAKEDBG] f{frame_idx} auto pause on make {'ON' if AUTO_PAUSE_ON_MAKE else 'OFF'}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== RESULTS ===")
    if made_events:
        print("Make timestamps (seconds):")
        for fi, t, tid in made_events:
            print(f"  track {tid:3d}  frame {fi:6d}  ->  {t:8.3f}s")
    print(f"\n\nTotal makes counted: {len(made_events)} || file {video_path}")


if __name__ == "__main__":
    main()
