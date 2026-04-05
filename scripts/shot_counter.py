"""
Semi-automatic shot counter for a robotics match video.

What it does:
- Lets you outline the HUB opening as a polygon ROI (via left mouse clicks).
- Lets you tune an HSV color range for the FUEL (via trackbars).
- Tracks FUEL centroids via simple color segmentation + contour filtering.
- Counts a successful shot when a FUEL centroid enters the HUB ROI and then disappears
  (or exits downward) within a short time window.

Notes:
- This does NOT automatically identify team 9470's robot. To filter only 9470's shots,
  you can optionally define an additional "robot shooting zone" ROI near where 9470
  stands and only count makes when a FUEL originated from that zone.
- Works best when FUEL have a distinct color (tune to yellow)
  and the HUB is stationary (this means the camera POV should be stationary).

Installation:
- `pip install opencv-python numpy yt-dlp`, OR
- set up a conda environment using `environment.yml`
Run:
- `cd` into the project directory
- `./.venv/bin/activate` OR `conda activate <env>` (based on setup used above)
- `python src/shot_counter.py assets/<path/to/video.mov>`
"""

import os
import sys

import cv2
import numpy as np

from cache_handler import get_roi_cache_path, load_roi_cache, save_roi_cache, poly_from_cache
from config import CONFIG
from utils import get_screen_size

def resize_to_fit(image, max_w, max_h):
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return image
    scale = min(max_w / w, max_h / h)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

# ============================
# Helper: Window Management
# ============================
def center_window(win_name, window_w, window_h, screen_w, screen_h):
    """Center an OpenCV window on the screen."""
    x = max(0, (screen_w - window_w) // 2)
    y = max(0, (screen_h - window_h) // 2)
    cv2.moveWindow(win_name, x, y)


# ============================
# Helpers: Image Processing
# ============================
def process_mask(mask):
    """Apply morphological operations to clean the binary mask."""
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    return mask


def extract_candidates_from_contours(contours, config):
    """Extract ball candidate objects from contours based on shape/area criteria."""
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config["MIN_CONTOUR_AREA"] or area > config["MAX_CONTOUR_AREA"]:
            continue
        (x, y, ww, hh) = cv2.boundingRect(c)
        cx = x + ww / 2.0
        cy = y + hh / 2.0
        perimeter = cv2.arcLength(c, True)
        circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 1e-6 else 0.0
        aspect = (ww / float(hh)) if hh > 0 else 0.0
        trackable = (
            area <= config["MAX_TRACKABLE_AREA"]
            and circularity >= config["MIN_TRACKABLE_CIRCULARITY"]
            and 0.45 <= aspect <= 2.2
        )
        candidates.append({
            "area": area,
            "centroid": (cx, cy),
            "bbox": (x, y, ww, hh),
            "circularity": circularity,
            "trackable": trackable,
        })
    return candidates


# ============================
# Helpers: Object Tracking
# ============================
def match_tracks_to_candidates(tracks, candidates, config, frame_idx):
    """Associate existing tracks with detected candidates using nearest-neighbor matching."""
    active_track_ids = [
        tid for tid, tr in tracks.items()
        if (frame_idx - tr["last_seen_frame"]) <= config["TRACK_MAX_MISSED"]
    ]
    trackable_indices = [i for i, cand in enumerate(candidates) if cand["trackable"]]

    pairs = []
    for tid in active_track_ids:
        tx, ty = tracks[tid]["centroid"]
        for ci in trackable_indices:
            cx, cy = candidates[ci]["centroid"]
            d = float(np.hypot(cx - tx, cy - ty))
            if d <= config["MAX_TRACK_DIST"]:
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
        if motion >= config["MIN_TRACK_MOTION_PX"]:
            track["last_motion_frame"] = frame_idx

        assigned_tracks.add(tid)
        assigned_candidates.add(ci)

    return assigned_tracks, assigned_candidates, trackable_indices


def spawn_new_tracks(candidates, assigned_candidates, trackable_indices, use_zone, zone_poly,
                     frame_idx, next_track_id, tracks):
    """Create new tracks for unmatched candidates."""
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
    return next_track_id


def update_track_state(track, frame_idx, basket_poly, zone_poly, use_zone, config,
                       zone_window_frames, airborne_y_max):
    """Calculate and update track state properties for current frame."""
    visible = (frame_idx - track["last_seen_frame"]) <= 1
    zone_ok = True
    if use_zone:
        zone_ok = (frame_idx - track["last_seen_in_zone_frame"]) <= zone_window_frames
    just_spawned = (frame_idx - track["first_seen_frame"]) <= 2
    recently_moving = ((frame_idx - track["last_motion_frame"]) <= config["MOTION_MEMORY_FRAMES"]) or just_spawned
    airborne = track["centroid"][1] <= airborne_y_max

    inside_basket = visible and point_in_poly(track["centroid"], basket_poly)
    entering_basket = inside_basket and not track["prev_inside_basket"]

    track["visible"] = visible
    track["inside_basket"] = inside_basket
    track["zone_ok"] = zone_ok
    track["recently_moving"] = recently_moving
    track["airborne"] = airborne
    track["preferred"] = visible and recently_moving and airborne
    track["prev_inside_basket"] = inside_basket

    return entering_basket, zone_ok, recently_moving, airborne


def handle_make_detection(track, frame_idx, fps, entering_basket, use_zone,
                          zone_ok, recently_moving, airborne, config, made_events,
                          push_event_fn):
    """Check if ball made and record if so; returns True if make detected."""
    if frame_idx >= track["cooldown_until"] and entering_basket:
        entry_zone_ok = (not use_zone) or (not config["STRICT_ZONE_GATE"]) or zone_ok
        entry_ok = entry_zone_ok and (recently_moving or airborne)
        if entry_ok:
            t = frame_idx / fps
            made_events.append((frame_idx, t, track["id"]))
            track["cooldown_until"] = frame_idx + int(config["TRACK_COOLDOWN_FRAMES_FACTOR"] * fps)
            push_event_fn(
                f"[MAKEDBG] f{frame_idx} T{track['id']} MAKE (entry) "
                f"(zone_ok={zone_ok}, strict_zone={config['STRICT_ZONE_GATE']}, moving={recently_moving}, airborne={airborne})"
            )
            return True
        else:
            push_event_fn(
                f"[MAKEDBG] f{frame_idx} T{track['id']} REJECT enter "
                f"(zone_ok={zone_ok}, strict_zone={config['STRICT_ZONE_GATE']}, moving={recently_moving}, airborne={airborne})"
            )
    return False


# ============================
# Helpers: Visualization
# ============================
def draw_rois(vis, basket_poly, zone_poly, use_zone):
    """Draw basket and zone ROIs on visualization."""
    cv2.polylines(vis, [basket_poly], True, (0, 255, 255), 2)
    cv2.putText(vis, "Basket ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if use_zone:
        cv2.polylines(vis, [zone_poly], True, (255, 255, 0), 2)
        cv2.putText(vis, "9470 Zone ROI", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)


def draw_candidates(vis, candidates, debug_overlays):
    """Draw candidate contours on visualization."""
    if not debug_overlays:
        return
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


def draw_tracks(vis, tracks, frame_idx, made_track_id_this_frame, config):
    """Draw tracked objects on visualization."""
    for tid, track in sorted(tracks.items()):
        age = frame_idx - track["last_seen_frame"]
        x, y, ww, hh = track["bbox"]
        cxy = track["centroid"]

        if made_track_id_this_frame is not None and tid == made_track_id_this_frame:
            color = (255, 0, 0)
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
        if config["STRICT_ZONE_GATE"] and not track["zone_ok"]:
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


def draw_debug_info(vis, candidates, tracks, frame_idx, use_zone, config, h,
                    last_event_text, recent_events, made_events):
    """Draw debug information on visualization."""
    if config["DEBUG_OVERLAYS"]:
        visible_tracks = [tr for tr in tracks.values() if tr["visible"]]
        preferred_tracks = [tr for tr in visible_tracks if tr["preferred"]]
        inside_ids = [tid for tid, tr in tracks.items() if tr["inside_basket"]]
        trackable_count = sum(1 for c in candidates if c["trackable"])
        debug_lines = [
            f"Debug[d]: ON  candidates={len(candidates)} trackable={trackable_count}",
            f"tracks={len(tracks)} visible={len(visible_tracks)} preferred={len(preferred_tracks)}",
            f"inside_roi_tracks={inside_ids[:6]} use_zone={use_zone}",
            f"motion_px>={config['MIN_TRACK_MOTION_PX']:.1f}",
            f"strict_zone_gate[z]={config['STRICT_ZONE_GATE']}",
            f"auto_pause_on_make[w]={config['AUTO_PAUSE_ON_MAKE']}",
            f"make_debug_print[m]={config['MAKE_DEBUG_PRINT']}",
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

    cv2.putText(vis, f"Makes: {len(made_events)}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


def handle_keyboard_input(key, config, frame_idx, push_event_fn):
    """Handle keyboard input; returns 'exit', 'pause', or None."""
    if key == 27:  # ESC
        return "exit"
    elif key == ord('p'):
        return "pause"
    elif key == ord('d'):
        config["DEBUG_OVERLAYS"] = not config["DEBUG_OVERLAYS"]
    elif key == ord('m'):
        config["MAKE_DEBUG_PRINT"] = not config["MAKE_DEBUG_PRINT"]
        push_event_fn(f"[MAKEDBG] f{frame_idx} console logging {'ON' if config['MAKE_DEBUG_PRINT'] else 'OFF'}")
    elif key == ord('z'):
        config["STRICT_ZONE_GATE"] = not config["STRICT_ZONE_GATE"]
        push_event_fn(f"[MAKEDBG] f{frame_idx} strict zone gate {'ON' if config['STRICT_ZONE_GATE'] else 'OFF'}")
    elif key == ord('w'):
        config["AUTO_PAUSE_ON_MAKE"] = not config["AUTO_PAUSE_ON_MAKE"]
        push_event_fn(f"[MAKEDBG] f{frame_idx} auto pause on make {'ON' if config['AUTO_PAUSE_ON_MAKE'] else 'OFF'}")
    return None


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

        screen_w_approx, screen_h_approx = get_screen_size()
        center_window(self.win, target_w, target_h, screen_w_approx, screen_h_approx)

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
def make_hsv_tuner(win="HSV Tuner", window_size=(1000, 260), screen_dims=None):
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, window_size[0], window_size[1])
    if screen_dims:
        center_window(win, window_size[0], window_size[1], screen_dims[0], screen_dims[1])

    def nothing(x):
        pass

    # Reasonable defaults for "orange ball" — tune as needed
    # OLD HSV -- H (5, 25); S (120, 255); V (120, 255)
    # NEW HSV -- H (20, 32); S (150, 255); V (150, 255)
    cv2.createTrackbar("H min", win, CONFIG["H_MIN"], 179, nothing)
    cv2.createTrackbar("H max", win, CONFIG["H_MAX"], 179, nothing)
    cv2.createTrackbar("S min", win, CONFIG["S_MIN"], 255, nothing)
    cv2.createTrackbar("S max", win, CONFIG["S_MAX"], 255, nothing)
    cv2.createTrackbar("V min", win, CONFIG["V_MIN"], 255, nothing)
    cv2.createTrackbar("V max", win, CONFIG["V_MAX"], 255, nothing)

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

    screen_w_full, screen_h_full = get_screen_size()
    center_window(win_name, target_w, target_h, screen_w_full, screen_h_full)

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
        print("Usage: python scripts/shot_counter.py /path/to/video")
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
    tuner_win = make_hsv_tuner(window_size=(hsv_win_w, hsv_win_h), screen_dims=(screen_w, screen_h))
    main_win = "Shot Counter (Left=Video, Right=Mask)"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(main_win, screen_w, screen_h)
    center_window(main_win, hsv_win_w, hsv_win_h, screen_w, screen_h)

    # Pre-compute frame-dependent config values
    CONFIG["MOTION_MEMORY_FRAMES"] = int(CONFIG["MOTION_MEMORY_FRAMES_FACTOR"] * fps)
    CONFIG["TRACK_COOLDOWN_FRAMES"] = int(CONFIG["TRACK_COOLDOWN_FRAMES_FACTOR"] * fps)

    # Tracks are keyed by integer ID and updated with nearest-neighbor matching.
    tracks = {}
    next_track_id = 1

    zone_window_frames = int(CONFIG["ZONE_WINDOW_FRAMES_FACTOR"] * fps)
    airborne_y_max = int(CONFIG["AIRBORNE_Y_MAX_FACTOR"] * h)

    made_events = []  # list of (frame_idx, seconds, track_id)
    frame_idx = 0
    last_event_text = "none"
    recent_events = []
    auto_pause_pending = False

    def push_event(msg):
        nonlocal last_event_text
        last_event_text = msg
        recent_events.append(msg)
        if len(recent_events) > CONFIG["EVENT_LOG_MAX"]:
            recent_events.pop(0)
        if CONFIG["MAKE_DEBUG_PRINT"]:
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

        mask = process_mask(mask)

        # Find candidate ball contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = extract_candidates_from_contours(contours, CONFIG)

        assigned_tracks, assigned_candidates, trackable_indices = match_tracks_to_candidates(
            tracks, candidates, CONFIG, frame_idx
        )

        # Handle zone updates for matched tracks
        for tid in assigned_tracks:
            if use_zone and point_in_poly(tracks[tid]["centroid"], zone_poly):
                tracks[tid]["last_seen_in_zone_frame"] = frame_idx

        next_track_id = spawn_new_tracks(
            candidates, assigned_candidates, trackable_indices, use_zone,
            zone_poly, frame_idx, next_track_id, tracks
        )

        # Per-track make logic: count a make immediately on ROI entry.
        made_track_id_this_frame = None
        for tid, track in tracks.items():
            entering_basket, zone_ok, recently_moving, airborne = update_track_state(
                track, frame_idx, basket_poly, zone_poly, use_zone, CONFIG,
                zone_window_frames, airborne_y_max
            )
            is_make = handle_make_detection(
                track, frame_idx, fps, entering_basket, use_zone, zone_ok,
                recently_moving, airborne, CONFIG, made_events, push_event
            )
            if is_make:
                made_track_id_this_frame = tid
                if CONFIG["AUTO_PAUSE_ON_MAKE"]:
                    auto_pause_pending = True

        # Drop very stale tracks to keep state compact.
        stale_track_ids = [
            tid for tid, tr in tracks.items()
            if (frame_idx - tr["last_seen_frame"]) > CONFIG["TRACK_MAX_MISSED"]
        ]
        for tid in stale_track_ids:
            del tracks[tid]

        # ----------------------------
        # Visualization
        # ----------------------------
        vis = frame.copy()
        draw_rois(vis, basket_poly, zone_poly, use_zone)
        draw_candidates(vis, candidates, CONFIG["DEBUG_OVERLAYS"])
        draw_tracks(vis, tracks, frame_idx, made_track_id_this_frame, CONFIG)
        draw_debug_info(vis, candidates, tracks, frame_idx, use_zone, CONFIG, h,
                        last_event_text, recent_events, made_events)

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
        key_action = handle_keyboard_input(key, CONFIG, frame_idx, push_event)
        if key_action == "exit":
            break
        elif key_action == "pause":
            while True:
                k2 = cv2.waitKey(0) & 0xFF
                if k2 == ord('p') or k2 == 27:
                    break
            if k2 == 27:
                break

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
