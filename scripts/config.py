import os
from dotenv import load_dotenv

# Load .env from project root (one level above scripts/)
load_dotenv(
    os.path.join(
        os.path.dirname(__file__), '..', '.env'
    )
)

# ============================
# Configuration Variables (adjust as needed)
# ============================
CONFIG = {
    # Reject noise (increase if too many false detections)
    "MIN_CONTOUR_AREA": 80,
    # Reject giant blobs
    "MAX_CONTOUR_AREA": 20_000,
    # Prefer smaller blobs over giant fuel piles
    "MAX_TRACKABLE_AREA": 8_000,
    "MIN_TRACKABLE_CIRCULARITY": 0.20,
    "MAX_TRACK_DIST": 90, # in px; association radius
    "TRACK_MAX_MISSED": 8,
    "MIN_TRACK_MOTION_PX": 2.5,
    "MOTION_MEMORY_FRAMES_FACTOR": 0.50,
    "TRACK_COOLDOWN_FRAMES_FACTOR": 0.40,
    "EVENT_LOG_MAX": 12,
    # If True, require `zone_ok` for basket entry when zone ROI is enabled
    "STRICT_ZONE_GATE": False,
    # Toggle with 'w' -- pauses after every successful shot detected
    "AUTO_PAUSE_ON_MAKE": False,
    "MAKE_DEBUG_PRINT": True,
    "ZONE_WINDOW_FRAMES_FACTOR": 1.0,
    "AIRBORNE_Y_MAX_FACTOR": 0.86,
    "DEBUG_OVERLAYS": True,

    # HSV values (loaded from .env, with defaults for practice field)
    "H_MIN": int(os.environ.get("H_MIN", 5)),
    "H_MAX": int(os.environ.get("H_MAX", 25)),
    "S_MIN": int(os.environ.get("S_MIN", 120)),
    "S_MAX": int(os.environ.get("S_MAX", 255)),
    "V_MIN": int(os.environ.get("V_MIN", 120)),
    "V_MAX": int(os.environ.get("V_MAX", 255)),
}