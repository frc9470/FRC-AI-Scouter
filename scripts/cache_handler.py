"""
Persistence helpers for polygon ROI caches.

The ROI cache is a JSON file keyed by video filename, storing basket and
zone polygon coordinates so the user doesn't have to redraw them every run.
"""

import json
import os

import numpy as np

# TODO: File handlers need more warnings logged to the console.
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
    except Exception as e:
        print(f"Warning: failed to parse ROI cache: {e}")
        return np.array([], dtype=np.int32)
    
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        print(f"Warning: ROI cache incorrectly formatted: {e}")
        return np.array([], dtype=np.int32)
    
    return poly
