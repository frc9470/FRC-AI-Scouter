import argparse
import base64
import os
import threading
import traceback
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, render_template_string

from shot_counter import (
    analyze_frame,
    build_analysis_context,
    get_hsv_bounds_from_config,
    get_roi_cache_path,
    load_roi_cache,
    make_runtime_config,
    poly_from_cache,
    render_analysis_frame,
    save_roi_cache,
    serialize_made_events,
)


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FRC Shot Counter</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #f59e0b;
      --accent-2: #22d3ee;
      --good: #22c55e;
      --bad: #ef4444;
      --border: rgba(148, 163, 184, 0.2);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(245, 158, 11, 0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(34, 211, 238, 0.15), transparent 30%),
        var(--bg);
      color: var(--text);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
    }
    main {
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto 48px;
      display: grid;
      gap: 20px;
    }
    .panel {
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
      backdrop-filter: blur(8px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.22);
    }
    h1, h2, p { margin: 0; }
    .header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      flex-wrap: wrap;
    }
    .subtle {
      color: var(--muted);
      font-size: 0.95rem;
      margin-top: 6px;
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }
    button {
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 999px;
      padding: 10px 14px;
      cursor: pointer;
      font: inherit;
    }
    button.active {
      border-color: rgba(245, 158, 11, 0.65);
      background: rgba(245, 158, 11, 0.16);
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    input {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--text);
      padding: 10px 12px;
      font: inherit;
    }
    .viewer {
      display: grid;
      gap: 12px;
    }
    canvas {
      width: 100%;
      max-height: 72vh;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #020617;
      display: block;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .swatch {
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 999px;
      margin-right: 6px;
    }
    #statusText {
      margin-top: 16px;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.85);
      color: var(--text);
      white-space: pre-wrap;
    }
    #resultsPanel[hidden] { display: none; }
    video {
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #000;
    }
    pre {
      margin: 0;
      padding: 14px;
      border-radius: 12px;
      background: #020617;
      overflow: auto;
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <div class="header">
        <div>
          <h1>FRC Shot Counter Web App</h1>
          <p class="subtle">Video: <span id="videoPath">loading...</span></p>
        </div>
        <div class="subtle">Draw a basket ROI, optionally mark a zone ROI, then run the HSV pipeline in the background.</div>
      </div>
      <div class="toolbar">
        <button id="loadFrameBtn">Load First Frame</button>
        <button id="basketModeBtn" class="active">Draw Basket ROI</button>
        <button id="zoneModeBtn">Draw Zone ROI</button>
        <button id="undoBtn">Undo Point</button>
        <button id="clearBtn">Clear Current ROI</button>
        <button id="startBtn">Start Processing</button>
      </div>
      <div class="grid">
        <label>H Min <input id="H_MIN" type="number" min="0" max="179"></label>
        <label>H Max <input id="H_MAX" type="number" min="0" max="179"></label>
        <label>S Min <input id="S_MIN" type="number" min="0" max="255"></label>
        <label>S Max <input id="S_MAX" type="number" min="0" max="255"></label>
        <label>V Min <input id="V_MIN" type="number" min="0" max="255"></label>
        <label>V Max <input id="V_MAX" type="number" min="0" max="255"></label>
      </div>
      <div id="statusText">Waiting for the first frame.</div>
    </section>

    <section class="panel viewer">
      <canvas id="frameCanvas"></canvas>
      <div class="legend">
        <span><span class="swatch" style="background:#f59e0b"></span>Basket ROI</span>
        <span><span class="swatch" style="background:#22d3ee"></span>Zone ROI</span>
        <span>Left click adds points. Basket ROI requires at least 3 points.</span>
      </div>
    </section>

    <section class="panel" id="resultsPanel" hidden>
      <h2>Results</h2>
      <div class="subtle" style="margin-bottom: 14px;">Annotated output video and counted makes.</div>
      <video id="outputVideo" controls></video>
      <pre id="resultsText"></pre>
    </section>
  </main>

  <script>
    const canvas = document.getElementById("frameCanvas");
    const ctx = canvas.getContext("2d");
    const frameImage = new Image();

    const state = {
      drawMode: "basket",
      basketPoints: [],
      zonePoints: [],
      imageLoaded: false,
      pollingHandle: null,
      imageRect: { x: 0, y: 0, width: 0, height: 0 },
    };

    const statusText = document.getElementById("statusText");
    const resultsPanel = document.getElementById("resultsPanel");
    const resultsText = document.getElementById("resultsText");
    const outputVideo = document.getElementById("outputVideo");
    const videoPathLabel = document.getElementById("videoPath");

    function setMode(mode) {
      state.drawMode = mode;
      document.getElementById("basketModeBtn").classList.toggle("active", mode === "basket");
      document.getElementById("zoneModeBtn").classList.toggle("active", mode === "zone");
      drawCanvas();
    }

    function currentPoints() {
      return state.drawMode === "basket" ? state.basketPoints : state.zonePoints;
    }

    function setCurrentPoints(points) {
      if (state.drawMode === "basket") {
        state.basketPoints = points;
      } else {
        state.zonePoints = points;
      }
    }

    function getConfig() {
      const keys = ["H_MIN", "H_MAX", "S_MIN", "S_MAX", "V_MIN", "V_MAX"];
      const config = {};
      for (const key of keys) {
        config[key] = Number(document.getElementById(key).value);
      }
      return config;
    }

    function setConfig(config) {
      for (const [key, value] of Object.entries(config)) {
        const input = document.getElementById(key);
        if (input) input.value = value;
      }
    }

    function drawPolygon(points, color, closed) {
      if (!points.length) return;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      points.forEach((point, index) => {
        const x = state.imageRect.x + point[0] * state.imageRect.width / frameImage.naturalWidth;
        const y = state.imageRect.y + point[1] * state.imageRect.height / frameImage.naturalHeight;
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      if (closed && points.length >= 3) ctx.closePath();
      ctx.stroke();

      for (const point of points) {
        const x = state.imageRect.x + point[0] * state.imageRect.width / frameImage.naturalWidth;
        const y = state.imageRect.y + point[1] * state.imageRect.height / frameImage.naturalHeight;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function drawCanvas() {
      const width = canvas.clientWidth || 960;
      const height = Math.max(540, Math.round(width * 9 / 16));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      if (!state.imageLoaded) {
        ctx.fillStyle = "#94a3b8";
        ctx.font = "18px sans-serif";
        ctx.fillText("Load the first frame to start drawing ROIs.", 24, 40);
        return;
      }

      const scale = Math.min(canvas.width / frameImage.naturalWidth, canvas.height / frameImage.naturalHeight);
      const drawWidth = Math.round(frameImage.naturalWidth * scale);
      const drawHeight = Math.round(frameImage.naturalHeight * scale);
      const offsetX = Math.round((canvas.width - drawWidth) / 2);
      const offsetY = Math.round((canvas.height - drawHeight) / 2);
      state.imageRect = { x: offsetX, y: offsetY, width: drawWidth, height: drawHeight };

      ctx.drawImage(frameImage, offsetX, offsetY, drawWidth, drawHeight);
      drawPolygon(state.basketPoints, "#f59e0b", true);
      drawPolygon(state.zonePoints, "#22d3ee", true);

      ctx.fillStyle = "rgba(2, 6, 23, 0.82)";
      ctx.fillRect(18, 18, 320, 68);
      ctx.fillStyle = "#e5e7eb";
      ctx.font = "16px sans-serif";
      ctx.fillText(`Mode: ${state.drawMode === "basket" ? "Basket ROI" : "Zone ROI"}`, 30, 44);
      ctx.fillText(`Basket points: ${state.basketPoints.length} | Zone points: ${state.zonePoints.length}`, 30, 68);
    }

    function canvasToImagePoint(event) {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const r = state.imageRect;
      if (!state.imageLoaded || x < r.x || y < r.y || x > r.x + r.width || y > r.y + r.height) {
        return null;
      }
      const imgX = Math.round((x - r.x) * frameImage.naturalWidth / r.width);
      const imgY = Math.round((y - r.y) * frameImage.naturalHeight / r.height);
      return [imgX, imgY];
    }

    async function loadFirstFrame() {
      const response = await fetch("/api/first_frame");
      const data = await response.json();
      if (!response.ok) {
        statusText.textContent = data.error || "Failed to load the first frame.";
        return;
      }

      state.basketPoints = data.cached_basket || [];
      state.zonePoints = data.cached_zone || [];
      setConfig(data.config || {});
      videoPathLabel.textContent = data.video_path;
      frameImage.onload = () => {
        state.imageLoaded = true;
        drawCanvas();
      };
      frameImage.src = data.image;
      statusText.textContent = "First frame loaded. Draw the basket ROI, adjust HSV values if needed, then start processing.";
    }

    async function fetchResults() {
      const response = await fetch("/api/results");
      const data = await response.json();
      if (!response.ok) {
        statusText.textContent = data.error || "Failed to load results.";
        return;
      }

      resultsPanel.hidden = false;
      outputVideo.src = data.output_url || "";
      resultsText.textContent = JSON.stringify(data, null, 2);
    }

    async function pollStatus() {
      const response = await fetch("/api/status");
      const data = await response.json();
      if (!response.ok) {
        statusText.textContent = data.error || "Status request failed.";
        return;
      }

      const percent = data.total_frames > 0 ? ((data.progress / data.total_frames) * 100).toFixed(1) : "0.0";
      statusText.textContent =
        `Processing: ${data.is_processing}\\n` +
        `Finished: ${data.is_finished}\\n` +
        `Progress: ${data.progress}/${data.total_frames} (${percent}%)\\n` +
        `Total makes: ${data.total_makes}\\n` +
        `Candidates: ${data.current_candidates} | Trackable: ${data.current_trackable} | Visible tracks: ${data.current_visible_tracks}\\n` +
        `Last event: ${data.last_event || "none"}\\n` +
        `Error: ${data.error || "none"}`;

      if (data.is_finished || data.error) {
        clearInterval(state.pollingHandle);
        state.pollingHandle = null;
        await fetchResults();
      }
    }

    async function startProcessing() {
      if (state.basketPoints.length < 3) {
        statusText.textContent = "Basket ROI needs at least 3 points.";
        return;
      }

      resultsPanel.hidden = true;
      outputVideo.removeAttribute("src");
      outputVideo.load();

      const response = await fetch("/api/set_roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          basket_points: state.basketPoints,
          zone_points: state.zonePoints,
          config: getConfig(),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        statusText.textContent = data.error || "Failed to start processing.";
        return;
      }

      statusText.textContent = "Background processing started.";
      if (state.pollingHandle) clearInterval(state.pollingHandle);
      state.pollingHandle = setInterval(pollStatus, 1000);
      pollStatus();
    }

    document.getElementById("loadFrameBtn").addEventListener("click", loadFirstFrame);
    document.getElementById("basketModeBtn").addEventListener("click", () => setMode("basket"));
    document.getElementById("zoneModeBtn").addEventListener("click", () => setMode("zone"));
    document.getElementById("undoBtn").addEventListener("click", () => {
      const points = currentPoints().slice(0, -1);
      setCurrentPoints(points);
      drawCanvas();
    });
    document.getElementById("clearBtn").addEventListener("click", () => {
      setCurrentPoints([]);
      drawCanvas();
    });
    document.getElementById("startBtn").addEventListener("click", startProcessing);
    canvas.addEventListener("click", (event) => {
      const point = canvasToImagePoint(event);
      if (!point) return;
      const points = currentPoints().concat([point]);
      setCurrentPoints(points);
      drawCanvas();
    });
    window.addEventListener("resize", drawCanvas);
    loadFirstFrame();
  </script>
</body>
</html>
"""


def encode_frame_data_url(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Could not encode frame as JPEG.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def serialize_config(config):
    keys = ["H_MIN", "H_MAX", "S_MIN", "S_MAX", "V_MIN", "V_MAX"]
    return {key: int(config[key]) for key in keys}


def default_video_path():
    candidates = sorted((REPO_DIR / "assets").glob("*.mp4"))
    if candidates:
        return str(candidates[0])
    return str(REPO_DIR / "match_video.mp4")


def resolve_video_path(raw_path):
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    repo_candidate = REPO_DIR / candidate
    if repo_candidate.exists():
        return str(repo_candidate)

    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return str(cwd_candidate)

    return str(repo_candidate)


class State:
    lock = threading.Lock()
    video_path = default_video_path()
    first_frame = None
    basket_poly = None
    zone_poly = None
    config = make_runtime_config()
    is_processing = False
    is_finished = False
    error = None
    progress = 0
    total_frames = 1
    total_makes = 0
    current_candidates = 0
    current_trackable = 0
    current_visible_tracks = 0
    last_event = "none"
    recent_events = []
    made_events = []
    output_url = None


def get_cached_polys(video_path, frame_shape):
    cache = load_roi_cache(get_roi_cache_path())
    video_key = os.path.basename(video_path)
    cached = cache.get(video_key, {})
    cached_size = cached.get("frame_size", [])
    height, width = frame_shape[:2]
    size_matches = (
        isinstance(cached_size, list)
        and len(cached_size) == 2
        and int(cached_size[0]) == int(width)
        and int(cached_size[1]) == int(height)
    )
    if not size_matches:
        return [], []

    basket_poly = poly_from_cache(cached.get("basket_poly", []))
    zone_poly = poly_from_cache(cached.get("zone_poly", []))
    return basket_poly.tolist(), zone_poly.tolist()


def save_cached_polys(video_path, frame_shape, basket_points, zone_points):
    cache_path = get_roi_cache_path()
    cache = load_roi_cache(cache_path)
    video_key = os.path.basename(video_path)
    height, width = frame_shape[:2]
    cache[video_key] = {
        "frame_size": [int(width), int(height)],
        "basket_poly": basket_points,
        "zone_poly": zone_points if len(zone_points) >= 3 else [],
    }
    save_roi_cache(cache_path, cache)


def create_video_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer


def process_video_task(video_path, basket_points, zone_points, config_overrides):
    output_path = STATIC_DIR / "output.mp4"
    cap = None
    writer = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 1

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Could not read the first frame from the video.")

        height, width = first_frame.shape[:2]
        context = build_analysis_context(
            first_frame.shape,
            fps,
            basket_points,
            zone_points if len(zone_points) >= 3 else None,
            config_overrides=config_overrides,
        )

        save_cached_polys(video_path, first_frame.shape, basket_points, zone_points)

        with State.lock:
            State.total_frames = total_frames
            State.first_frame = first_frame.copy()
            State.error = None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        writer = create_video_writer(output_path, fps, width, height)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_result = analyze_frame(frame, frame_idx, context)
            annotated = render_analysis_frame(frame, frame_result, context)
            writer.write(annotated)

            with State.lock:
                State.progress = frame_idx + 1
                State.total_makes = len(context["made_events"])
                State.current_candidates = context["current_candidate_count"]
                State.current_trackable = context["current_trackable_count"]
                State.current_visible_tracks = context["current_visible_tracks"]
                State.last_event = context["last_event_text"]
                State.recent_events = list(context["recent_events"])

            frame_idx += 1

        output_url = f"/static/{output_path.name}?v={int(output_path.stat().st_mtime)}"
        with State.lock:
            State.is_processing = False
            State.is_finished = True
            State.output_url = output_url
            State.made_events = serialize_made_events(context["made_events"])
            State.total_makes = len(context["made_events"])
            State.current_candidates = context["current_candidate_count"]
            State.current_trackable = context["current_trackable_count"]
            State.current_visible_tracks = context["current_visible_tracks"]
            State.last_event = context["last_event_text"]
            State.recent_events = list(context["recent_events"])
            State.config = context["config"]
    except Exception as exc:
        traceback.print_exc()
        with State.lock:
            State.is_processing = False
            State.is_finished = True
            State.error = f"{exc.__class__.__name__}: {exc}"
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/first_frame")
def api_first_frame():
    video_path = State.video_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return jsonify({"error": f"Could not open video: {video_path}"}), 500

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Could not read the first frame."}), 500

    cached_basket, cached_zone = get_cached_polys(video_path, frame.shape)
    with State.lock:
        State.first_frame = frame.copy()

    return jsonify(
        {
            "image": encode_frame_data_url(frame),
            "video_path": video_path,
            "cached_basket": cached_basket,
            "cached_zone": cached_zone,
            "config": serialize_config(State.config),
        }
    )


@app.route("/api/set_roi", methods=["POST"])
def api_set_roi():
    payload = request.get_json(silent=True) or {}
    basket_points = payload.get("basket_points", [])
    zone_points = payload.get("zone_points", [])
    config_overrides = payload.get("config", {})

    if len(basket_points) < 3:
        return jsonify({"error": "Basket ROI needs at least 3 points."}), 400

    with State.lock:
        if State.is_processing:
            return jsonify({"error": "Processing is already running."}), 409

        State.basket_poly = basket_points
        State.zone_poly = zone_points if len(zone_points) >= 3 else []
        State.config = make_runtime_config(config_overrides)
        State.is_processing = True
        State.is_finished = False
        State.error = None
        State.progress = 0
        State.total_frames = 1
        State.total_makes = 0
        State.current_candidates = 0
        State.current_trackable = 0
        State.current_visible_tracks = 0
        State.last_event = "none"
        State.recent_events = []
        State.made_events = []
        State.output_url = None

        thread = threading.Thread(
            target=process_video_task,
            args=(State.video_path, basket_points, State.zone_poly, config_overrides),
            daemon=True,
        )
        thread.start()

    return jsonify({"success": True})


@app.route("/api/status")
def api_status():
    with State.lock:
        return jsonify(
            {
                "is_processing": State.is_processing,
                "is_finished": State.is_finished,
                "error": State.error,
                "progress": State.progress,
                "total_frames": State.total_frames,
                "total_makes": State.total_makes,
                "current_candidates": State.current_candidates,
                "current_trackable": State.current_trackable,
                "current_visible_tracks": State.current_visible_tracks,
                "last_event": State.last_event,
            }
        )


@app.route("/api/results")
def api_results():
    with State.lock:
        if not State.is_finished and not State.error:
            return jsonify({"error": "Processing has not finished yet."}), 409

        lower, upper = get_hsv_bounds_from_config(State.config)
        return jsonify(
            {
                "video_path": State.video_path,
                "total_makes": State.total_makes,
                "made_events": State.made_events,
                "recent_events": State.recent_events,
                "output_url": State.output_url,
                "basket_points": State.basket_poly,
                "zone_points": State.zone_poly,
                "config": serialize_config(State.config),
                "lower_hsv": lower.tolist(),
                "upper_hsv": upper.tolist(),
                "error": State.error,
            }
        )


def main():
    parser = argparse.ArgumentParser(description="Web app for the HSV-based FRC shot counter.")
    parser.add_argument("--video", default=default_video_path(), help="Path to the input video.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the Flask app to.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the Flask app to.")
    args = parser.parse_args()

    with State.lock:
        State.video_path = resolve_video_path(args.video)

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
