"""
main.py — Guitar Tab Recorder launcher
Starts an HTTP server with a session/device API, opens the browser UI,
and manages the GateRecorder lifecycle.
"""

import json
import os
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dissertation.predict import get_model
from dissertation.guitar_capture import GateRecorder

# ── Config ────────────────────────────────────────────────────────
PORT         = 8765
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

MODEL_PATH   = "checkpoints_v5/best_v5.pt"

# ── Globals ───────────────────────────────────────────────────────
model            = None
current_recorder = None
recorder_thread  = None
recorder_lock    = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def get_audio_devices():
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        devices = []
        try:
            default_info = pa.get_default_input_device_info()
            default_idx  = default_info["index"]
            default_ch   = default_info["maxInputChannels"]
        except OSError:
            default_idx = -1
            default_ch  = 0
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            ch   = int(info["maxInputChannels"])
            for c in range(ch):
                print(c)
            if ch > 0:
                devices.append({
                    "index":      i,
                    "name":       info["name"],
                    "channels":   ch,
                    "is_default": i == default_idx,
                    # List each channel as a selectable input
                    "inputs": [
                        {
                            "channel": 1,
                            "label":   f"Channel {c + 1}" if ch > 1 else "Mono",
                        }
                        for c in range(ch)
                    ],
                })
        pa.terminate()
        return devices
    except Exception as e:
        print(f"[devices] Warning: {e}")
        return []
 


def sanitize_name(name: str) -> str:
    """Strip anything that isn't alphanumeric, space, hyphen, or underscore."""
    return re.sub(r"[^\w\s\-]", "", name).strip()


def get_sessions():
    return sorted(p.stem for p in SESSIONS_DIR.glob("*.json"))


def create_session(name: str):
    safe = sanitize_name(name)
    if not safe:
        return None, "Invalid session name — use letters, numbers, spaces or hyphens."
    path = SESSIONS_DIR / f"{safe}.json"
    if not path.exists():
        path.write_text("[]")
    return safe, None


def start_recorder(session: str, device_index):
    global current_recorder, recorder_thread

    json_path = str(SESSIONS_DIR / f"{session}.json")

    with recorder_lock:
        # Stop any existing recorder cleanly
        if current_recorder is not None:
            current_recorder.stop()
            if recorder_thread and recorder_thread.is_alive():
                recorder_thread.join(timeout=8)

        current_recorder = GateRecorder(
            model      = model,
            output_dir = "recordings",
            gate_db    = 30,
            tail_ms    = 400,
            device     = device_index,
            json_path  = json_path,
        )
        recorder_thread = threading.Thread(
            target=current_recorder.run,
            daemon=True,
            name=f"recorder-{session}",
        )
        recorder_thread.start()


# ═══════════════════════════════════════════════════════════════════
#  HTTP Request Handler
# ═══════════════════════════════════════════════════════════════════

MIME = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json",
    ".css":  "text/css",
    ".js":   "application/javascript",
}


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # keep the terminal clean

    # ── Helpers ────────────────────────────────────────────────────

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str):
        try:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    # ── GET ────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path

        # API
        if path == "/api/devices":
            return self.send_json(get_audio_devices())

        if path == "/api/sessions":
            return self.send_json(get_sessions())

        if path == "/api/config":
            from dissertation.guitar_capture import CALIBRATE_SEC
            return self.send_json({"calibrate_sec": CALIBRATE_SEC})

        if path == "/api/stop":
            with recorder_lock:
                if current_recorder is not None:
                    current_recorder.stop()
            return self.send_json({"ok": True})

        if path == "/api/quit":
            self.send_json({"ok": True})
            def _shutdown():
                import time
                time.sleep(0.3)
                os._exit(0)
            threading.Thread(target=_shutdown, daemon=True).start()
            return

        # Session JSON files
        if path.startswith("/sessions/") and path.endswith(".json"):
            fname = path[len("/sessions/"):]
            return self.send_file(SESSIONS_DIR / fname, "application/json")

        # Static files
        if path in ("/", "/index.html"):
            return self.send_file(Path("guitar_tab.html"), "text/html; charset=utf-8")

        local = Path(path.lstrip("/"))
        ext   = local.suffix.lower()
        if ext in MIME and local.exists():
            return self.send_file(local, MIME[ext])

        self.send_response(404)
        self.end_headers()

    # ── POST ───────────────────────────────────────────────────────

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.read_body()

        if path == "/api/sessions/create":
            name = body.get("name", "")
            safe, err = create_session(name)
            if err:
                return self.send_json({"error": err}, 400)
            return self.send_json({"name": safe})

        if path == "/api/start":
            session = body.get("session")
            device  = body.get("device")  # None = system default

            if not session:
                return self.send_json({"error": "No session specified."}, 400)

            session_path = SESSIONS_DIR / f"{session}.json"
            if not session_path.exists():
                return self.send_json({"error": f"Session '{session}' not found."}, 404)

            try:
                start_recorder(session, device)
                # Block here until the recorder finishes calibrating
                current_recorder.calibration_done.wait(timeout=15)
                return self.send_json({"ok": True, "session": session})
            except Exception as e:
                return self.send_json({"error": str(e)}, 500)

        self.send_response(404)
        self.end_headers()


# ═══════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    global model

    print("Loading model...")
    model = get_model(MODEL_PATH)
    print("Model ready.\n")

    server = HTTPServer(("localhost", PORT), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{PORT}/"
    print(f"Server running at {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        with recorder_lock:
            if current_recorder:
                current_recorder.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
