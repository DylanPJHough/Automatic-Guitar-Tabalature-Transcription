

import argparse
import os
import queue
import json
import sys
import threading
import wave
from datetime import datetime

import numpy as np

from dissertation.predict import get_string_fret_from_wav


# ── constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 44100
CHANNELS       = 1
CHUNK          = 512          # ~11.6 ms per read
CALIBRATE_SEC  = 2.0          # seconds of silence for noise floor measurement
PRE_ROLL_MS    = 30           # ms of audio kept before gate opens (captures attack)
MIN_NOTE_MS    = 60           # notes shorter than this are discarded (noise blips)

# ── helpers ────────────────────────────────────────────────────────────────────

def rms_db(chunk: np.ndarray):
    rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-9))


def save_wav(frames: list, path: str):
    audio = np.concatenate(frames).astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return len(audio) / SAMPLE_RATE


def list_devices():
    try:
        import pyaudio
    except ImportError:
        sys.exit("pyaudio not installed — run:  pip install pyaudio")
    pa = pyaudio.PyAudio()
    default = -1
    try:
        default = pa.get_default_input_device_info()["index"]
    except OSError:
        pass
    print("\nInput devices:")
    print("─" * 50)
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            tag = "  ← default" if i == default else ""
            print(f"  [{i:2d}]  {info['name']}{tag}")
    pa.terminate()
    print()


# ── recorder ───────────────────────────────────────────────────────────────────

class GateRecorder:
    def __init__(self, model, output_dir, gate_db, tail_ms, device,
                 json_path="notes.json"):
        self.model      = model
        self.output_dir = output_dir
        self.gate_db    = gate_db
        self.tail_ms    = tail_ms
        self.device     = device
        self.json_path  = json_path

        os.makedirs(output_dir, exist_ok=True)

        # State
        self.note_count   = 0
        self._open        = False
        self._frames      = []
        self._tail_chunks = 0
        self._pre_roll    = []

        self._tail_needed = max(1, int(tail_ms / 1000 * SAMPLE_RATE / CHUNK))
        self._min_chunks  = max(1, int(MIN_NOTE_MS / 1000 * SAMPLE_RATE / CHUNK))
        self._pre_chunks  = max(1, int(PRE_ROLL_MS / 1000 * SAMPLE_RATE / CHUNK))

        self._threshold_db = None

        # Stop signal
        self._stop_event = threading.Event()
        # Set once calibration finishes — lets the API block until ready
        self.calibration_done = threading.Event()
        # Background saver so file I/O never stalls audio
        self._q      = queue.Queue()
        self._saver  = threading.Thread(target=self._save_worker, daemon=True)
        self._saver.start()

    def stop(self):
        """Signal the recording loop to exit cleanly."""
        self._stop_event.set()

    # ── background file writer ─────────────────────────────────────────────────

    def _save_worker(self):
        """Runs in a background thread: saves WAV then runs prediction."""
        while True:
            item = self._q.get()
            if item is None:
                break
            frames, path = item
            dur = save_wav(frames, path)
            print(f"  ✓ saved {os.path.basename(path)}  ({dur:.2f}s)")
            self._run_prediction(path)

    def _run_prediction(self, path):
       
        """Predict string/fret and append the result to the session JSON."""
        try:
            string, fret = get_string_fret_from_wav(path, model=self.model)
            print(f"  → string: {string}  fret: {fret}")

            note_data = {"string": string, "fret": fret}

            # Read → append → write (simple, safe enough for single-writer use)
            if os.path.exists(self.json_path):
                with open(self.json_path, "r") as f:
                    notes = json.load(f)
            else:
                notes = []

            notes.append(note_data)

            with open(self.json_path, "w") as f:
                json.dump(notes, f, indent=2)

        except Exception as e:
            print(f"  [prediction error] {e}")

    # ── calibration ───────────────────────────────────────────────────────────

    def _calibrate(self, stream) -> float:
        n      = int(CALIBRATE_SEC * SAMPLE_RATE / CHUNK)
        levels = []
        print(f"  Calibrating ({CALIBRATE_SEC:.0f}s) — keep quiet...", end="", flush=True)
        for _ in range(n):
            raw   = stream.read(CHUNK, exception_on_overflow=False)
            chunk = np.frombuffer(raw, dtype=np.float32).copy()
            levels.append(rms_db(chunk))

        noise_ceil_db = float(np.percentile(levels, 90))
        threshold_db  = noise_ceil_db + self.gate_db

        print(f" done.  Noise ceiling: {noise_ceil_db:.1f} dBFS  |  Gate opens at: {threshold_db:.1f} dBFS")
        return threshold_db

    # ── per-chunk processing ───────────────────────────────────────────────────

    def _process(self, chunk: np.ndarray):
        level = rms_db(chunk)
        loud  = level >= self._threshold_db

        if loud:
            if not self._open:
                self._open        = True
                self._frames      = list(self._pre_roll) + [chunk.copy()]
                self._tail_chunks = 0
                print(f"  ● recording...  ({level:.1f} dBFS)", end="\r")
            else:
                self._frames.append(chunk.copy())
                self._tail_chunks = 0
        else:
            if self._open:
                self._frames.append(chunk.copy())
                self._tail_chunks += 1

                if self._tail_chunks >= self._tail_needed:
                    self._open = False
                    if len(self._frames) >= self._min_chunks:
                        self._queue_save()
                    else:
                        self._frames = []
                    self._tail_chunks = 0

            if not self._open:
                self._pre_roll.append(chunk.copy())
                if len(self._pre_roll) > self._pre_chunks:
                    self._pre_roll.pop(0)

    def _queue_save(self):
        self.note_count += 1
        ts   = datetime.now().strftime("%H%M%S")
        name = f"note_{self.note_count:04d}_{ts}.wav"
        path = os.path.join(self.output_dir, name)
        self._q.put((self._frames[:], path))
        self._frames = []
        print(f"  → queued note {self.note_count:04d}...                    ")

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        try:
            import pyaudio
        except ImportError:
            sys.exit("pyaudio not installed — run:  pip install pyaudio")

        self._stop_event.clear()

        pa = pyaudio.PyAudio()

        if self.device is None:
            try:
                info       = pa.get_default_input_device_info()
                self.device = info["index"]
                print(f"Using default device: [{self.device}] {info['name']}")
            except OSError:
                sys.exit("No default input device.  Use --device N")

        dev = pa.get_device_info_by_index(self.device)

        stream = pa.open(
            format             = pyaudio.paFloat32,
            channels           = CHANNELS,
            rate               = SAMPLE_RATE,
            input              = True,
            input_device_index = self.device,
            frames_per_buffer  = CHUNK,
        )

        self._threshold_db = self._calibrate(stream)
        self.calibration_done.set()

        json_name = os.path.basename(self.json_path)
        

        try:
            while not self._stop_event.is_set():
                raw   = stream.read(CHUNK, exception_on_overflow=False)
                chunk = np.frombuffer(raw, dtype=np.float32).copy()
                self._process(chunk)

        except KeyboardInterrupt:
            print("\nStopping...")

        finally:
            if self._open and len(self._frames) >= self._min_chunks:
                self._queue_save()
            stream.stop_stream()
            stream.close()
            pa.terminate()
            self._q.put(None)
            self._saver.join(timeout=15)
            print(f"\nDone.  {self.note_count} note(s) saved.")


# ── CLI ────────────────────────────────────────────────────────────────────────
