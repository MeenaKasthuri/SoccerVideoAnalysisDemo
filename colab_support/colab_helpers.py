"""Colab support helpers for the Soccer Video Analysis pipeline.

These helpers wrap the existing project without modifying it:
- stand-in modules for imports that main.py makes but never uses on Colab
  (streamlit, pyodbc / SQL Server),
- GPU verification of the real Tracker before detection starts,
- short sample creation that streams frames instead of loading a video,
- log filtering, RAM monitoring and output validation for pipeline runs.
"""
import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import types
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from drawing_colors import ColorSafeDrawingMixin

# Options for the initial Colab runs: no database, no Streamlit, and none of
# the pitch / zone / hull / heatmap videos that depend on the fixed
# homography, which is not calibrated for moving-camera footage.
SAFE_OPTIONS = {
    "pitch": False,
    "heatmap": False,
    "zones": False,
    "hulls": False,
    "overlay_heatmap": False,
    "json": True,
    "csv": True,
    "database": False,
    "spatial_reliable": False,
}

# Relative directories main.py writes into. stubs/ must exist because the
# tracker pickles its output there; assets/ and output_heatmaps/ receive
# cv2.imwrite calls that fail silently when the directory is missing.
PROJECT_OUTPUT_DIRS = [
    "stubs",
    "JSON_data",
    "CSV_data",
    "output_videos",
    "assets",
    "output_heatmaps",
    "sports_db",
]

REQUIRED_CLASSES = {"player", "goalkeeper", "referee", "ball"}

# Detection settings used by Tracker.detect_frames
DETECT_KWARGS = {"conf": 0.25, "iou": 0.35, "imgsz": 1280, "verbose": False}

GB = 1024 ** 3


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

def install_import_stubs():
    """Register stand-ins for streamlit and pyodbc.

    main.py imports streamlit (unused) and database.sql_server imports pyodbc
    at module level. Both are only needed when options["database"] is True
    or the Streamlit UI runs, so the stand-ins raise a clear error if any
    code actually tries to use them.
    """
    def _disabled(name):
        def __getattr__(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            raise RuntimeError(
                f"{name}.{attr} was called, but {name} is disabled in the "
                f"Colab run. Keep options['database'] = False and do not "
                f"use the Streamlit UI in this notebook."
            )
        return __getattr__

    installed = []
    for name in ("streamlit", "pyodbc"):
        if name in sys.modules and not getattr(sys.modules[name], "__colab_stub__", False):
            continue
        module = types.ModuleType(name)
        module.__colab_stub__ = True
        module.__getattr__ = _disabled(name)
        sys.modules[name] = module
        installed.append(name)
    return installed


def prepare_project(project_root):
    """Make project_root importable and the working directory, create output dirs."""
    project_root = str(Path(project_root).resolve())
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    for folder in PROJECT_OUTPUT_DIRS:
        os.makedirs(folder, exist_ok=True)
    return project_root


def system_ram():
    import psutil
    vm = psutil.virtual_memory()
    return {"total_gb": vm.total / GB, "available_gb": vm.available / GB}


# ----------------------------------------------------------------------
# Video helpers
# ----------------------------------------------------------------------

def video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {video_path}")
    info = {
        "path": str(video_path),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


def estimate_pipeline_ram_gb(frames, width, height):
    """Rough peak RAM for main.main() on a video.

    main.py keeps every decoded frame (read_video) and a full copy of every
    annotated frame (draw_annotations) in memory at the same time, plus about
    1.5 GB for Python, PyTorch and the CUDA context. Calibrate against the
    measured peak from the sample runs.
    """
    frame_bytes = width * height * 3
    raw_gb = frames * frame_bytes / GB
    return {"raw_frames_gb": raw_gb, "estimated_peak_gb": 2 * raw_gb + 1.5}


def read_frame(video_path, frame_index=0):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame


def count_frames(video_path):
    """Count frames by decoding (use only on short clips)."""
    cap = cv2.VideoCapture(str(video_path))
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


def safe_stem(path):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(path).stem).strip("_")


def make_sample(source, output_dir, start_s, duration_s, use_ffmpeg=True):
    """Write a short clip without loading the source video into memory.

    Uses ffmpeg (high-quality H.264, exact frame count) when available and
    falls back to OpenCV streaming read/write.
    """
    info = video_info(source)
    fps = info["fps"]
    start_frame = int(round(start_s * fps))
    n_frames = int(round(duration_s * fps))
    if start_frame + n_frames > info["frames"]:
        raise ValueError(
            f"Sample {start_s}s + {duration_s}s exceeds video length "
            f"({info['duration_s']:.1f}s)"
        )

    os.makedirs(output_dir, exist_ok=True)
    duration_tag = f"{duration_s:.3g}".replace(".", "p")
    output = Path(output_dir) / f"{safe_stem(source)}_s{int(start_s):04d}_{duration_tag}s.mp4"

    method = None
    if use_ffmpeg and shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start_frame / fps:.6f}", "-i", str(source),
            "-frames:v", str(n_frames), "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-pix_fmt", "yuv420p", str(output),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            method = "ffmpeg"
        else:
            print("ffmpeg failed, falling back to OpenCV:\n", proc.stderr[-2000:])

    if method is None:
        cap = cv2.VideoCapture(str(source))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (info["width"], info["height"])
        )
        written = 0
        while written < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
        cap.release()
        writer.release()
        method = "opencv"

    sample = video_info(output)
    sample["decoded_frames"] = count_frames(output)
    sample["method"] = method
    sample["source_start_frame"] = start_frame
    sample["requested_frames"] = n_frames
    if sample["decoded_frames"] != n_frames:
        print(
            f"WARNING: sample has {sample['decoded_frames']} decodable frames, "
            f"expected {n_frames}"
        )
    return sample


# ----------------------------------------------------------------------
# GPU verification
# ----------------------------------------------------------------------

def inference_device(yolo_model):
    """Device the Ultralytics predictor actually ran on (after one predict)."""
    predictor = getattr(yolo_model, "predictor", None)
    if predictor is None:
        raise RuntimeError("Model has not run predict() yet")
    device = getattr(predictor, "device", None)
    # AutoBackend wraps the torch module; check where the weights really live
    try:
        param_device = next(predictor.model.model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        param_device = getattr(predictor.model, "device", None)
    return device, param_device


def check_model_classes(yolo_model):
    names = set(yolo_model.names.values())
    missing = REQUIRED_CLASSES - names
    if missing:
        raise RuntimeError(
            f"Model is missing classes required by Tracker: {sorted(missing)}. "
            f"Model classes: {yolo_model.names}"
        )
    return yolo_model.names


def verify_tracker_device(tracker, require_gpu=True, log=print):
    """Run one dummy predict through the Tracker's model and check its device."""
    if require_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Select a Colab GPU runtime before requesting inference")
    dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    tracker.model.predict(dummy, **DETECT_KWARGS)
    device, param_device = inference_device(tracker.model)
    log(f"[GPU CHECK] Tracker YOLO predictor device={device} weights device={param_device}")
    on_gpu = getattr(device, "type", str(device)) == "cuda"
    if require_gpu and not on_gpu:
        raise RuntimeError(
            "Tracker YOLO is running on CPU. In Colab use Runtime > Change "
            "runtime type > GPU, then Runtime > Restart session and rerun. "
            "Set REQUIRE_GPU = False only to test deliberately on CPU."
        )
    return device


def make_verified_tracker_class(base_cls, require_gpu=True, log=print):
    """Subclass of the project's Tracker that verifies the device on creation."""
    drawing_base = base_cls
    if not issubclass(base_cls, ColorSafeDrawingMixin):
        class DrawingSafeTracker(ColorSafeDrawingMixin, base_cls):
            pass
        drawing_base = DrawingSafeTracker

    class GpuVerifiedTracker(drawing_base):
        def __init__(self, model_path):
            super().__init__(model_path)
            self.inference_device = verify_tracker_device(self, require_gpu, log)

    GpuVerifiedTracker.__name__ = f"GpuVerified{base_cls.__name__}"
    return GpuVerifiedTracker


# ----------------------------------------------------------------------
# Run monitoring
# ----------------------------------------------------------------------

class FilteredTee(io.TextIOBase):
    """Write everything to a log file, echo only progress lines to the console.

    The tracker prints ~15 debug lines per frame; showing all of it in Colab
    makes the browser slow on long runs.
    """
    KEY_PREFIXES = (
        "STEP", "[TIME]", "[DETAIL]", "[GPU CHECK]", "ERROR", "Traceback",
        "CSV WRITE", "JSON WRITE", "EVENT JSON", "Not enough", "No ball",
        "Loaded model", "Total events", "Collected colors", "Input frames",
        "Output frames", "FPS:", ">>> ",
    )
    FRAME_RE = re.compile(r"^FRAME: (\d+)$")

    def __init__(self, log_file, console, total_frames=None, every_n_frames=25,
                 echo_all=False):
        self.log_file = log_file
        self.console = console
        self.total_frames = total_frames
        self.every_n_frames = every_n_frames
        self.echo_all = echo_all
        self._pending = ""
        self._start = time.perf_counter()

    def writable(self):
        return True

    def write(self, text):
        self.log_file.write(text)
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        for line in lines:
            self._echo(line)
        return len(text)

    def flush(self):
        if not self.log_file.closed:
            self.log_file.flush()
        self.console.flush()

    def _echo(self, line):
        if self.echo_all:
            self.console.write(line + "\n")
            return
        match = self.FRAME_RE.match(line.strip())
        if match:
            frame = int(match.group(1))
            if frame % self.every_n_frames == 0:
                total = f"/{self.total_frames}" if self.total_frames else ""
                elapsed = time.perf_counter() - self._start
                self.console.write(f"  ... tracking frame {frame}{total} ({elapsed:.0f}s elapsed)\n")
        elif line.startswith(self.KEY_PREFIXES):
            self.console.write(line[:300] + "\n")


class ResourceMonitor:
    """Sample process RSS and available system RAM in a background thread."""

    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self.peak_rss_gb = 0.0
        self.min_available_gb = float("inf")
        self._stop = threading.Event()

    def __enter__(self):
        import psutil
        self._process = psutil.Process()
        self._psutil = psutil
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def _sample(self):
        self.peak_rss_gb = max(self.peak_rss_gb, self._process.memory_info().rss / GB)
        self.min_available_gb = min(
            self.min_available_gb, self._psutil.virtual_memory().available / GB
        )

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()
        self.peak_gpu_gb = None
        try:
            import torch
            if torch.cuda.is_available():
                self.peak_gpu_gb = torch.cuda.max_memory_allocated() / GB
        except ImportError:
            pass
        return False


# ----------------------------------------------------------------------
# Output validation
# ----------------------------------------------------------------------

def expected_outputs(video_path, run_label):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return {
        "annotated_video": f"output_videos/{run_label}/{video_name}_annotated.avi",
        "events_json": f"JSON_data/{video_name}_events.json",
        "tracking_json": f"JSON_data/{video_name}_tracking.json",
        "tracking_csv": f"CSV_data/{video_name}_tracking.csv",
    }


def validate_outputs(outputs, expected_frames):
    """Check each output file; return (all_ok, report dict)."""
    report = {}
    expected_rows = None

    def fail(key, message):
        report[key] = {"ok": False, "error": message}

    for key, path in outputs.items():
        if not os.path.exists(path):
            fail(key, f"missing: {path}")
        elif os.path.getsize(path) == 0:
            fail(key, f"empty file: {path}")

    if "annotated_video" not in report:
        path = outputs["annotated_video"]
        cap = cv2.VideoCapture(path)
        opened = cap.isOpened()
        ok_first, _ = cap.read() if opened else (False, None)
        cap.release()
        frames = count_frames(path) if opened else 0
        ok = opened and ok_first and frames == expected_frames
        report["annotated_video"] = {
            "ok": ok, "frames": frames, "expected_frames": expected_frames,
            "size_mb": round(os.path.getsize(path) / 1024 ** 2, 2),
        }
        if not ok:
            report["annotated_video"]["error"] = "unreadable or frame count mismatch"

    if "tracking_json" not in report:
        with open(outputs["tracking_json"], encoding="utf-8") as f:
            frames = json.load(f)
        players = [len(frame["players"]) for frame in frames]
        ball_frames = sum(1 for frame in frames if frame["ball"] is not None)
        expected_rows = sum(len(frame["players"]) + len(frame["referees"])
                            + int(frame["ball"] is not None) for frame in frames)
        ok = len(frames) == expected_frames
        report["tracking_json"] = {
            "ok": ok, "frames": len(frames), "expected_frames": expected_frames,
            "avg_players_per_frame": round(float(np.mean(players)), 2) if players else 0,
            "unique_player_ids": len({p["track_id"] for f in frames for p in f["players"]}),
            "frames_with_ball": ball_frames,
        }
        if not ok:
            report["tracking_json"]["error"] = "frame count mismatch (tracking stopped early?)"

    if "events_json" not in report:
        with open(outputs["events_json"], encoding="utf-8") as f:
            events = json.load(f)
        ok = isinstance(events, list)
        report["events_json"] = {
            "ok": ok, "events": len(events) if ok else None,
            "by_type": dict(Counter(e.get("event_type") for e in events)) if ok else None,
        }

    if "tracking_csv" not in report:
        with open(outputs["tracking_csv"], newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames or []
        needed = {"frame", "object_type", "track_id", "bbox_x1", "team", "has_ball"}
        ok = needed.issubset(header) and (expected_rows is None or len(rows) == expected_rows)
        report["tracking_csv"] = {
            "ok": ok, "rows": len(rows), "expected_rows": expected_rows,
            "by_object_type": dict(Counter(r["object_type"] for r in rows)),
        }
        if not ok:
            report["tracking_csv"]["error"] = "Header or JSON/CSV object count mismatch"

    return all(item["ok"] for item in report.values()), report


# ----------------------------------------------------------------------
# Pipeline run
# ----------------------------------------------------------------------

def run_pipeline(project_root, video_path, run_label, drive_output_dir,
                 require_gpu=True, options=None, expected_frames=None):
    """Run main.main() on one video with Colab-safe options and validate outputs.

    Validated outputs, the full log and a run_summary.json are copied to
    drive_output_dir/run_label. Returns the summary dict.
    """
    options = dict(SAFE_OPTIONS if options is None else options)
    console = sys.stdout
    prepare_project(project_root)
    install_import_stubs()

    import main as pipeline
    from trackers.tracker import Tracker as ProjectTracker

    pipeline.Tracker = make_verified_tracker_class(
        ProjectTracker, require_gpu=require_gpu, log=print
    )

    if expected_frames is None:
        expected_frames = count_frames(video_path)

    outputs = expected_outputs(video_path, run_label)
    for path in outputs.values():
        if os.path.exists(path):
            os.remove(path)  # never validate a stale file from an earlier run

    run_dir = Path(drive_output_dir) / run_label
    local_log = Path("output_videos") / run_label / "pipeline_log.txt"
    local_log.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_label": run_label,
        "video": str(video_path),
        "expected_frames": expected_frames,
        "options": options,
        "status": "failed",
    }
    console.write(f"Running pipeline on {video_path} ({expected_frames} frames)\n")
    console.write(f"Full log: {local_log}\n")

    start = time.perf_counter()
    with open(local_log, "w", encoding="utf-8") as log_file, \
            ResourceMonitor() as monitor:
        tee = FilteredTee(log_file, console, total_frames=expected_frames)
        err_tee = FilteredTee(log_file, sys.stderr, echo_all=True)
        try:
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(err_tee):
                results = pipeline.main(
                    str(video_path),
                    output_folder=f"output_videos/{run_label}",
                    options=options,
                )
            summary["main_returned"] = results
            if results is None:
                summary["error"] = (
                    "main() returned early (usually 'Not enough data for team "
                    "clustering'): sample has too few clear player crops."
                )
        except Exception as exc:
            tb = traceback.format_exc()
            log_file.write(tb)
            console.write("\nPIPELINE ERROR:\n" + tb)
            summary["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                summary["diagnosis"] = "Out of memory: use a shorter sample."
            elif isinstance(exc, KeyError) and exc.args == (1,):
                summary["diagnosis"] = (
                    "main.py reads track ID 1 in frame 0 (debug prints). The "
                    "sample's first frame probably has no detected player: "
                    "choose a different SAMPLE_START_S."
                )
            elif isinstance(exc, IndexError):
                summary["diagnosis"] = (
                    "main.py indexes tracks['ball'][0..2]; samples need at "
                    "least 3 frames and a tracking result for every frame."
                )
            if "diagnosis" in summary:
                console.write(f"DIAGNOSIS: {summary['diagnosis']}\n")
    summary["elapsed_s"] = round(time.perf_counter() - start, 1)
    summary["peak_rss_gb"] = round(monitor.peak_rss_gb, 2)
    summary["min_available_ram_gb"] = round(monitor.min_available_gb, 2)
    summary["peak_gpu_allocated_gb"] = (
        round(monitor.peak_gpu_gb, 2) if monitor.peak_gpu_gb is not None else None
    )

    if "error" not in summary:
        all_ok, report = validate_outputs(outputs, expected_frames)
        summary["validation"] = report
        summary["status"] = "ok" if all_ok else "validation_failed"

    run_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for key, path in outputs.items():
        if summary.get("validation", {}).get(key, {}).get("ok"):
            shutil.copy2(path, run_dir / os.path.basename(path))
            copied.append(os.path.basename(path))
    shutil.copy2(local_log, run_dir / local_log.name)
    summary["copied_to_drive"] = copied
    summary["drive_dir"] = str(run_dir)
    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    console.write(f"\nStatus: {summary['status']}  ({summary['elapsed_s']}s, "
                  f"peak RAM {summary['peak_rss_gb']} GB, "
                  f"peak GPU {summary['peak_gpu_allocated_gb']} GB)\n")
    if "error" in summary:
        console.write(f"Error: {summary['error']}\n")
    for key, item in summary.get("validation", {}).items():
        console.write(f"  {'OK  ' if item['ok'] else 'FAIL'} {key}: "
                      f"{ {k: v for k, v in item.items() if k != 'ok'} }\n")
    console.write(f"Saved to Drive: {run_dir}\n")
    return summary
