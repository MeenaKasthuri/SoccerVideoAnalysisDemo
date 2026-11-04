"""Replay saved real tracking through annotation only; no YOLO/OCR/event processing.

This CLI is deliberately excluded from unittest discovery. Supply copied local
fixtures explicitly; it never reads Drive or runs inference on its own.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def restored_tracks(records):
    """Restore native NumPy colors lost through JSON, without changing values."""
    tracks = {category: [] for category in ("players", "referees", "ball")}
    for index, frame in enumerate(records):
        if frame.get("frame", index) != index:
            raise ValueError("Replay requires contiguous tracking frames starting at zero")
        for category in ("players", "referees"):
            objects = {}
            for row in frame.get(category, []):
                obj = dict(row)
                track_id = int(obj.pop("track_id"))
                if track_id in objects:
                    raise ValueError("Duplicate saved track ID in one frame")
                if obj.get("team_color") is not None:
                    # Integer unknown [0,0,255] becomes int64, as the baseline
                    # NumPy fallback; floating centroids remain float64.
                    obj["team_color"] = np.asarray(obj["team_color"])
                objects[track_id] = obj
            tracks[category].append(objects)
        ball = dict(frame["ball"]) if frame.get("ball") is not None else None
        if ball is not None:
            ball.pop("track_id", None)
        tracks["ball"].append({1: ball} if ball is not None else {})
    return tracks


def possession_from_saved_tracks(tracks):
    """Use existing exported possession; do not rerun assignment or events."""
    result = []
    for players in tracks["players"]:
        owners = [player for player in players.values() if player.get("has_ball")]
        if len(owners) > 1:
            raise ValueError("Saved frame has multiple possession owners")
        result.append(int(owners[0].get("team", 0)) if owners else (result[-1] if result else 0))
    return np.asarray(result)


def drawing_init(self, model_path=None):
    """Only bypass the model constructor; every drawing method remains real."""
    import supervision as sv
    from camera_motion.camera_motion import CameraMotionEstimator
    class InferenceForbidden:
        def predict(self, *args, **kwargs):
            raise AssertionError("Annotation replay must never call detector inference")
    self.model = InferenceForbidden()
    self.tracker = sv.ByteTrack()
    self.track_history = {}
    self.pretrack_history = {}
    self.last_ball_bbox = None
    self.previous_ball_center = None
    self.smoothed_ball = None
    self.ball_alpha = .7
    self.previous_ids = set()
    self.motion_estimator = CameraMotionEstimator()
    self.ball_history = []


def reproduce_original(Tracker, video, tracks, control, chunk_size, iter_chunks):
    tracker = Tracker.__new__(Tracker)
    drawing_init(tracker)
    progress = {"frame": 0, "chunk_start": 0}
    context = {}
    real_ellipse = tracker.draw_ellipse
    real_control = tracker.draw_team_ball_control
    def ellipse(frame, bbox, color, track_id=None, draw_id=True):
        try:
            return real_ellipse(frame, bbox, color, track_id, draw_id)
        except cv2.error:
            values = np.asarray(color)
            context.update(frame=progress["frame"], track_id=track_id,
                           color=values.tolist(), color_dtype=str(values.dtype),
                           color_element_types=[type(value).__name__ for value in values],
                           bbox=list(map(float, bbox)))
            raise
    def team_control(frame, local_frame, values):
        global_frame = progress["chunk_start"] + local_frame
        result = real_control(frame, global_frame, control)
        progress["frame"] = global_frame + 1
        return result
    tracker.draw_ellipse = ellipse
    tracker.draw_team_ball_control = team_control
    try:
        for start, frames in iter_chunks(video, chunk_size):
            progress.update(frame=start, chunk_start=start)
            subset = {key: value[start:start + len(frames)] for key, value in tracks.items()}
            tracker.draw_annotations(frames, subset, control)
    except cv2.error as exc:
        message = str(exc)
        return {"reproduced": bool(context) and "color" in message.lower()
                and "not numeric" in message.lower(), "exception_type": "cv2.error",
                "error": message, **context}
    return {"reproduced": False, "completed_frames": progress["frame"]}


def replay_fixed(Tracker, make_tracker, video, tracks, control, output, chunk_size, info, iter_chunks, mode,
                 failure_frame):
    with patch.object(Tracker, "__init__", drawing_init):
        tracker = make_tracker("INFERENCE_DISABLED", info["fps"], None if mode == "baseline" else {}, output)
    tracker.full_team_ball_control = control
    destination = output / (mode + "_annotation_replay.avi")
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"XVID"),
                             info["fps"], (info["width"], info["height"]))
    if not writer.isOpened():
        raise RuntimeError("Could not open annotation replay writer")
    written, boundaries, snapshots = 0, [], []
    selected = {0, failure_frame, len(tracks["players"]) - 1}
    selected.update(frame for start in range(chunk_size, len(tracks["players"]), chunk_size)
                    for frame in (start - 1, start))
    try:
        for start, frames in iter_chunks(video, chunk_size):
            boundaries.append(start)
            tracker.frame_offset = start
            subset = {key: value[start:start + len(frames)] for key, value in tracks.items()}
            annotated = tracker.draw_annotations(frames, subset, control)
            if len(annotated) != len(frames):
                raise RuntimeError("Annotation dropped frames")
            for local, frame in enumerate(annotated):
                if frame.shape != (info["height"], info["width"], 3) or frame.dtype != np.uint8:
                    raise RuntimeError("Annotation changed frame format")
                writer.write(frame)
                if start + local in selected:
                    path = output / f"{mode}_frame_{start + local:05d}.png"
                    if not cv2.imwrite(str(path), frame):
                        raise RuntimeError("Could not save replay snapshot")
                    snapshots.append(str(path))
                written += 1
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(destination))
    decoded, bad_shapes = 0, []
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (info["height"], info["width"], 3):
                bad_shapes.append(decoded)
            decoded += 1
    finally:
        capture.release()
    if written != len(tracks["players"]) or decoded != written or bad_shapes or abs(fps - info["fps"]) > .01:
        raise RuntimeError("Annotation AVI readback failed")
    return {"mode": mode, "written_frames": written, "decoded_frames": decoded,
            "chunk_starts": boundaries, "width": info["width"], "height": info["height"],
            "fps": fps, "video": str(destination), "video_sha256": digest(destination),
            "snapshots": snapshots, "readback_ok": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", default=r"D:\SoccerProject\SoccerVideoAnalysisDemo")
    for name in ("video", "tracks", "events", "run-summary", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--expected-frames", type=int, default=120)
    parser.add_argument("--expected-events", type=int, default=44)
    args = parser.parse_args()
    stage = Path(__file__).resolve().parents[1]
    source = Path(args.source_project).resolve()
    sys.path[:0] = [str(stage / "colab_support"), str(source)]
    import colab_helpers as ch
    ch.install_import_stubs()
    from chunked_pipeline import iter_video_chunks
    from pipeline_tracker import make_tracker
    from trackers.tracker import Tracker

    if args.chunk_size < 5 or args.chunk_size % 5:
        raise ValueError("Chunk size must be a positive multiple of five")
    inputs = {name: getattr(args, name) for name in ("video", "tracks", "events", "run_summary")}
    hashes = {name: digest(path) for name, path in inputs.items()}
    records = json.loads(args.tracks.read_text(encoding="utf-8"))
    events = json.loads(args.events.read_text(encoding="utf-8"))
    gpu_run = json.loads(args.run_summary.read_text(encoding="utf-8"))
    if len(records) != args.expected_frames or len(events) != args.expected_events:
        raise ValueError("Copied fixture does not match expected failed-run frame/event counts")
    info = ch.video_info(args.video)
    if info["frames"] != len(records) or ch.count_frames(args.video) != len(records):
        raise ValueError("Saved tracking and real sample have different frame counts")
    if args.output.resolve().is_relative_to(source) or any(args.output.resolve() == path.resolve() for path in inputs.values()):
        raise ValueError("Replay output must be separate from original project and fixture files")
    args.output.mkdir(parents=True, exist_ok=False)
    tracks = restored_tracks(records)
    control = possession_from_saved_tracks(tracks)
    report = {"status": "running", "annotation_replay_only": True,
              "new_yolo_inference_run": False, "new_colab_gpu_run": False,
              "ocr_run": False, "events_recomputed": False, "event_count": len(events),
              "input_files": {key: str(path) for key, path in inputs.items()}, "input_sha256_before": hashes,
              "saved_gpu_run": {key: gpu_run.get(key) for key in ("status", "frames", "events", "inference_device", "error")},
              "opencv_version": cv2.__version__, "numpy_version": np.__version__,
              "source_tracker_sha256": digest(source / "trackers/tracker.py"),
              "staged_adapter_sha256": digest(stage / "colab_support/pipeline_tracker.py"),
              "color_restoration": "np.asarray(saved_color): integer arrays int64, floating arrays float64; values unchanged"}
    report["integer_color_observations"] = [
        {"frame": frame_index, "track_id": track_id, "team": player.get("team"),
         "color": player["team_color"].tolist(), "dtype": str(player["team_color"].dtype)}
        for frame_index, players in enumerate(tracks["players"])
        for track_id, player in players.items()
        if player.get("team_color") is not None and np.issubdtype(player["team_color"].dtype, np.integer)]
    try:
        with (args.output / "annotation_replay.log").open("w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            report["original_failure"] = reproduce_original(Tracker, args.video, tracks, control,
                                                              args.chunk_size, iter_video_chunks)
            if not report["original_failure"]["reproduced"]:
                raise RuntimeError("Original drawing did not reproduce the saved color failure")
            report["fixed_replays"] = [replay_fixed(Tracker, make_tracker, args.video, tracks, control,
                args.output, args.chunk_size, info, iter_video_chunks, mode,
                report["original_failure"]["frame"]) for mode in ("baseline", "improved")]
        report["input_sha256_after"] = {name: digest(path) for name, path in inputs.items()}
        report["inputs_unchanged"] = report["input_sha256_after"] == hashes
        if not report["inputs_unchanged"]:
            raise RuntimeError("Replay changed an input fixture")
        report["status"] = "annotation_replay_passed"
    except Exception as exc:
        report.update(status="validation_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        (args.output / "annotation_replay_receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "annotation_replay_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
