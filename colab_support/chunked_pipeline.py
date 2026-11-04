"""Memory-bounded Colab pipeline with optional, isolated player improvements.

Both modes reuse the original YOLO settings, drawing, ball processing and events.
A single ByteTrack and team/evidence state survives all chunks. Baseline keeps
original classification/role behavior with empty-frame and warmup safety fixes.
Improved mode adds field/duplicate prefilters, goalkeeper output, FPS-aware
tracking, team consensus and optional number evidence. No IDs are merged.

Decoded images are bounded by chunk_size; lightweight tracks remain in memory
for the original full-track analytics pass. Coordinates are explicitly invalid
because the historical homography is not calibrated for these moving cameras.
"""
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

import colab_helpers as ch

MODEL_PATH = "models/best (2).pt"

# Same homography calibration points as main.py
IMAGE_POINTS = [[174, 914], [1752, 901], [1456, 182], [432, 205]]
FIELD_POINTS = [[0, 68], [105, 68], [105, 0], [0, 0]]
MIN_TEAM_COLORS = 20  # collect_player_colors(min_colors=20) in main.py


def iter_video_chunks(video_path, chunk_size):
    """Yield (start_frame, [frames]) without holding more than one chunk."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {video_path}")
    start = 0
    try:
        while True:
            frames = []
            while len(frames) < chunk_size:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            if not frames:
                return
            yield start, frames
            start += len(frames)
            if len(frames) < chunk_size:
                return
    finally:
        cap.release()


def _crop(frame, bbox):
    """Clip crop coordinates before reusing the original team classifier."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()


def _crop_bbox(crop):
    return (0, 0, crop.shape[1], crop.shape[0])


class StreamingTeamAssigner:
    """Reproduces main.py's team assignment order without keeping frames.

    main.py collects player colours from frame 0 onward until >= 20, fits the
    team KMeans, then assigns every player of every frame from frame 0. Until
    the fit, the crops of frames already seen are buffered (bounded: every
    buffered frame adds at least one colour, so at most ~20 frames). The
    KMeans calls happen in the same order as in main.py.
    """

    def __init__(self, all_players):
        from team_assigner import TeamAssigner
        self.assigner = TeamAssigner()
        self.all_players = all_players  # tracks["players"], grows chunk by chunk
        self.colors = []
        self.fitted = False
        self.pending = {}               # global frame -> {player_id: crop}
        self.next_frame = 0             # next global frame to assign

    def process_chunk(self, start, frames):
        if not self.fitted:
            for i, frame in enumerate(frames):
                players = self.all_players[start + i]
                if not players:
                    continue
                self.pending[start + i] = {
                    pid: _crop(frame, p["bbox"]) for pid, p in players.items()
                }
                for crop in self.pending[start + i].values():
                    color = self._color(crop)
                    if color is not None:
                        self.colors.append(color)
                if len(self.colors) >= MIN_TEAM_COLORS or len(self.pending) >= 60:
                    self._fit()
                    break
        if self.fitted:
            self._assign_through(start + len(frames), start, frames)

    def finish(self):
        """Fit with whatever was collected if the video ended first."""
        if not self.fitted:
            self._fit()
        n = len(self.all_players)
        self._assign_through(n, n, [])

    def _fit(self):
        self.assigner.assign_team_color_from_colors(self.colors)
        self.fitted = True

    def _color(self, crop):
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        return self.assigner.get_player_color(crop, _crop_bbox(crop))

    def _assign_through(self, end, start, frames):
        while self.next_frame < end:
            g = self.next_frame
            crops = self.pending.pop(g, None)
            frame = frames[g - start] if g >= start else None
            for pid, player in self.all_players[g].items():
                if crops is not None:
                    crop = crops[pid]
                elif frame is not None:
                    crop = _crop(frame, player["bbox"])
                else:
                    raise RuntimeError(f"No frame or buffered crop for frame {g}")
                team = (self.assigner.get_player_team(crop, _crop_bbox(crop), pid)
                        if crop.shape[0] >= 8 and crop.shape[1] >= 8 else 0)
                player["team"] = team                          # mirrors main.py
                # Display-only unknown fallback: native BGR ints are required by
                # OpenCV 5; tuple(ndarray.astype(int)) still contains NumPy ints.
                player["team_color"] = self.assigner.team_colors.get(team, (0, 0, 255))
            self.next_frame += 1


def export_tracking_json_streaming(tracks, output_path):
    """Compatibility alias for the extended evidence-aware streaming export."""
    from tracking_exports import export_tracking_json
    return export_tracking_json(tracks, output_path)


class Progress:
    def __init__(self, console, total_frames, min_available_gb):
        import psutil
        self.console = console
        self.total = total_frames
        self.min_available_gb = min_available_gb
        self.process = psutil.Process()
        self.psutil = psutil
        self.peak_rss_gb = 0.0

    def memory(self):
        rss = self.process.memory_info().rss / ch.GB
        available = self.psutil.virtual_memory().available / ch.GB
        self.peak_rss_gb = max(self.peak_rss_gb, rss)
        gpu = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu = torch.cuda.memory_allocated() / ch.GB
        except ImportError:
            pass
        return rss, available, gpu

    def check_memory(self, where):
        rss, available, gpu = self.memory()
        if available < self.min_available_gb:
            raise MemoryError(
                f"Only {available:.1f} GB RAM available at {where} "
                f"(limit {self.min_available_gb} GB). Reduce CHUNK_SIZE."
            )
        return rss, available, gpu

    def report(self, label, done, started):
        rss, available, gpu = self.check_memory(label)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed > 0 else 0
        eta = (self.total - done) / rate if rate > 0 and self.total else 0
        gpu_text = f" | GPU {gpu:.2f} GB" if gpu is not None else ""
        self.console.write(
            f"[{label}] {done}/{self.total} frames | {rate:.1f} fps | "
            f"elapsed {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min | "
            f"RAM {rss:.2f} GB used, {available:.1f} GB free{gpu_text}\n"
        )
        self.console.flush()

    def say(self, text):
        self.console.write(text + "\n")
        self.console.flush()


def run_chunked(video_path, output_dir, chunk_size=60, require_gpu=True,
                min_available_gb=2.0, seed=None, model_path=MODEL_PATH,
                console=None, improvements=None):
    """Run the chunked pipeline. Returns a summary dict (also saved as JSON).

    output_dir receives <video>_annotated.avi, <video>_events.json,
    <video>_tracking.json, <video>_tracking.csv, pipeline_log.txt and
    run_summary.json. chunk_size must be a multiple of 5 so the every-5-frames
    camera-motion cadence in draw_annotations stays aligned with main.py.
    """
    if chunk_size < 5 or chunk_size % 5:
        raise ValueError("chunk_size must be a positive multiple of 5")
    console = console or sys.stdout
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory; preserving existing results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(str(video_path)))[0]
    paths = {
        "annotated_video": output_dir / f"{video_name}_annotated.avi",
        "events_json": output_dir / f"{video_name}_events.json",
        "tracking_json": output_dir / f"{video_name}_tracking.json",
        "tracking_csv": output_dir / f"{video_name}_tracking.csv",
    }
    log_path = output_dir / "pipeline_log.txt"

    info = ch.video_info(video_path)
    progress = Progress(console, info["frames"], min_available_gb)
    summary = {
        "video": str(video_path), "video_info": info, "chunk_size": chunk_size,
        "seed": seed, "status": "failed", "outputs": {k: str(v) for k, v in paths.items()},
        "mode": "baseline" if improvements is None else "improved",
        "improvements": improvements,
        "baseline_safety_fixes": ["empty frames do not truncate processing", "robust export and warmup"],
    }
    progress.say(f"Video: {info['width']}x{info['height']} @ {info['fps']:.3f} fps, "
                 f"~{info['frames']} frames ({info['duration_s']:.1f} s), chunk_size={chunk_size}")
    progress.say(f"Writing to {output_dir} (full debug log: {log_path.name})")

    total_start = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log_file, \
            contextlib.redirect_stdout(log_file), \
            contextlib.redirect_stderr(ch.FilteredTee(log_file, sys.stderr, echo_all=True)):
        try:
            _run(video_path, paths, chunk_size, require_gpu, seed, model_path,
                 progress, summary, improvements)
            valid, checks = ch.validate_outputs(paths, summary["frames"])
            summary["validation"] = checks
            summary["status"] = "ok" if valid else "validation_failed"
        except Exception as exc:
            log_file.write(traceback.format_exc())
            console.write("\nCHUNKED PIPELINE ERROR:\n" + traceback.format_exc())
            summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["elapsed_s"] = round(time.perf_counter() - total_start, 1)
    summary["peak_rss_gb"] = round(progress.peak_rss_gb, 2)
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    progress.say(f"Status: {summary['status']} in {summary['elapsed_s'] / 60:.1f} min, "
                 f"peak RAM {summary['peak_rss_gb']} GB")
    return summary


def _run(video_path, paths, chunk_size, require_gpu, seed, model_path, progress, summary, improvements=None):
    from homography.homography import HomographyGenerator
    from player_ball_assigner import PlayerBallAssigner
    from analytics.event_model import EventModel
    from tracking_exports import export_tracking_csv, export_tracking_json
    from exporters.event_exporter import export_events_json
    from pipeline_tracker import make_tracker
    from identity_evidence import StableTeamAssigner, JerseyNumberRecognizer

    if seed is not None:
        np.random.seed(seed)  # KMeans in TeamAssigner uses the global RNG

    fps = summary["video_info"]["fps"]
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    ownership_config = (improvements or {}).get("ball_ownership")
    player_assigner = PlayerBallAssigner.from_configuration(
        ownership_config,
        image_size=(summary["video_info"]["width"], summary["video_info"]["height"]))
    summary["ball_ownership_configuration"] = player_assigner.configuration
    if require_gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required; select a Colab GPU runtime before inference")
    tracker = make_tracker(model_path, fps, improvements, paths["tracking_json"].parent)
    summary["inference_device"] = str(ch.verify_tracker_device(tracker, require_gpu, progress.say))

    homography = HomographyGenerator()
    homography.set_points(IMAGE_POINTS, FIELD_POINTS)
    homography.compute_homography()

    tracks = {"players": [], "referees": [], "ball": []}
    teams = (StreamingTeamAssigner(tracks["players"]) if improvements is None
             else StableTeamAssigner(tracks["players"]))
    number_reader = JerseyNumberRecognizer(
        enabled=bool((improvements or {}).get("enable_ocr", False)), gpu=require_gpu)

    # ---------------- Pass 1: detection, tracking, teams ----------------
    progress.check_memory("start")
    started = time.perf_counter()
    chunk_starts = []
    for start, frames in iter_video_chunks(video_path, chunk_size):
        chunk_starts.append(start)
        chunk_tracks = tracker.get_object_tracker(frames, read_from_stub=False, stub_path=None)
        if any(len(chunk_tracks[k]) != len(frames) for k in ("players", "referees", "ball")):
            raise RuntimeError(
                f"Tracker stopped early in chunk starting at frame {start} "
                f"(see 'ERROR AT FRAME' in pipeline_log.txt)"
            )
        homography.transform_tracks(chunk_tracks)
        for category in ["players", "ball", "referees"]:      # mirrors main.py, spatial_reliable=False
            for frame_objects in chunk_tracks[category]:
                for obj in frame_objects.values():
                    obj["coordinate_valid"] = False
        for key in ("players", "referees", "ball"):
            tracks[key].extend(chunk_tracks[key])
        teams.process_chunk(start, frames)
        for local, frame in enumerate(frames):
            number_reader.process(frame, tracks["players"][start + local], start + local)
        del frames, chunk_tracks
        progress.report("pass 1/2 detect+track", len(tracks["players"]), started)
    teams.finish()
    n_frames = len(tracks["players"])
    progress.total = n_frames
    summary["frames"] = n_frames
    summary["chunk_starts"] = chunk_starts
    summary["team_colors"] = {k: v.tolist() for k, v in teams.assigner.team_colors.items()}
    summary["detection_counts"] = tracker.counts
    summary["tracking_configuration"] = tracker.tracking_config
    summary["role_transitions"] = str(tracker.role_transition_log)
    summary["ocr"] = number_reader.summary
    if number_reader.backend_error:
        summary.setdefault("warnings", []).append("OCR backend failed; inspect jersey evidence errors")
    summary["detection_log"] = str(tracker.detection_log)
    summary["field_regions"] = str(tracker.region_log)
    summary["tracking_notes"] = "Raw ByteTrack IDs; no cross-ID merging. Counts do not measure identity accuracy."
    if n_frames < 1:
        raise RuntimeError("No frames decoded")

    # ---------------- Analytics on tracks only ----------------
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    for frame_ball in tracks["ball"]:                          # mirrors main.py
        if 1 not in frame_ball:
            continue
        ball = frame_ball[1]
        bbox = ball["bbox"]
        image_x = (bbox[0] + bbox[2]) / 2
        image_y = (bbox[1] + bbox[3]) / 2
        pitch_x, pitch_y = homography.transform_homography(image_x, image_y)
        ball["pitch_x"] = pitch_x
        ball["pitch_y"] = pitch_y
        ball["coordinate_valid"] = False

    team_ball_control = []
    tracks["divided_ball"] = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox', None)
        if ball_bbox is None:
            tracks["divided_ball"].append(None)
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue
        divided_ball = player_assigner.detect_divided_ball(player_track, ball_bbox)
        if divided_ball is None:
            tracks["divided_ball"].append(None)
        else:
            ball = tracks['ball'][frame_num][1]
            tracks["divided_ball"].append({
                "player_ids": divided_ball["player_ids"],
                "distances": divided_ball["distances"],
                "teams": [player_track[player_id].get("team")
                          for player_id in divided_ball["player_ids"]],
                "pitch_x": ball.get("pitch_x"),
                "pitch_y": ball.get("pitch_y"),
                "coordinate_valid": ball.get("coordinate_valid", True)
            })
        ownership = player_assigner.ownership_observation(player_track, ball_bbox)
        if ownership_config is not None:
            tracks['ball'][frame_num][1]['ownership_observation'] = ownership
        assigned_player = ownership["player_id"] if ownership["player_id"] is not None else -1
        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(player_track[assigned_player]['team'])
        elif len(team_ball_control) > 0:
            team_ball_control.append(team_ball_control[-1])
        else:
            team_ball_control.append(0)
    team_ball_control = np.array(team_ball_control)

    events = EventModel(fps).build_events(tracks)
    export_events_json(events, str(paths["events_json"]))
    export_tracking_json(tracks, paths["tracking_json"])
    export_tracking_csv(tracks, str(paths["tracking_csv"]))
    summary["events"] = len(events)
    progress.say(f"Analytics done: {len(events)} events; tracking JSON/CSV written")

    # ---------------- Pass 2: annotation, streamed to the AVI ----------------
    tracker.full_team_ball_control = team_ball_control
    writer = None
    written = 0
    started = time.perf_counter()
    try:
        for start, frames in iter_video_chunks(video_path, chunk_size):
            if start + len(frames) > n_frames:
                raise RuntimeError("Second decoding pass returned more frames than the first")
            chunk_tracks = {k: tracks[k][start:start + len(frames)]
                            for k in ("players", "referees", "ball")}
            tracker.frame_offset = start
            annotated = tracker.draw_annotations(frames, chunk_tracks, team_ball_control)
            if writer is None:                                  # same codec as save_video
                height, width = annotated[0].shape[:2]
                writer = cv2.VideoWriter(str(paths["annotated_video"]),
                                         cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("Could not open annotated video writer")
            for frame in annotated:
                writer.write(frame)
            written += len(annotated)
            del frames, annotated, chunk_tracks
            progress.report("pass 2/2 annotate", written, started)
    finally:
        if writer is not None:
            writer.release()
    if written != n_frames:
        raise RuntimeError(f"Annotated {written} frames, tracked {n_frames}")
    summary["annotated_frames"] = written


# ----------------------------------------------------------------------
# Comparison against the original main.py outputs
# ----------------------------------------------------------------------

def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare_runs(reference, chunked, boundaries, window=3):
    """Compare main.py outputs with chunked outputs (dicts of output paths).

    Returns a report: byte equality per file, per-frame ID/bbox/team
    differences, video pixel differences, and the IDs and events around
    every chunk boundary.
    """
    report = {}
    for key in ("events_json", "tracking_json", "tracking_csv", "annotated_video"):
        with open(reference[key], "rb") as a, open(chunked[key], "rb") as b:
            report[f"{key}_identical_bytes"] = a.read() == b.read()

    ref_frames = _load_json(reference["tracking_json"])
    chk_frames = _load_json(chunked["tracking_json"])
    report["frames"] = (len(ref_frames), len(chk_frames))
    id_mismatch, bbox_diff, team_mismatch = [], 0.0, 0
    for r, c in zip(ref_frames, chk_frames):
        r_players = {p["track_id"]: p for p in r["players"]}
        c_players = {p["track_id"]: p for p in c["players"]}
        if set(r_players) != set(c_players):
            id_mismatch.append(r["frame"])
            continue
        for pid, rp in r_players.items():
            bbox_diff = max(bbox_diff, float(np.max(np.abs(np.subtract(rp["bbox"], c_players[pid]["bbox"])))))
            team_mismatch += rp["team"] != c_players[pid]["team"]
    report["frames_with_different_player_ids"] = id_mismatch
    report["max_bbox_diff_px"] = bbox_diff
    report["player_team_mismatches"] = team_mismatch

    ref_events = _load_json(reference["events_json"])
    chk_events = _load_json(chunked["events_json"])
    report["events"] = (len(ref_events), len(chk_events))
    report["events_identical"] = ref_events == chk_events

    cap_r, cap_c = cv2.VideoCapture(reference["annotated_video"]), cv2.VideoCapture(chunked["annotated_video"])
    max_pixel_diff, differing_frames, n = 0, 0, 0
    while True:
        ok_r, fr = cap_r.read()
        ok_c, fc = cap_c.read()
        if not ok_r or not ok_c:
            report["video_frame_counts_match"] = ok_r == ok_c
            break
        diff = int(np.max(cv2.absdiff(fr, fc)))
        max_pixel_diff = max(max_pixel_diff, diff)
        differing_frames += diff > 0
        n += 1
    cap_r.release()
    cap_c.release()
    report["video_frames_compared"] = n
    report["video_frames_differing"] = differing_frames
    report["video_max_pixel_diff"] = max_pixel_diff

    boundary_report = []
    for b in boundaries:
        if b <= 0 or b >= len(chk_frames):
            continue
        before = {p["track_id"] for p in chk_frames[b - 1]["players"]}
        after = {p["track_id"] for p in chk_frames[b]["players"]}
        boundary_report.append({
            "boundary_frame": b,
            "ids_before": sorted(before),
            "ids_after": sorted(after),
            "ids_continuing": len(before & after),
            "new_ids_after": sorted(after - before),
            "ids_match_reference": (
                {p["track_id"] for p in ref_frames[b - 1]["players"]} == before
                and {p["track_id"] for p in ref_frames[b]["players"]} == after
            ),
            "events_near_boundary": [
                (e["frame"], e["event_type"], e.get("player_id"), e.get("team_id"))
                for e in chk_events if b - window <= e["frame"] < b + window
            ],
            "reference_events_near_boundary": [
                (e["frame"], e["event_type"], e.get("player_id"), e.get("team_id"))
                for e in ref_events if b - window <= e["frame"] < b + window
            ],
        })
    report["boundaries"] = boundary_report
    return report
