"""Short GPU comparison and review artifacts; counts are diagnostics, not accuracy."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def validation_key(config, manifest):
    payload = {"config": config, "files": manifest.get("files", {})}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def tracking_statistics(frames, boundaries=(), source_frame_offset=0):
    """Never infer true switches or real player counts from tracker IDs."""
    observations = defaultdict(list)
    teams, numbers = Counter(), Counter()
    unknown_team = unknown_number = 0
    count_per_frame = []
    for fallback, frame in enumerate(frames):
        index = int(frame.get("frame", fallback))
        players = frame.get("players", [])
        count_per_frame.append(len(players))
        for player in players:
            pid = str(player["track_id"])
            observations[pid].append((index, player))
            team = player.get("team")
            if team in (None, 0, "0", "unknown"):
                unknown_team += 1
            else:
                teams[str(team)] += 1
            number = player.get("jersey_number", player.get("number"))
            if number in (None, "", "unknown", "Unknown"):
                unknown_number += 1
            else:
                numbers[str(number)] += 1
    gaps, team_changes = [], []
    for pid, rows in observations.items():
        for previous, current in zip(rows, rows[1:]):
            if current[0] > previous[0] + 1:
                gaps.append({"track_id": pid, "last_frame": previous[0],
                             "return_frame": current[0], "missing_frames": current[0] - previous[0] - 1})
            if previous[1].get("team") not in (None, 0) and current[1].get("team") not in (None, 0) \
                    and previous[1].get("team") != current[1].get("team"):
                team_changes.append({"track_id": pid, "frame": current[0]})
    boundary_results = []
    for boundary in boundaries:
        if not 0 < boundary < len(frames):
            continue
        before = {str(p["track_id"]) for p in frames[boundary - 1].get("players", [])}
        after = {str(p["track_id"]) for p in frames[boundary].get("players", [])}
        boundary_results.append({"frame": boundary, "continuing_ids": sorted(before & after),
                                 "new_ids": sorted(after - before), "missing_ids": sorted(before - after)})
    # Nearby, non-overlapping tracklets are only review candidates. Camera motion,
    # exits/re-entry and different overlapping people can all cause these signals.
    candidates = []
    starts = sorted((rows[0][0], pid, rows[0][1]) for pid, rows in observations.items())
    for old_id, rows in observations.items():
        last_frame, old = rows[-1]
        a = old.get("bbox")
        if not a:
            continue
        for first_frame, new_id, new in starts:
            gap = first_frame - last_frame
            if not 0 < gap <= 15 or old_id == new_id:
                continue
            b = new.get("bbox")
            if not b:
                continue
            scale = max(a[3] - a[1], b[3] - b[1], 1)
            distance = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, a[3] - b[3])
            if distance <= scale * 0.75:
                candidates.append({"old_id": old_id, "new_id": new_id, "last_frame": last_frame,
                                   "first_frame": first_frame, "gap_frames": gap,
                                   "foot_distance_px": round(distance, 2), "manual_label": ""})
    for gap in gaps:
        gap.update(source_last_frame=gap["last_frame"] + source_frame_offset,
                   source_return_frame=gap["return_frame"] + source_frame_offset)
    for candidate in candidates:
        candidate.update(source_last_frame=candidate["last_frame"] + source_frame_offset,
                         source_first_frame=candidate["first_frame"] + source_frame_offset)
    for boundary in boundary_results:
        boundary["source_frame"] = boundary["frame"] + source_frame_offset
    return {
        "source_frame_offset": source_frame_offset,
        "frames": len(frames), "player_observations": sum(count_per_frame),
        "mean_players_visible": float(np.mean(count_per_frame)) if count_per_frame else 0,
        "unique_tracker_ids_not_player_count": len(observations),
        "observations_per_track": {pid: len(rows) for pid, rows in observations.items()},
        "team_observations": dict(teams), "unknown_team_observations": unknown_team,
        "jersey_number_observations": dict(numbers), "unknown_jersey_number_observations": unknown_number,
        "team_change_signals": team_changes, "same_id_return_after_gap": gaps,
        "possible_fragmentation_review_candidates": candidates, "chunk_boundaries": boundary_results,
        "true_id_switches": None, "real_player_count": None,
        "note": "Signals require human review; exits/re-entry and true identity switches are not automatically distinguishable.",
    }


def detection_statistics(path):
    counts = Counter()
    records = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
                    counts[(bool(record.get("kept")), str(record.get("reason", "unspecified")))] += 1
    return {"logged_detections": len(records), "retained": sum(n for (kept, _), n in counts.items() if kept),
            "rejected": sum(n for (kept, _), n in counts.items() if not kept),
            "by_decision_reason": [{"kept": kept, "reason": reason, "count": n}
                                   for (kept, reason), n in sorted(counts.items())],
            "spectator_accuracy": None,
            "note": "Rejected means rejected by a rule; it does not establish that the person was a spectator."}, records


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def make_visual_review(baseline_video, improved_video, output_dir, records, boundaries, max_frames=12,
                       source_frame_offset=0):
    """Side-by-side video, sampled JPEGs, contact sheet and unlabeled review CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    caps = [cv2.VideoCapture(str(path)) for path in (baseline_video, improved_video)]
    writer = None
    thumbnails, frame_paths, review_rows = [], {}, []
    try:
        if not all(cap.isOpened() for cap in caps):
            raise RuntimeError("Comparison videos could not be opened")
        n = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        if n < 1 or fps <= 0:
            raise RuntimeError("Comparison video has invalid metadata")
        sampled = {int(i) for i in np.linspace(0, n - 1, min(max_frames, n))}
        sampled.update(b for b in boundaries if 0 <= b < n)
        sampled.update(b - 1 for b in boundaries if 0 < b <= n)
        rejection_frames = sorted({int(r["frame"]) for r in records if not r.get("kept", True)})
        room = max(0, max_frames * 2 - len(sampled))
        if rejection_frames and room:
            sampled.update(rejection_frames[int(i)] for i in np.linspace(0, len(rejection_frames) - 1,
                                                                         min(room, len(rejection_frames))))
        rejected_by_frame = defaultdict(list)
        for record in records:
            if not record.get("kept", True):
                rejected_by_frame[int(record["frame"])].append(record)
        index = 0
        while True:
            reads = [cap.read() for cap in caps]
            if not reads[0][0] and not reads[1][0]:
                break
            if not all(ok for ok, _ in reads):
                raise RuntimeError("Baseline and improved videos have different decoded frame counts")
            panels = []
            for column, (_, frame) in enumerate(reads):
                if column == 1:
                    for record in rejected_by_frame[index]:
                        if record.get("bbox") is not None:
                            x1, y1, x2, y2 = map(int, record["bbox"])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                height, width = frame.shape[:2]
                panel = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
                cv2.rectangle(panel, (0, 0), (640, 26), (0, 0, 0), -1)
                title = ("BASELINE" if column == 0 else "IMPROVED (red = rejected)") + f" source frame {index + source_frame_offset}"
                cv2.putText(panel, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                panels.append(panel)
            combined = np.hstack(panels)
            if writer is None:
                writer = cv2.VideoWriter(str(output_dir / "baseline_vs_improved.avi"),
                                         cv2.VideoWriter_fourcc(*"XVID"), fps,
                                         (combined.shape[1], combined.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Could not open comparison video writer")
            writer.write(combined)
            if index in sampled:
                destination = output_dir / f"comparison_frame_{index:05d}.jpg"
                if not cv2.imwrite(str(destination), combined):
                    raise RuntimeError(f"Could not write {destination}")
                frame_paths[index] = destination.name
                thumbnails.append(cv2.resize(combined, (640, combined.shape[0] // 2)))
                review_rows.append({"frame": index, "source_frame": index + source_frame_offset, "image": destination.name,
                                    "review_target": "whole_frame", "reason": "Check retained false positives and missed players",
                                    "manual_label": "", "manual_bbox": "", "notes": ""})
            index += 1
        if index != n:
            raise RuntimeError(f"Comparison decoded {index} frames, expected {n}")
    finally:
        for cap in caps:
            cap.release()
        if writer is not None:
            writer.release()
    for record in records:
        if int(record["frame"]) in frame_paths:
            review_rows.append({"frame": record["frame"], "source_frame": int(record["frame"]) + source_frame_offset,
                                "image": frame_paths[int(record["frame"])],
                                "review_target": "retained_detection" if record.get("kept") else "rejected_detection",
                                "bbox": record.get("bbox"), "reason": record.get("reason", ""),
                                "manual_label": "", "manual_bbox": "", "notes": ""})
    _write_csv(output_dir / "manual_review.csv", review_rows,
               ["frame", "source_frame", "image", "review_target", "bbox", "reason", "manual_label", "manual_bbox", "notes"])
    if thumbnails and not cv2.imwrite(str(output_dir / "contact_sheet.jpg"), np.vstack(thumbnails)):
        raise RuntimeError("Could not write contact sheet")
    return {"frames": index, "sampled_frames": sorted(frame_paths), "manual_labels_supplied": 0,
            "instructions": "Fill manual_review.csv: correct_player, false_positive, missed_player, correct_rejection, false_rejection, unclear. Add missed-player bbox manually. Blank labels are not measured accuracy."}


def field_statistics(path, expected_frames):
    records = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    records[int(record["frame"])] = record
    active = sum(bool(record.get("filtering_active")) for frame, record in records.items()
                 if 0 <= frame < expected_frames)
    coverage = active / expected_frames if expected_frames else 0.0
    status = ("not_evaluated_no_calibrated_frames" if active == 0 else
              "partial_field_coverage_requires_review" if active < expected_frames else
              "field_coverage_complete_accuracy_unmeasured")
    return {"expected_frames": expected_frames, "logged_frames": len(records),
            "calibrated_or_propagated_frames": active, "active_filter_frame_ratio": coverage,
            "inactive_or_unlogged_frames": expected_frames - active,
            "reasons": dict(Counter(r.get("reason", "unknown") for r in records.values())),
            "spectator_filter_evaluation_status": status, "accuracy_validated": False}


def verify_sample(sample, chunk_size):
    if sample["decoded_frames"] != sample["requested_frames"]:
        raise RuntimeError(f"Sample decoded {sample['decoded_frames']} frames; requested {sample['requested_frames']}")
    if sample["decoded_frames"] <= chunk_size:
        raise ValueError("Sample must cover at least two consecutive chunks")


def validation_failures(summary, improvements):
    """Only require field coverage when the user enabled calibrated filtering."""
    failures = []
    if improvements.get("enable_ocr"):
        if not summary["ocr"].get("enabled"):
            failures.append("OCR requested but not enabled/reported by the pipeline")
        if summary["ocr"].get("backend_error") or summary["ocr"].get("error_count", 0):
            failures.append("Required OCR backend failed; inspect the OCR status/error fields")
    summary["field_filter"]["requested"] = bool(improvements.get("filter_field", True))
    if not summary["field_filter"]["requested"]:
        summary["field_filter"]["spectator_filter_evaluation_status"] = "disabled_no_spectator_quality_claim"
    if summary["field_filter"]["requested"] and summary["field_filter"]["calibrated_or_propagated_frames"] == 0:
        failures.append("No calibrated field frames: spectator filtering was not evaluated")
    return failures


def sample_configuration(improvements, source_start_frame, sample_frames):
    """Rebase only reviewed keyframes actually inside this clip; never carry an old polygon."""
    config = dict(improvements)
    config["field_keyframes"] = {
        int(frame) - source_start_frame: polygon
        for frame, polygon in improvements.get("field_keyframes", {}).items()
        if source_start_frame <= int(frame) < source_start_frame + sample_frames
    }
    return config


def run_short_comparison(source_video, output_dir, improvements, chunk_size=30, duration_s=2,
                         require_gpu=True, min_available_gb=2.0, start_s=0, model_path=None):
    import colab_helpers as ch
    import chunked_pipeline as cp

    if not math.isfinite(duration_s) or not 0 < duration_s <= 5:
        raise ValueError("Short validation is limited to at most five seconds")
    if not math.isfinite(start_s) or start_s < 0:
        raise ValueError("Short validation start must be a finite nonnegative number")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    source_info = ch.video_info(source_video)
    if start_s >= source_info["duration_s"]:
        raise ValueError("Short validation start is outside the source video")
    seconds = min(duration_s, source_info["duration_s"] - start_s)
    sample = ch.make_sample(source_video, root / "sample", start_s, seconds)
    verify_sample(sample, chunk_size)
    sample["requested_start_s"] = start_s
    sample["requested_duration_s"] = duration_s
    configuration = sample_configuration(improvements, sample["source_start_frame"], sample["decoded_frames"])
    selected_model = str(model_path or cp.MODEL_PATH)
    model_digest = hashlib.sha256()
    with open(selected_model, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            model_digest.update(block)
    protocol = {"name": "original_baseline_vs_identity_overlay_v1", "seed": 0,
                "gpu_warmup_detector_settings": ch.DETECT_KWARGS,
                "baseline_detector_settings": {"confidence_blurred": .22, "confidence_sharp": .25,
                                               "blur_threshold": 200, "iou": .35, "imgsz": 1280},
                "model_sha256": model_digest.hexdigest(),
                "model_path": selected_model, "chunk_size": chunk_size,
                "baseline_improvements": None, "same_sample_for_both_runs": True}
    runs = {}
    for label, run_configuration in (("baseline", None), ("improved", configuration)):
        run = cp.run_chunked(sample["path"], root / label, chunk_size=chunk_size,
                             require_gpu=require_gpu, min_available_gb=min_available_gb,
                             seed=0, improvements=run_configuration, model_path=selected_model)
        runs[label] = run
        if run["status"] != "ok":
            raise RuntimeError(f"{label} failed: {run.get('error')}; inspect {root / label}")
        valid, checks = ch.validate_outputs(run["outputs"], sample["decoded_frames"])
        run["validation"] = checks
        if not valid:
            raise RuntimeError(f"{label} outputs failed validation: {checks}")
        if require_gpu and not str(run.get("inference_device", "")).startswith("cuda"):
            raise RuntimeError(f"{label} was not verified on GPU")
        if len(run.get("chunk_starts", [])) < 2:
            raise RuntimeError(f"{label} did not process two consecutive chunks")
    protocol["tracking_configuration"] = {label: run.get("tracking_configuration", {})
                                            for label, run in runs.items()}
    boundaries = runs["improved"]["chunk_starts"][1:]
    summary = {"source": str(source_video), "sample": sample, "configuration": improvements,
               "sample_configuration": configuration, "comparison_protocol": protocol,
               "runs": runs, "status": "technical_checks_passed_review_required", "manual_review_approved": False}
    summary["ocr"] = runs["improved"].get("ocr", {})
    summary["field_filter"] = field_statistics(root / "improved" / "field_regions.jsonl", sample["decoded_frames"])
    failures = validation_failures(summary, configuration)
    if failures:
        summary["status"] = "validation_failed"
        summary["validation_errors"] = failures
    for label, run in runs.items():
        frames = _read_json(run["outputs"]["tracking_json"])
        summary[label] = tracking_statistics(frames, boundaries, sample["source_start_frame"])
    detection_summary, records = detection_statistics(root / "improved" / "detections.jsonl")
    for record in records:
        record["source_frame"] = int(record["frame"]) + sample["source_start_frame"]
    summary["detections"] = detection_summary
    _write_csv(root / "detection_decisions.csv", records,
               sorted({key for record in records for key in record}) or ["frame", "bbox", "role", "reason", "kept"])
    summary["visual_review"] = make_visual_review(
        runs["baseline"]["outputs"]["annotated_video"], runs["improved"]["outputs"]["annotated_video"],
        root / "review", records, boundaries, source_frame_offset=sample["source_start_frame"])
    _write_csv(root / "continuity_candidates.csv", summary["improved"]["possible_fragmentation_review_candidates"],
               ["old_id", "new_id", "last_frame", "first_frame", "source_last_frame", "source_first_frame",
                "gap_frames", "foot_distance_px", "manual_label"])
    with open(root / "comparison_summary.json", "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=str)
    _write_csv(root / "comparison_metrics.csv", [
        {"run": label, **{key: value for key, value in summary[label].items()
                          if isinstance(value, (str, int, float)) or value is None}}
        for label in ("baseline", "improved")],
        ["run", "frames", "player_observations", "mean_players_visible", "unique_tracker_ids_not_player_count",
         "unknown_team_observations", "unknown_jersey_number_observations", "true_id_switches", "real_player_count", "note"])
    if failures:
        raise RuntimeError("; ".join(failures) + f". Review {root / 'comparison_summary.json'}")
    return summary
