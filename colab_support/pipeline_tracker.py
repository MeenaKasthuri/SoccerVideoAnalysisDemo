"""Colab-only adapter around the original detector, drawing and ByteTrack.

No re-identification or ID merging: team and number evidence remain metadata.
Baseline retains the original detection settings and class output behavior;
both paths handle empty detections without silently truncating the video.
Improved detection exposes ByteTrack's low-score continuation stage without
lowering the confidence required to create a new track. Ball thresholds stay
at the original blur-dependent values.
"""
import json
from pathlib import Path

import numpy as np
import supervision as sv

from field_filter import FieldFilter, duplicate_mask
from drawing_colors import ColorSafeDrawingMixin


def make_tracker(model_path, fps, improvements, output_dir):
    from trackers.tracker import Tracker

    class ColabTracker(ColorSafeDrawingMixin, Tracker):
        frame_offset = 0
        full_team_ball_control = None

        def __init__(self):
            self.improved = improvements is not None
            self.config = dict(improvements or {})
            self.detector_confidence_floor = float(self.config.get("detector_confidence_floor", .10))
            activation = float(self.config.get("track_activation_threshold", .25))
            if self.improved and (not np.isfinite(self.detector_confidence_floor)
                    or not .10 <= self.detector_confidence_floor < min(.22, activation)):
                raise ValueError("detector_confidence_floor must be finite, >= 0.10 and below "
                                 "both 0.22 and track_activation_threshold")
            super().__init__(model_path)
            self.global_frame = 0
            self.field = FieldFilter(self.config)
            self.detection_log = Path(output_dir) / "detections.jsonl"
            self.region_log = Path(output_dir) / "field_regions.jsonl"
            self.role_transition_log = Path(output_dir) / "role_transitions.jsonl"
            self.last_roles = {}
            if self.improved:
                # Buffer is measured in time: use source FPS instead of default 30.
                self.tracker = sv.ByteTrack(
                    frame_rate=max(1, round(fps)),
                    lost_track_buffer=int(self.config.get("lost_track_buffer", 30)),
                    track_activation_threshold=float(self.config.get("track_activation_threshold", .25)),
                    minimum_matching_threshold=float(self.config.get("minimum_matching_threshold", .8)),
                )
            self.counts = {"input_people": 0, "retained_people": 0,
                           "outside_field": 0, "duplicate": 0, "invalid_bbox": 0,
                           "recovery_detections": 0, "recovery_matches": 0,
                           "role_transitions": 0}
            self.tracking_config = {
                "detector_confidence_floor": self.detector_confidence_floor if self.improved else None,
                "ball_confidence_blurred": .22, "ball_confidence_sharp": .25,
                "detector_iou": .35, "detector_imgsz": 1280,
                "track_activation_threshold": self.tracker.track_activation_threshold,
                "new_track_threshold": self.tracker.det_thresh,
                "minimum_matching_threshold": self.tracker.minimum_matching_threshold,
                "max_time_lost_frames": self.tracker.max_time_lost,
                "identity_merging": False,
            }

        def detect_frames(self, frames, use_batch=False):
            if not self.improved:
                return super().detect_frames(frames, use_batch=use_batch)
            # ByteTrack can continue an active track from score > .10 and < .25.
            # The old .22/.25 detector discarded most such evidence beforehand.
            # Keep original image size/NMS and the existing GPU model placement.
            detections = []
            batch_size = 16 if use_batch else 1
            for start in range(0, len(frames), batch_size):
                batch = frames[start:start + batch_size]
                results = self.model.predict(
                    batch if use_batch else batch[0],
                    conf=self.detector_confidence_floor, iou=.35,
                    imgsz=1280, verbose=False)
                if len(results) != len(batch):
                    raise RuntimeError("Detector returned an incomplete batch")
                detections.extend(results)
            return detections

        def draw_team_ball_control(self, frame, frame_num, team_ball_control):
            return super().draw_team_ball_control(
                frame, self.frame_offset + frame_num, self.full_team_ball_control)

        def get_object_tracker(self, frames, read_from_stub=False, stub_path=None, use_batch=False):
            if read_from_stub or stub_path:
                raise ValueError("Colab validation requires fresh detections; stub IO is disabled")
            detections = self.detect_frames(frames, use_batch=use_batch)
            if len(detections) != len(frames):
                raise RuntimeError("Detector returned an incomplete chunk")
            tracks = {"players": [], "referees": [], "ball": []}
            with self.detection_log.open("a", encoding="utf-8") as log, \
                    self.region_log.open("a", encoding="utf-8") as regions, \
                    self.role_transition_log.open("a", encoding="utf-8") as role_events:
                for frame, result in zip(frames, detections):
                    g = self.global_frame
                    raw = sv.Detections.from_ultralytics(result)
                    names = result.names
                    ids = {name: key for key, name in names.items()}
                    missing = {"player", "goalkeeper", "referee", "ball"} - set(ids)
                    if missing:
                        raise ValueError(f"Model classes missing: {sorted(missing)}")
                    people = raw[np.isin(raw.class_id, [ids[r] for r in ("player", "goalkeeper", "referee")])]
                    roles = [names[int(c)] for c in people.class_id]
                    keep = np.ones(len(people), dtype=bool)
                    reasons = ["retained"] * len(people)
                    state = self.field.update(frame, g, people.xyxy) if self.improved else {"status": "baseline"}
                    regions.write(json.dumps({"frame": g, **state}) + "\n")
                    for i, box in enumerate(people.xyxy):
                        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                            keep[i], reasons[i] = False, "invalid_bbox"
                        elif self.improved and self.config.get("filter_field", True):
                            rejected, reason = self.field.reject(box, roles[i])
                            if rejected:
                                keep[i], reasons[i] = False, "outside_field"
                            else:
                                reasons[i] = reason
                    if self.improved and self.config.get("suppress_duplicates", True):
                        selected = np.flatnonzero(keep)
                        local = duplicate_mask(people.xyxy[selected], people.confidence[selected],
                                               [roles[i] for i in selected])
                        for i in selected[~local]:
                            keep[i], reasons[i] = False, "duplicate"
                    for i, box in enumerate(people.xyxy):
                        self.counts["input_people"] += 1
                        self.counts["retained_people"] += int(keep[i])
                        if not keep[i]:
                            self.counts[reasons[i]] += 1
                        recovery_only = bool(people.confidence[i] < self.tracker.track_activation_threshold)
                        if self.improved and keep[i] and recovery_only:
                            self.counts["recovery_detections"] += 1
                        log.write(json.dumps({"frame": g, "detection_index": i,
                            "bbox": box.tolist(), "confidence": float(people.confidence[i]),
                            "role": roles[i], "kept": bool(keep[i]), "reason": reasons[i],
                            "association_tier": "recovery_only" if recovery_only else "high_score",
                            "field_status": state.get("status")}) + "\n")
                    tracked = self.tracker.update_with_detections(people[keep])
                    players, referees, ball = {}, {}, {}
                    for box, _, confidence, cls, tid, _ in tracked:
                        role = names[int(cls)]
                        tid = int(tid)
                        if self.improved and confidence < self.tracker.track_activation_threshold:
                            self.counts["recovery_matches"] += 1
                        previous_role = self.last_roles.get(tid)
                        if previous_role is not None and previous_role != role:
                            self.counts["role_transitions"] += 1
                            role_events.write(json.dumps({"frame": g, "track_id": tid,
                                "previous_role": previous_role, "role": role,
                                "confidence": float(confidence), "bbox": box.tolist(),
                                "status": "detector_role_changed_requires_review"}) + "\n")
                        self.last_roles[tid] = role
                        record = {"bbox": box.tolist(), "role": role,
                                  "detection_confidence": float(confidence),
                                  "jersey_number": None, "jersey_number_status": "disabled"}
                        if role == "player" or (self.improved and role == "goalkeeper"):
                            players[int(tid)] = record
                        elif role == "referee":
                            referees[int(tid)] = record
                    # Preserve original ball confidence selection/jump logic.
                    balls = raw[raw.class_id == ids["ball"]]
                    if self.improved:
                        ball_floor = .22 if self.frame_blur_score(frame) < 200 else .25
                        balls = balls[balls.confidence >= ball_floor]
                    if len(balls):
                        box = balls.xyxy[int(np.argmax(balls.confidence))].tolist()
                        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        if self.previous_ball_center is None or np.linalg.norm(
                                np.subtract(center, self.previous_ball_center)) <= 200:
                            ball[1] = {"bbox": box}
                            self.previous_ball_center = center
                    tracks["players"].append(players)
                    tracks["referees"].append(referees)
                    tracks["ball"].append(ball)
                    self.global_frame += 1
            return tracks

    return ColabTracker()
