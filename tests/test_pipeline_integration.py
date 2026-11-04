"""Offline plumbing checks: real ByteTrack/OpenCV, synthetic detections, no YOLO."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import supervision as sv
import torch

import chunked_pipeline as cp
from pipeline_tracker import make_tracker
from trackers.tracker import Tracker
from camera_motion.camera_motion import CameraMotionEstimator
from ultralytics.engine.results import Results

NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}


class FakeModel:
    """Synthetic YOLO outputs, respecting the detector's actual score cutoff."""
    def __init__(self, owner):
        self.owner = owner
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append(dict(kwargs))
        frames = source if isinstance(source, list) else [source]
        results = fake_detect(self.owner, frames)
        return [result[result.boxes.conf >= kwargs["conf"]] for result in results]


def fake_init(self, model_path):
    self.tracker = sv.ByteTrack()
    self.previous_ball_center = None
    self.model = FakeModel(self)
    self.test_frame = 0
    self.scripted_rows = None
    self.track_history = {}
    self.pretrack_history = {}
    self.last_ball_bbox = None
    self.smoothed_ball = None
    self.ball_alpha = .7
    self.previous_ids = set()
    self.motion_estimator = CameraMotionEstimator()
    self.ball_history = []


def fake_detect(self, frames, use_batch=False):
    results = []
    for frame in frames:
        # Empty frame exactly at chunk boundary must not stop tracking.
        if self.scripted_rows is not None:
            rows = self.scripted_rows[self.test_frame]
        else:
            rows = [] if self.test_frame == 5 else [
                [20, 20, 40, 75, .95, 2], [45, 20, 65, 75, .96, 1],
                [70, 20, 85, 75, .97, 3], [48, 72, 52, 76, .9, 0],
                [105, 5, 120, 25, .95, 2]]
        results.append(Results(frame, "synthetic", NAMES,
                               boxes=torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)))
        self.test_frame += 1
    return results


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.init_patch = patch.object(Tracker, "__init__", fake_init)
        self.init_patch.start()

    def tearDown(self):
        self.init_patch.stop()
        self.tmp.cleanup()

    def test_empty_boundary_keeps_state_and_goalkeepers(self):
        frame = np.zeros((96, 128, 3), np.uint8)
        tracker = make_tracker("unused", 60, {}, self.root)
        first = tracker.get_object_tracker([frame] * 5)
        second = tracker.get_object_tracker([frame] * 5)
        self.assertEqual(tracker.global_frame, 10)
        self.assertEqual(len(second["players"]), 5)
        self.assertEqual(second["players"][0], {})
        self.assertEqual(set(first["players"][-1]), set(second["players"][1]))
        self.assertIn("goalkeeper", [p["role"] for p in first["players"][0].values()])
        self.assertEqual(tracker.tracker.max_time_lost, 60)
        logs = [json.loads(line) for line in tracker.detection_log.read_text().splitlines()]
        self.assertEqual(max(row["frame"] for row in logs), 9)

    def test_pretrack_filter_diagnostics(self):
        frame = np.zeros((96, 128, 3), np.uint8)
        config = {"field_keyframes": {0: [[0, .5], [1, .5], [1, 1], [0, 1]]},
                  "boundary_margin_px": 2, "boundary_margin_box_ratio": 0}
        tracker = make_tracker("unused", 60, config, self.root)
        tracks = tracker.get_object_tracker([frame])
        self.assertEqual(len(tracks["players"][0]), 2)
        self.assertEqual(len(tracks["referees"][0]), 1)
        self.assertEqual(tracker.counts["outside_field"], 1)
        rejected = [json.loads(line) for line in tracker.detection_log.read_text().splitlines()
                    if not json.loads(line)["kept"]]
        self.assertEqual(rejected[0]["reason"], "outside_field")

    def test_full_two_chunk_exports_no_inference(self):
        video = self.root / "fixture.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60, (128, 96))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.full((96, 128, 3), 100, dtype=np.uint8))
        writer.release()
        with patch.object(cp.ch, "verify_tracker_device", return_value="cpu (mock)"):
            for label, config in (("baseline", None), ("improved", {})):
                summary = cp.run_chunked(video, self.root / label, chunk_size=5,
                    require_gpu=False, min_available_gb=0, seed=0, improvements=config,
                    console=io.StringIO())
                self.assertEqual(summary["status"], "ok", summary)
                self.assertEqual(summary["chunk_starts"], [0, 5])
                frames = json.loads(Path(summary["outputs"]["tracking_json"]).read_text())
                self.assertEqual(len(frames), 10)
                self.assertFalse(frames[0]["players"][0]["coordinate_valid"])
                self.assertIsNone(frames[0]["players"][0]["jersey_number"])
                self.assertIn("jersey_number_status", Path(summary["outputs"]["tracking_csv"]).read_text())
                csv_path = Path(summary["outputs"]["tracking_csv"])
                csv_path.write_text(csv_path.read_text().splitlines()[0] + "\n")
                valid, checks = cp.ch.validate_outputs(summary["outputs"], 10)
                self.assertFalse(valid)
                self.assertFalse(checks["tracking_csv"]["ok"])
                with self.assertRaises(FileExistsError):
                    cp.run_chunked(video, self.root / label, require_gpu=False)

    def test_entire_empty_clip_is_valid_data(self):
        video = self.root / "empty.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 60, (128, 96))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.zeros((96, 128, 3), dtype=np.uint8))
        writer.release()
        def empty_detect(self, frames, use_batch=False):
            return [Results(frame, "empty", NAMES, boxes=torch.empty((0, 6))) for frame in frames]
        with patch(__name__ + ".fake_detect", empty_detect), \
                patch.object(cp.ch, "verify_tracker_device", return_value="cpu (mock)"):
            summary = cp.run_chunked(video, self.root / "empty_run", chunk_size=5,
                require_gpu=False, min_available_gb=0, improvements={}, console=io.StringIO())
        self.assertEqual(summary["status"], "ok", summary)
        self.assertEqual(summary["validation"]["tracking_csv"]["rows"], 0)


class ContinuityTests(unittest.TestCase):
    """Exercise the installed ByteTrack; only model inference is synthesized."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.init_patch = patch.object(Tracker, "__init__", fake_init)
        self.init_patch.start()
        self.frame = np.zeros((96, 128, 3), np.uint8)

    def tearDown(self):
        self.init_patch.stop()
        self.tmp.cleanup()

    def tracker(self, config=None, baseline=False):
        return make_tracker("unused", 60, None if baseline else dict(config or {}), self.root)

    def test_low_score_continues_same_real_bytetrack_id_across_chunks(self):
        tracker = self.tracker()
        tracker.scripted_rows = [
            [[20, 20, 40, 75, .95, 2]],
            [[21, 20, 41, 75, .14, 2]],
            [[22, 20, 42, 75, .95, 2]],
        ]
        first = tracker.get_object_tracker([self.frame])
        second = tracker.get_object_tracker([self.frame, self.frame])
        original_id = set(first["players"][0])
        self.assertEqual(len(original_id), 1)
        self.assertEqual(original_id, set(second["players"][0]))
        self.assertEqual(original_id, set(second["players"][1]))
        self.assertEqual(tracker.counts["recovery_matches"], 1)
        self.assertAlmostEqual(tracker.tracker.track_activation_threshold, .25)
        self.assertAlmostEqual(tracker.tracker.det_thresh, .35)
        self.assertAlmostEqual(tracker.tracker.minimum_matching_threshold, .8)
        self.assertEqual(tracker.model.calls[0]["conf"], .1)
        self.assertEqual(tracker.model.calls[0]["iou"], .35)
        self.assertEqual(tracker.model.calls[0]["imgsz"], 1280)

    def test_isolated_low_score_box_cannot_create_a_new_id(self):
        tracker = self.tracker()
        tracker.scripted_rows = [[[20, 20, 40, 75, .14, 2]]] * 3
        tracks = tracker.get_object_tracker([self.frame] * 3)
        self.assertTrue(all(not frame for frame in tracks["players"]))
        self.assertEqual(tracker.tracker.tracked_tracks, [])
        self.assertEqual(tracker.counts["recovery_detections"], 3)
        self.assertEqual(tracker.counts["recovery_matches"], 0)

    def test_two_nearby_people_keep_distinct_ids(self):
        tracker = self.tracker()
        tracker.scripted_rows = [
            [[20, 20, 44, 75, .95, 2], [32, 20, 56, 75, .96, 2]],
            [[21, 20, 45, 75, .14, 2], [33, 20, 57, 75, .15, 2]],
        ]
        first = tracker.get_object_tracker([self.frame])
        second = tracker.get_object_tracker([self.frame])
        self.assertEqual(len(first["players"][0]), 2)
        self.assertEqual(set(first["players"][0]), set(second["players"][0]))
        left_ids = [pid for pid, value in first["players"][0].items() if value["bbox"][0] == 20]
        self.assertEqual(second["players"][0][left_ids[0]]["bbox"][0], 21)
        self.assertEqual(tracker.counts["duplicate"], 0)

    def test_long_weak_evidence_prevents_fragmentation_without_extending_buffer(self):
        # The weak interval exceeds both modes' buffers. Only accepting genuine
        # overlapping low-score evidence can preserve this continuously seen ID.
        sequence = ([[[20, 20, 40, 75, .95, 2]]]
                    + [[[20, 20, 40, 75, .14, 2]]] * 75
                    + [[[20, 20, 40, 75, .95, 2]]] * 3)
        for baseline in (True, False):
            with self.subTest(baseline=baseline):
                tracker = self.tracker(baseline=baseline)
                tracker.scripted_rows = sequence
                players = []
                for start in range(0, len(sequence), 5):
                    players.extend(tracker.get_object_tracker(
                        [self.frame] * min(5, len(sequence) - start))["players"])
                self.assertEqual(set(players[0]), {1})
                self.assertEqual(set(players[-1]), {2} if baseline else {1})
                if not baseline:
                    self.assertEqual(tracker.counts["recovery_matches"], 75)

    def test_baseline_uses_original_blur_cutoffs_and_does_not_recover_weak_box(self):
        tracker = self.tracker(baseline=True)
        tracker.scripted_rows = [[[20, 20, 40, 75, .95, 2]],
                                 [[21, 20, 41, 75, .14, 2]]]
        first = tracker.get_object_tracker([self.frame])
        second = tracker.get_object_tracker([self.frame])
        self.assertEqual(len(first["players"][0]), 1)
        self.assertEqual(second["players"][0], {})
        self.assertEqual([call["conf"] for call in tracker.model.calls], [.22, .22])
        self.assertEqual(tracker.tracker.max_time_lost, 30)

    def test_improved_lower_floor_preserves_ball_blur_thresholds(self):
        sharp = np.indices((96, 128)).sum(axis=0) % 2 * 255
        sharp = np.repeat(sharp[:, :, None], 3, axis=2).astype(np.uint8)
        tracker = self.tracker()
        self.assertGreaterEqual(tracker.frame_blur_score(sharp), 200)
        tracker.scripted_rows = [
            [[48, 72, 52, 76, .14, 0]],
            [[48, 72, 52, 76, .23, 0]],
            [[48, 72, 52, 76, .23, 0]],
            [[48, 72, 52, 76, .26, 0]],
        ]
        tracks = tracker.get_object_tracker([self.frame, self.frame, sharp, sharp])
        self.assertEqual([bool(row) for row in tracks["ball"]], [False, True, False, True])

    def test_batch_detector_keeps_order_and_recovery(self):
        tracker = self.tracker()
        tracker.scripted_rows = [
            [[20, 20, 40, 75, .95 if index == 0 else .14, 2]] for index in range(18)]
        tracks = tracker.get_object_tracker([self.frame] * 18, use_batch=True)
        self.assertEqual(len(tracker.model.calls), 2)
        self.assertEqual(len(tracks["players"]), 18)
        self.assertTrue(all(set(row) == {1} for row in tracks["players"]))

    def test_invalid_lower_floor_fails_before_model_initialization(self):
        for value in (0, .09, .22, .25, 1, float("nan"), float("inf"), "bad"):
            with self.subTest(value=value), patch.object(Tracker, "__init__") as init:
                with self.assertRaises((ValueError, TypeError)):
                    self.tracker({"detector_confidence_floor": value})
                init.assert_not_called()

    def test_detector_role_change_is_logged_without_identity_merge(self):
        tracker = self.tracker()
        tracker.scripted_rows = [[[20, 20, 40, 75, .95, 2]],
                                 [[20, 20, 40, 75, .95, 3]]]
        tracks = tracker.get_object_tracker([self.frame, self.frame])
        self.assertEqual(set(tracks["players"][0]), set(tracks["referees"][1]))
        events = [json.loads(line) for line in tracker.role_transition_log.read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["previous_role"], "player")
        self.assertEqual(events[0]["role"], "referee")
        self.assertEqual(tracker.counts["role_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
