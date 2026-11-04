"""Exercise the installed OpenCV drawing paths without loading YOLO weights."""
import copy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import supervision as sv

import colab_helpers as ch
from chunked_pipeline import StreamingTeamAssigner
from drawing_colors import ColorSafeDrawingMixin, UNKNOWN_BGR, normalize_bgr
from pipeline_tracker import make_tracker
from trackers.tracker import Tracker


def drawing_init(self, model_path):
    self.model = object()
    self.tracker = sv.ByteTrack()
    self.track_history = {}
    self.pretrack_history = {}
    self.ball_history = []
    self.last_ball_bbox = None
    self.motion_estimator = SimpleNamespace(update_camera=lambda frame: (0, 0))


class NormalizeBgrTests(unittest.TestCase):
    def test_supported_colors_become_actual_python_ints(self):
        colors = [np.array([12, 127, 250], dtype=dtype)
                  for dtype in (np.int64, np.int32, np.uint8, np.float32)]
        colors += [(12, 127, 250), [12, 127, 250], (12., 127., 250.),
                   tuple(np.array([12., 127., 250.]).astype(int))]
        for color in colors:
            with self.subTest(color=repr(color)):
                converted = normalize_bgr(color)
                self.assertEqual(converted, (12, 127, 250))
                self.assertIs(type(converted), tuple)
                self.assertTrue(all(type(channel) is int for channel in converted))

    def test_clip_round_and_input_preservation(self):
        color = np.array([-20.1, 127.6, 400.2], dtype=np.float32)
        original = color.copy()
        self.assertEqual(normalize_bgr(color), (0, 128, 255))
        np.testing.assert_array_equal(color, original)
        self.assertEqual(normalize_bgr([12.5, 13.5, 254.6]), (12, 14, 255))
        self.assertEqual(normalize_bgr([-(1 << 4096), 0, 1 << 4096]), (0, 0, 255))

    def test_malformed_values_raise_clear_value_error(self):
        invalid = [None, True, 5, "123", (1, 2), (1, 2, 3, 4),
                   (True, 2, 3), (np.bool_(False), 2, 3), ("1", 2, 3),
                   (1 + 0j, 2, 3), (float("nan"), 2, 3),
                   (1, float("inf"), 3), (1, 2, -float("inf")),
                   [[1], [2], [3]], np.array([[1, 2, 3]]),
                   np.array([1, 2, 3], dtype=complex), np.array([1, 2, 3], dtype=str)]
        for color in invalid:
            with self.subTest(color=repr(color)):
                with self.assertRaisesRegex(ValueError, "exactly three finite real numeric"):
                    normalize_bgr(color)


class DrawingColorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.init_patch = patch.object(Tracker, "__init__", drawing_init)
        self.init_patch.start()
        self.frame = np.zeros((200, 320, 3), dtype=np.uint8)

    def tearDown(self):
        self.init_patch.stop()
        self.temp.cleanup()

    def active(self, improved=False):
        tracker = make_tracker("unused", 30, {} if improved else None, self.temp.name)
        tracker.full_team_ball_control = np.array([0, 0, 0])
        return tracker

    def test_actual_annotation_paths_accept_numpy_unknown_and_missing_team_colors(self):
        colors = ["missing", None, np.array([0, 0, 255], dtype=np.int64),
                  np.array([20, 80, 140], dtype=np.int32),
                  np.array([30, 90, 150], dtype=np.uint8),
                  np.array([40.2, 100.7, 160.1], dtype=np.float32),
                  tuple(np.array([50, 110, 170]).astype(int)),
                  [60., 120., 180.], (70, 130, 190)]
        for improved in (False, True):
            for color in colors:
                with self.subTest(improved=improved, color=repr(color)):
                    tracker = self.active(improved)
                    player = {"bbox": [80, 50, 105, 125], "has_ball": True, "team": 0}
                    if not isinstance(color, str):
                        player["team_color"] = color
                    tracks = {"players": [{1: copy.deepcopy(player)} for _ in range(3)],
                              "referees": [{2: {"bbox": [180, 50, 200, 125]}} for _ in range(3)],
                              "ball": [{1: {"bbox": [140 + i, 110, 145 + i, 115]}} for i in range(3)]}
                    before = copy.deepcopy(tracks)
                    rendered = tracker.draw_annotations([self.frame] * 3, tracks, np.array([0, 0, 0]))
                    self.assertEqual(len(rendered), 3)
                    self.assertEqual(len(tracker.track_history[1]), 3)
                    expected = UNKNOWN_BGR if color is None or isinstance(color, str) else normalize_bgr(color)
                    self.assertTrue(np.any(np.all(rendered[-1] == expected, axis=2)))
                    for index in range(3):
                        self.assertEqual(tracks["players"][index][1].keys(), before["players"][index][1].keys())
                        for key, value in before["players"][index][1].items():
                            if isinstance(value, np.ndarray):
                                np.testing.assert_array_equal(tracks["players"][index][1][key], value)
                                self.assertEqual(tracks["players"][index][1][key].dtype, value.dtype)
                            else:
                                self.assertEqual(tracks["players"][index][1][key], value)

    def test_improved_gray_stays_gray_and_none_uses_original_red(self):
        tracker = self.active(True)
        box = [80, 50, 105, 125]
        gray = np.array([128, 128, 128], dtype=np.int64)
        gray_frame = tracker.draw_ellipse(self.frame.copy(), box, gray)
        self.assertTrue(np.any(np.all(gray_frame == (128, 128, 128), axis=2)))
        none_frame = tracker.draw_ellipse(self.frame.copy(), box, None)
        self.assertTrue(np.any(np.all(none_frame == UNKNOWN_BGR, axis=2)))
        np.testing.assert_array_equal(gray, [128, 128, 128])

    def test_unknown_team_markers_do_not_change_identity_or_color_policy(self):
        for team in (0, None, "unknown", -1):
            with self.subTest(team=team):
                tracker = self.active()
                player = {"bbox": [80, 50, 105, 125], "team": team, "team_color": None}
                tracks = {"players": [{7: player}], "referees": [{}], "ball": [{}]}
                rendered = tracker.draw_annotations([self.frame], tracks, np.zeros(1))
                self.assertEqual(set(tracks["players"][0]), {7})
                self.assertEqual(player["team"], team)
                self.assertIsNone(player["team_color"])
                self.assertTrue(np.any(np.all(rendered[0] == UNKNOWN_BGR, axis=2)))

    def test_normalized_numpy_colors_work_on_all_requested_opencv_primitives(self):
        frame = self.frame.copy()
        color = normalize_bgr(tuple(np.array([12, 127, 250], dtype=np.int64)))
        cv2.rectangle(frame, (10, 10), (40, 40), color, 2)
        cv2.circle(frame, (80, 30), 12, color, 2)
        cv2.putText(frame, "7", (110, 40), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        cv2.ellipse(frame, (160, 30), (20, 10), 0, 0, 360, color, 2)
        self.assertTrue(np.any(np.all(frame == color, axis=2)))

    def test_baseline_tiny_crop_unknown_team_uses_native_red_fallback(self):
        # This is the actual narrow edge-box shape from the failed CUDA output.
        box = [0.0, 258.4487915, 6.552979946, 318.67739868]
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        players = [{3: {"bbox": box.copy()}} for _ in range(3)]
        assigner = StreamingTeamAssigner(players)
        assigner.process_chunk(0, [frame] * 3)
        assigner.finish()
        for player_frame in players:
            player = player_frame[3]
            self.assertEqual(player["team"], 0)
            self.assertEqual(player["team_color"], UNKNOWN_BGR)
            self.assertIs(type(player["team_color"]), tuple)
            self.assertTrue(all(type(channel) is int for channel in player["team_color"]))
        tracker = self.active()
        tracks = {"players": players, "referees": [{} for _ in range(3)],
                  "ball": [{} for _ in range(3)]}
        self.assertEqual(len(tracker.draw_annotations([frame] * 3, tracks, np.zeros(3))), 3)

    def test_every_drawing_boundary_normalizes_numpy_colors(self):
        tracker = self.active()
        tracker.ball_history = [(10, 10), (20, 20)]
        color = tuple(np.array([12, 127, 250], dtype=np.int64))
        box = [80, 50, 105, 125]
        tracker.draw_ellipse(self.frame.copy(), box, color)
        tracker.draw_trajectory(self.frame.copy(), 1, box, color)
        tracker.draw_trajectory(self.frame.copy(), 1, [81, 50, 106, 125], color)
        tracker.draw_triangle(self.frame.copy(), box, color)
        tracker.draw_ball_trail(self.frame.copy(), color)

    def test_malformed_color_errors_are_not_silently_replaced(self):
        tracker = self.active()
        box = [80, 50, 105, 125]
        for color in ([1, 2], ("red", 0, 0), (1, float("nan"), 3), (True, 2, 3)):
            calls = [lambda: tracker.draw_ellipse(self.frame, box, color),
                     lambda: tracker.draw_trajectory(self.frame, 1, box, color),
                     lambda: tracker.draw_triangle(self.frame, box, color),
                     lambda: tracker.draw_ball_trail(self.frame, color)]
            for call in calls:
                with self.subTest(color=repr(color), call=call):
                    with self.assertRaisesRegex(ValueError, "finite real numeric"):
                        call()

    def test_unrelated_opencv_geometry_error_propagates(self):
        tracker = self.active()
        with self.assertRaises(cv2.error):
            tracker.draw_ellipse(self.frame, [105, 50, 80, 125], np.array([0, 0, 255], dtype=np.int64))

    def test_legacy_helper_applies_same_drawing_guard_without_repeating_mixin(self):
        with patch.object(ch, "verify_tracker_device", return_value="cpu (mock)") as verify:
            tracker_class = ch.make_verified_tracker_class(Tracker, require_gpu=False)
            tracker = tracker_class("unused")
            self.assertIsInstance(tracker, ColorSafeDrawingMixin)
            tracker.draw_ellipse(self.frame.copy(), [80, 50, 105, 125], np.array([0, 0, 255]))
            tracker.draw_trajectory(self.frame.copy(), 1, [80, 50, 105, 125], np.array([0, 0, 255]))
            tracker.draw_trajectory(self.frame.copy(), 1, [81, 50, 106, 125], np.array([0, 0, 255]))
            verify.assert_called_once()
            existing_safe = type(self.active())
            wrapped = ch.make_verified_tracker_class(existing_safe, require_gpu=False)
            self.assertEqual(wrapped.__mro__.count(ColorSafeDrawingMixin), 1)


if __name__ == "__main__":
    unittest.main()
