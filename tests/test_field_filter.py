"""Synthetic geometry checks; these do not measure real soccer accuracy."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_support.field_filter import FieldFilter, duplicate_mask


class FieldFilterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(725)
        noise = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        self.frame = cv2.cvtColor(cv2.GaussianBlur(noise, (3, 3), 0), cv2.COLOR_GRAY2BGR)
        self.polygon = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

    def make_filter(self, **overrides):
        config = {"field_keyframes": {0: self.polygon}, "boundary_margin_px": 5,
                  "boundary_margin_box_ratio": 0.1}
        config.update(overrides)
        return FieldFilter(config)

    def test_no_calibration_fails_open(self):
        filt = FieldFilter()
        self.assertEqual(filt.update(self.frame, 0)["status"], "uncalibrated")
        self.assertFalse(filt.reject([0, 0, 10, 20])[0])

    def test_inside_outside_boundary_roles(self):
        filt = self.make_filter()
        filt.update(self.frame, 0)
        self.assertFalse(filt.reject([100, 90, 120, 150])[0])
        self.assertTrue(filt.reject([0, 50, 10, 80])[0])
        self.assertFalse(filt.reject([52, 50, 62, 80])[0])
        for role in ("goalkeeper", "referee"):
            self.assertFalse(filt.reject([49, 50, 59, 80], role)[0])
            self.assertTrue(filt.reject([0, 50, 10, 80], role)[0])

    def test_moving_camera_across_two_chunks(self):
        filt = self.make_filter()
        initial = np.asarray(filt.update(self.frame, 0)["polygon"])
        # One shared instance processes two consecutive chunks [0..2], [3..5].
        for chunk in ((1, 2), (3, 4, 5)):
            for index in chunk:
                affine = np.float32([[1, 0, 2 * index], [0, 1, index]])
                moved = cv2.warpAffine(self.frame, affine, (320, 240))
                state = filt.update(moved, index)
                self.assertEqual(state["status"], "propagated", state)
        np.testing.assert_allclose(np.asarray(state["polygon"]), initial + [10, 5], atol=0.8)

    def test_scene_cut_fails_open_and_keyframe_recovers(self):
        filt = self.make_filter(field_keyframes={0: self.polygon, 2: self.polygon})
        filt.update(self.frame, 0)
        cut = np.full_like(self.frame, 240)
        state = filt.update(cut, 1)
        self.assertFalse(state["filtering_active"])
        self.assertFalse(filt.reject([0, 50, 10, 80])[0])
        self.assertEqual(filt.update(cut, 2)["status"], "calibrated")
        self.assertTrue(filt.reject([0, 50, 10, 80])[0])

    def test_expiration_gap_and_dimension_change_fail_open(self):
        for next_frame, index, config, reason in (
                (self.frame, 1, {"max_propagation_frames": 0}, "calibration_expired"),
                (self.frame, 2, {}, "nonconsecutive_source_frames"),
                (self.frame[:100], 1, {}, "image_dimensions_changed")):
            filt = self.make_filter(**config)
            filt.update(self.frame, 0)
            self.assertEqual(filt.update(next_frame, index)["reason"], reason)
            self.assertFalse(filt.reject([0, 0, 10, 20])[0])

    def test_invalid_keyframe_fails_open(self):
        filt = self.make_filter(field_keyframes={0: [[0, 0], [1, 1], [1, 0], [0, 1]]})
        self.assertFalse(filt.update(self.frame, 0)["filtering_active"])

    def test_foreground_only_support_fails_open(self):
        filt = self.make_filter()
        filt.update(self.frame, 0, boxes=[[0, 0, 320, 240]])
        self.assertFalse(filt.update(self.frame, 1)["filtering_active"])

    def test_duplicates_choose_score_but_keep_overlaps_and_roles(self):
        boxes = [[10, 10, 30, 50], [10.2, 10.2, 30.2, 50.2],
                 [14, 10, 34, 50], [10, 10, 30, 50], [70, 10, 90, 50]]
        keep = duplicate_mask(boxes, [0.7, 0.9, 0.8, 0.8, 0.8],
                              ["player", "player", "player", "referee", "player"])
        np.testing.assert_array_equal(keep, [False, True, True, True, True])

    def test_empty_detections_and_invalid_boxes(self):
        self.assertEqual(len(duplicate_mask([], [], [])), 0)
        np.testing.assert_array_equal(duplicate_mask([[0, 0, 0, 0]], [0.5], ["player"]), [True])


if __name__ == "__main__":
    unittest.main()
