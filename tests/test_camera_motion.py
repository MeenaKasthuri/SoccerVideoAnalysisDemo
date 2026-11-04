"""Regression checks for incremental camera motion and failed-flow recovery."""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from camera_motion.camera_motion import CameraMotionEstimator


POINTS = np.array([[[20., 20.]], [[40., 40.]], [[60., 25.]]], dtype=np.float32)
VALID = np.ones((3, 1), dtype=np.uint8)


class CameraMotionTests(unittest.TestCase):
    def setUp(self):
        self.estimator = CameraMotionEstimator()
        self.frames = [np.full((80, 100, 3), value, dtype=np.uint8)
                       for value in (0, 20, 40, 60)]

    def assert_current_reference(self, frame):
        np.testing.assert_array_equal(
            self.estimator.prev_gray, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def successful_update(self, frame, dx, dy):
        with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=POINTS), \
             patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK',
                   return_value=(POINTS + np.array([dx, dy], dtype=np.float32), VALID, None)):
            return self.estimator.update_camera(frame)

    def test_first_frame_has_zero_motion(self):
        self.assertEqual(self.estimator.update_camera(self.frames[0]), (0., 0.))
        self.assert_current_reference(self.frames[0])

    def test_success_returns_increment_and_totals_accumulate(self):
        self.estimator.update_camera(self.frames[0])
        self.assertEqual(self.successful_update(self.frames[1], 4, -2), (4., -2.))
        self.assertEqual(self.successful_update(self.frames[2], 1, 3), (1., 3.))
        self.assertEqual((self.estimator.total_dx, self.estimator.total_dy), (5., 1.))

    def test_missing_features_returns_zero_without_reusing_total(self):
        self.estimator.update_camera(self.frames[0])
        self.successful_update(self.frames[1], 4, -2)
        with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=None):
            self.assertEqual(self.estimator.update_camera(self.frames[2]), (0., 0.))
        self.assertEqual((self.estimator.total_dx, self.estimator.total_dy), (4., -2.))
        self.assert_current_reference(self.frames[2])

    def test_empty_feature_array_does_not_call_optical_flow(self):
        self.estimator.update_camera(self.frames[0])
        with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack',
                   return_value=np.empty((0, 1, 2), dtype=np.float32)), \
             patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK') as flow:
            self.assertEqual(self.estimator.update_camera(self.frames[1]), (0., 0.))
            flow.assert_not_called()
        self.assert_current_reference(self.frames[1])

    def test_missing_optical_flow_outputs_return_zero_and_advance(self):
        for next_points, status in [(None, VALID), (POINTS, None), (None, None)]:
            with self.subTest(next_points=next_points is not None, status=status is not None):
                self.estimator = CameraMotionEstimator()
                self.estimator.update_camera(self.frames[0])
                self.successful_update(self.frames[1], 4, -2)
                with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=POINTS), \
                     patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK',
                           return_value=(next_points, status, None)):
                    self.assertEqual(self.estimator.update_camera(self.frames[2]), (0., 0.))
                self.assert_current_reference(self.frames[2])
                self.assertEqual((self.estimator.total_dx, self.estimator.total_dy), (4., -2.))

    def test_no_valid_matches_advances_reference_and_recovers(self):
        self.estimator.update_camera(self.frames[0])
        self.successful_update(self.frames[1], 4, -2)
        with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=POINTS), \
             patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK',
                   return_value=(POINTS, np.zeros_like(VALID), None)):
            self.assertEqual(self.estimator.update_camera(self.frames[2]), (0., 0.))
        self.assert_current_reference(self.frames[2])
        with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=POINTS), \
             patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK',
                   return_value=(POINTS + [2., 1.], VALID, None)) as flow:
            self.assertEqual(self.estimator.update_camera(self.frames[3]), (2., 1.))
            np.testing.assert_array_equal(
                flow.call_args.args[0], cv2.cvtColor(self.frames[2], cv2.COLOR_BGR2GRAY))
        self.assertEqual((self.estimator.total_dx, self.estimator.total_dy), (6., -1.))

    def test_empty_or_mismatched_flow_matches_advance_reference(self):
        for next_points, status in [
                (np.empty((0, 1, 2), dtype=np.float32), np.empty((0, 1), dtype=np.uint8)),
                (POINTS[:1], VALID), (POINTS, VALID[:1])]:
            with self.subTest(next_count=len(next_points), status_count=len(status)):
                self.estimator = CameraMotionEstimator()
                self.estimator.update_camera(self.frames[0])
                with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=POINTS), \
                     patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK',
                           return_value=(next_points, status, None)):
                    self.assertEqual(self.estimator.update_camera(self.frames[1]), (0., 0.))
                self.assert_current_reference(self.frames[1])

    def test_opencv_failure_advances_reference_and_next_call_recovers(self):
        for function in ('goodFeaturesToTrack', 'calcOpticalFlowPyrLK'):
            with self.subTest(function=function):
                self.estimator = CameraMotionEstimator()
                self.estimator.update_camera(self.frames[0])
                with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=POINTS), \
                     patch('camera_motion.camera_motion.cv2.' + function, side_effect=cv2.error('failed')):
                    self.assertEqual(self.estimator.update_camera(self.frames[1]), (0., 0.))
                self.assert_current_reference(self.frames[1])
                self.assertEqual(self.successful_update(self.frames[2], 2, -1), (2., -1.))

    def test_nonfinite_vectors_do_not_poison_totals(self):
        self.estimator.update_camera(self.frames[0])
        bad_points = POINTS.copy()
        bad_points[:] = np.nan
        with patch('camera_motion.camera_motion.cv2.goodFeaturesToTrack', return_value=POINTS), \
             patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK',
                   return_value=(bad_points, VALID, None)):
            self.assertEqual(self.estimator.update_camera(self.frames[1]), (0., 0.))
        self.assertEqual((self.estimator.total_dx, self.estimator.total_dy), (0., 0.))
        self.assert_current_reference(self.frames[1])

    def test_frame_size_change_resets_reference_without_optical_flow(self):
        self.estimator.update_camera(self.frames[0])
        resized = np.zeros((40, 50, 3), dtype=np.uint8)
        with patch('camera_motion.camera_motion.cv2.calcOpticalFlowPyrLK') as flow:
            self.assertEqual(self.estimator.update_camera(resized), (0., 0.))
            flow.assert_not_called()
        self.assert_current_reference(resized)

    def test_real_optical_flow_recovers_known_image_translation(self):
        image = np.zeros((160, 200, 3), dtype=np.uint8)
        for x in range(20, 190, 30):
            for y in range(20, 150, 30):
                cv2.rectangle(image, (x, y), (x + 6, y + 6), (255, 255, 255), -1)
        shifted = cv2.warpAffine(image, np.float32([[1, 0, 3], [0, 1, -2]]), (200, 160))
        self.estimator.update_camera(image)
        dx, dy = self.estimator.update_camera(shifted)
        self.assertAlmostEqual(dx, 3., delta=.15)
        self.assertAlmostEqual(dy, -2., delta=.15)


if __name__ == '__main__':
    unittest.main()
