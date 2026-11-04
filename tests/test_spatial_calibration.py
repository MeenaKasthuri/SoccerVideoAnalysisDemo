import copy
import unittest

from spatial_calibration import CalibrationError, SpatialCalibrator

HASH = "a" * 64


def configuration():
    # Known affine image-to-ground mapping for an independently checkable fixture.
    def mark(name, image):
        return {"id": name, "image": image,
                "pitch": [(image[0] - 100) / 10, (image[1] - 80) / 10]}
    return {"schema_version": 1,
            "source": {"sha256": HASH, "width": 1280, "height": 800},
            "pitch": {"length_m": 105, "width_m": 68, "dimensions_verified": True},
            "segments": [{"id": "fixed_view", "start_frame": 10, "end_frame": 20,
                          "keyframe_frame": 10, "camera_model": "static",
                          "review": {"status": "approved", "reviewer": "Fixture reviewer",
                                     "reviewed_at": "2000-01-01T00:00:00Z"},
                          "fit_landmarks": [mark("a", [100, 80]), mark("b", [1150, 80]),
                                            mark("c", [1150, 760]), mark("d", [100, 760])],
                          "validation_landmarks": [mark("e", [625, 420]), mark("f", [300, 280])]}]}


class SpatialCalibrationTests(unittest.TestCase):
    def test_independent_ground_projection_and_interval(self):
        c = SpatialCalibrator(configuration(), HASH, (1280, 800))
        result = c.transform(15, (625, 420))
        self.assertTrue(result["coordinate_valid"])
        self.assertAlmostEqual(result["pitch_x"], 52.5, places=5)
        self.assertAlmostEqual(result["pitch_y"], 34, places=5)
        self.assertLess(c.reports()[0]["checkpoint_image_rmse_px"], .001)
        self.assertIsNone(c.transform(21, (625, 420))["pitch_x"])
        self.assertFalse(c.transform(9, (625, 420))["coordinate_valid"])

    def test_review_pending_never_grants_metric_validity(self):
        config = configuration()
        config["segments"][0]["review"] = {"status": "pending", "reviewer": None}
        c = SpatialCalibrator(config, HASH, (1280, 800))
        self.assertEqual(c.reports()[0]["quality_gate_reasons"], ["review_pending"])
        self.assertFalse(c.transform(10, (625, 420))["coordinate_valid"])

    def test_nominal_dimensions_and_source_mismatch_stay_invalid(self):
        config = configuration()
        config["pitch"]["dimensions_verified"] = False
        c = SpatialCalibrator(config, "b" * 64, (1280, 800))
        self.assertIn("pitch_dimensions_unverified", c.reports()[0]["quality_gate_reasons"])
        self.assertIn("source_hash_mismatch", c.reports()[0]["quality_gate_reasons"])

    def test_image_dimension_change_stays_invalid(self):
        c = SpatialCalibrator(configuration(), HASH, (1920, 1080))
        self.assertIn("source_image_dimensions_mismatch", c.reports()[0]["quality_gate_reasons"])
        self.assertFalse(c.transform(10, (625, 420))["coordinate_valid"])

    def test_heldout_error_catches_wrong_correspondence_despite_perfect_fit(self):
        config = configuration()
        config["segments"][0]["validation_landmarks"][0]["image"][0] += 50
        c = SpatialCalibrator(config, HASH, (1280, 800))
        r = c.reports()[0]
        self.assertLess(r["fit_image_rmse_px"], .001)
        self.assertGreater(r["checkpoint_image_max_error_px"], 49)
        self.assertIn("independent_checkpoint_error_exceeds_limit", r["quality_gate_reasons"])

    def test_training_landmarks_are_not_independent_validation(self):
        config = configuration()
        config["segments"][0]["validation_landmarks"][0] = copy.deepcopy(config["segments"][0]["fit_landmarks"][0])
        with self.assertRaises(CalibrationError):
            SpatialCalibrator(config, HASH, (1280, 800))
        config["segments"][0]["validation_landmarks"][0]["id"] = "new_name_same_position"
        with self.assertRaises(CalibrationError):
            SpatialCalibrator(config, HASH, (1280, 800))

    def test_too_few_checkpoints_do_not_grant_acceptance(self):
        config = configuration()
        config["segments"][0]["validation_landmarks"] = []
        c = SpatialCalibrator(config, HASH, (1280, 800))
        self.assertIn("insufficient_independent_checkpoints", c.reports()[0]["quality_gate_reasons"])

    def test_renamed_duplicate_checkpoints_are_not_independent(self):
        for field in ("image", "pitch"):
            config = configuration()
            points = config["segments"][0]["validation_landmarks"]
            points[1][field] = list(points[0][field])
            with self.subTest(field=field), self.assertRaises(CalibrationError):
                SpatialCalibrator(config, HASH, (1280, 800))

    def test_collinear_or_duplicate_fit_is_rejected(self):
        for duplicate in (False, True):
            config = configuration()
            for i, point in enumerate(config["segments"][0]["fit_landmarks"]):
                point["image"] = [100 + 100 * (0 if duplicate else i), 200]
            with self.assertRaises(CalibrationError):
                SpatialCalibrator(config, HASH, (1280, 800))

    def test_keyframe_does_not_leak_into_panning_interval(self):
        config = configuration()
        config["segments"][0]["camera_model"] = "keyframe_only"
        with self.assertRaises(CalibrationError):
            SpatialCalibrator(config, HASH, (1280, 800))
        config["segments"][0]["end_frame"] = 10
        c = SpatialCalibrator(config, HASH, (1280, 800))
        self.assertTrue(c.transform(10, (625, 420))["coordinate_valid"])
        self.assertFalse(c.transform(11, (625, 420))["coordinate_valid"])

    def test_ambiguous_intervals_and_nonfinite_coordinates_are_rejected(self):
        config = configuration()
        another = copy.deepcopy(config["segments"][0]);another["id"] = "overlap"
        config["segments"].append(another)
        with self.assertRaises(CalibrationError):
            SpatialCalibrator(config, HASH, (1280, 800))
        config = configuration();config["segments"][0]["fit_landmarks"][0]["image"][0] = float("nan")
        with self.assertRaises(CalibrationError):
            SpatialCalibrator(config, HASH, (1280, 800))

    def test_unconfigured_video_and_invalid_projection_abstain(self):
        config = configuration();config["segments"] = []
        c = SpatialCalibrator(config, HASH, (1280, 800))
        self.assertIsNone(c.transform(0, (625, 420))["pitch_x"])
        c = SpatialCalibrator(configuration(), HASH, (1280, 800))
        self.assertFalse(c.transform(10, (1270, 790))["coordinate_valid"])
        with self.assertRaises(CalibrationError):
            c.transform(10, (float("inf"), 420))

    def test_mutating_input_or_report_cannot_change_acceptance(self):
        config = configuration();config["segments"][0]["review"]["status"] = "pending"
        c = SpatialCalibrator(config, HASH, (1280, 800))
        config["segments"][0]["review"]["status"] = "approved"
        c.reports()[0]["quality_gate_reasons"].clear()
        self.assertFalse(c.transform(10, (625, 420))["coordinate_valid"])


if __name__ == "__main__":
    unittest.main()
