"""Small deterministic diagnostic tests, with no detector or OCR downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab_support"))
from short_validation import (detection_statistics, field_statistics, make_visual_review, tracking_statistics,
                              validation_key, verify_sample, sample_configuration)


class ShortValidationTests(unittest.TestCase):
    def test_samples_require_actual_expected_count_and_two_chunks(self):
        verify_sample({"decoded_frames": 60, "requested_frames": 60}, 30)
        with self.assertRaises(RuntimeError):
            verify_sample({"decoded_frames": 59, "requested_frames": 60}, 30)
        with self.assertRaises(ValueError):
            verify_sample({"decoded_frames": 30, "requested_frames": 30}, 30)

    def test_nonzero_clip_rebases_only_keyframes_inside_sample(self):
        polygon = [[0, 0], [1, 0], [1, 1]]
        original = {"filter_field": True, "field_keyframes": {0: polygon, "60": polygon, 90: polygon, 120: polygon}}
        shifted = sample_configuration(original, 60, 60)
        self.assertEqual(set(shifted["field_keyframes"]), {0, 30})
        self.assertEqual(set(original["field_keyframes"]), {0, "60", 90, 120})
        self.assertEqual(sample_configuration(original, 121, 30)["field_keyframes"], {})

    def test_continuity_candidates_record_source_frames(self):
        frames = [{"frame": 0, "players": [{"track_id": 1, "bbox": [0, 0, 10, 30]}]},
                  {"frame": 1, "players": [{"track_id": 2, "bbox": [0, 0, 10, 30]}]}]
        result = tracking_statistics(frames, [1], source_frame_offset=60)
        self.assertEqual(result["chunk_boundaries"][0]["source_frame"], 61)
        self.assertEqual(result["possible_fragmentation_review_candidates"][0]["source_last_frame"], 60)

    def test_short_runner_uses_same_model_and_nonzero_sample_for_both_modes(self):
        import chunked_pipeline as cp
        import colab_helpers as ch
        from short_validation import run_short_comparison
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model = root / "weights.pt"
            model.write_bytes(b"offline fixture; no inference")
            output = root / "comparison"
            sample = {"path": str(root / "sample.mp4"), "decoded_frames": 60,
                      "requested_frames": 60, "source_start_frame": 60}
            def run(video, destination, **kwargs):
                self.assertEqual(video, sample["path"])
                self.assertEqual(kwargs["model_path"], str(model))
                return {"status": "ok", "outputs": {"tracking_json": "unused", "annotated_video": "unused"},
                        "inference_device": "cuda:0", "chunk_starts": [0, 30], "ocr": {}}
            with patch.object(ch, "video_info", return_value={"duration_s": 10}), \
                    patch.object(ch, "make_sample", return_value=sample) as make_sample, \
                    patch.object(ch, "validate_outputs", return_value=(True, {"artifact": {"ok": True}})), \
                    patch.object(cp, "run_chunked", side_effect=run) as runner, \
                    patch("short_validation._read_json", return_value=[]), \
                    patch("short_validation.make_visual_review", return_value={}):
                result = run_short_comparison("source.mp4", output,
                    {"filter_field": False, "field_keyframes": {0: [[0, 0], [1, 0], [1, 1]]}},
                    start_s=2, duration_s=2, model_path=model)
            self.assertEqual(make_sample.call_args.args[2:], (2, 2))
            self.assertIsNone(runner.call_args_list[0].kwargs["improvements"])
            self.assertEqual(runner.call_args_list[1].kwargs["improvements"]["field_keyframes"], {})
            self.assertEqual(result["sample"]["source_start_frame"], 60)
            self.assertTrue(result["comparison_protocol"]["same_sample_for_both_runs"])

    def test_field_coverage_reports_missing_and_uncalibrated_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "field_regions.jsonl"
            unavailable = field_statistics(path, 10)
            self.assertEqual(unavailable["active_filter_frame_ratio"], 0)
            self.assertEqual(unavailable["spectator_filter_evaluation_status"], "not_evaluated_no_calibrated_frames")
            path.write_text('\n'.join(json.dumps(r) for r in [
                {"frame": 0, "filtering_active": True, "reason": "manual_keyframe"},
                {"frame": 1, "filtering_active": False, "reason": "insufficient_background_features"}]))
            partial = field_statistics(path, 10)
            self.assertEqual(partial["active_filter_frame_ratio"], 0.1)
            self.assertEqual(partial["inactive_or_unlogged_frames"], 9)
            self.assertFalse(partial["accuracy_validated"])

    def test_unknowns_gaps_and_two_chunk_boundary_are_diagnostics(self):
        def player(pid, team=None, number=None):
            return {"track_id": pid, "team": team, "jersey_number": number, "bbox": [10, 10, 30, 60]}
        frames = [
            {"frame": 0, "players": [player(1, 1)]},
            {"frame": 1, "players": [player(1, 1)]},
            {"frame": 2, "players": []},
            {"frame": 3, "players": [player(1, 1, "7"), player(2)]},
            {"frame": 4, "players": [player(3, 2)]},
        ]
        result = tracking_statistics(frames, [3])
        self.assertEqual(result["unique_tracker_ids_not_player_count"], 3)
        self.assertEqual(result["unknown_jersey_number_observations"], 4)
        self.assertEqual(result["unknown_team_observations"], 1)
        self.assertEqual(result["same_id_return_after_gap"][0]["missing_frames"], 1)
        self.assertEqual(result["chunk_boundaries"][0]["frame"], 3)
        self.assertTrue(result["possible_fragmentation_review_candidates"])
        self.assertIsNone(result["true_id_switches"])

    def test_detection_decisions_do_not_become_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "detections.jsonl"
            path.write_text('\n'.join(json.dumps(r) for r in [
                {"frame": 0, "kept": True, "reason": "boundary_margin"},
                {"frame": 1, "kept": False, "reason": "outside_field"}]))
            result, records = detection_statistics(path)
            self.assertEqual((result["retained"], result["rejected"]), (1, 1))
            self.assertIsNone(result["spectator_accuracy"])
            self.assertEqual(len(records), 2)

    def test_config_or_source_change_invalidates_key(self):
        a = validation_key({"enable_ocr": False}, {"files": {"a": "1"}})
        self.assertNotEqual(a, validation_key({"enable_ocr": True}, {"files": {"a": "1"}}))
        self.assertNotEqual(a, validation_key({"enable_ocr": False}, {"files": {"a": "2"}}))

    def test_side_by_side_artifacts_preserve_two_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ("baseline", "improved"):
                writer = cv2.VideoWriter(str(root / (name + ".avi")), cv2.VideoWriter_fourcc(*"MJPG"), 10, (80, 60))
                self.assertTrue(writer.isOpened())
                for index in range(10):
                    writer.write(np.full((60, 80, 3), index * 20, np.uint8))
                writer.release()
            review = root / "review"
            result = make_visual_review(root / "baseline.avi", root / "improved.avi", review,
                [{"frame": 5, "bbox": [5, 5, 20, 30], "kept": False, "reason": "outside_field"}], [5])
            self.assertEqual(result["frames"], 10)
            self.assertIn(4, result["sampled_frames"])
            self.assertIn(5, result["sampled_frames"])
            self.assertEqual(result["manual_labels_supplied"], 0)
            self.assertTrue((review / "manual_review.csv").is_file())
            self.assertTrue((review / "contact_sheet.jpg").is_file())
            cap = cv2.VideoCapture(str(review / "baseline_vs_improved.avi"))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
            cap.release()


if __name__ == "__main__":
    unittest.main()
