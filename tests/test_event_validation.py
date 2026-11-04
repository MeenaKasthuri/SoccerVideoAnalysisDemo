"""Reference event matching boundaries and unavailable-accuracy behavior."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab_support"))
from event_validation import evaluate_events, interval_iou


def episode(start, end, **extra):
    return {"event_type": "divided_ball", "start_frame": start, "end_frame": end, **extra}


def point(frame, kind="possession_change", **extra):
    return {"event_type": kind, "frame": frame, **extra}


def reference(events, kinds=None, **extra):
    return {"fps": 25, "start_frame": 0, "end_frame": 99,
            "events": events, "complete_event_types": kinds or ["divided_ball", "possession_change"],
            "human_reviewed": False, **extra}


class EventReferenceTests(unittest.TestCase):
    def test_continuous_episode_is_counted_once_and_duplicate_predictions_are_false(self):
        result = evaluate_events([episode(10, 20), episode(10, 20)], reference([episode(11, 21)]))
        contest = result["per_type"][0]
        self.assertEqual((contest["tp"], contest["fp"], contest["fn"]), (1, 1, 0))
        self.assertEqual(contest["precision"], .5)

    def test_interval_iou_includes_single_frame_episodes(self):
        self.assertEqual(interval_iou((3, 3), (3, 3)), 1)
        self.assertEqual(interval_iou((3, 3), (4, 4)), 0)
        self.assertEqual(interval_iou((0, 1), (1, 2)), 1 / 3)

    def test_point_tolerance_boundary_uses_reference_fps(self):
        result = evaluate_events([point(15), point(26)], reference([point(10), point(20)]), tolerance_seconds=.2)
        change = result["per_type"][1]
        self.assertEqual((change["tp"], change["fp"], change["fn"]), (1, 1, 1))

    def test_event_types_do_not_cross_match(self):
        result = evaluate_events([point(10, "touch")], reference([point(10)], kinds=["possession_change"]))
        self.assertEqual(result["per_type"][0]["fn"], 1)

    def test_incomplete_labels_do_not_report_zero_errors(self):
        result = evaluate_events([point(10)], {"fps": 25, "events": []})
        self.assertEqual(result["status"], "unavailable_incomplete_reference")
        self.assertEqual(result["per_type"], [])

    def test_unreviewed_labels_never_become_human_accuracy(self):
        result = evaluate_events([point(10)], reference([point(10)]))
        self.assertFalse(result["human_reviewed"])
        self.assertFalse(result["accuracy_validated"])
        self.assertEqual(result["status"], "provisional_reference_review_pending")

    def test_uncertain_reference_regions_do_not_count_as_false_predictions(self):
        result = evaluate_events([episode(20, 25)], reference([episode(20, 30, uncertain=True)]))
        contest = result["per_type"][0]
        self.assertEqual((contest["tp"], contest["fp"], contest["fn"]), (0, 0, 0))
        self.assertEqual(contest["ignored_predictions"], 1)
        self.assertIsNone(contest["precision"])
        self.assertIsNone(contest["recall"])

    def test_window_boundary_predictions_are_reported_as_exclusions(self):
        result = evaluate_events([episode(90, 110)], reference([]))
        self.assertEqual(result["per_type"][0]["outside_or_boundary_predictions"], 1)
        self.assertEqual(result["per_type"][0]["fp"], 0)

    def test_source_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_events([], reference([], source_sha256="abc"), source_sha256="other")

    def test_old_per_frame_contest_records_and_invalid_bounds_are_rejected(self):
        for prediction in ({"event_type": "divided_ball", "frame": 2}, episode(4, 3), point(-1)):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                evaluate_events([prediction], reference([]))


if __name__ == "__main__":
    unittest.main()
