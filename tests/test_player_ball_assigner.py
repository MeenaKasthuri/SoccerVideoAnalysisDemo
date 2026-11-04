"""Ownership rule contracts, independent of model detections and human labels."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from player_ball_assigner import PlayerBallAssigner


class OwnershipRuleTests(unittest.TestCase):
    def test_default_reproduces_nearest_foot_rule_and_70px_boundary(self):
        assigner = PlayerBallAssigner()
        players = {1: {"bbox": [0, 0, 20, 100]}, 2: {"bbox": [100, 0, 120, 100]}}
        self.assertEqual(assigner.assign_ball_to_player(players, [58, 98, 62, 102]), 1)
        self.assertEqual(assigner.get_nearby_players({1: players[1]}, [88, 98, 92, 102]), [])
        self.assertEqual(assigner.get_nearby_players({1: players[1]}, [87, 98, 91, 102]),
                         [{"player_id": 1, "distance": 69.0}])

    def test_valid_baseline_matches_legacy_formula_for_many_geometries(self):
        assigner = PlayerBallAssigner()
        players = {i: {"bbox": [i * 23, i * 5, i * 23 + 15, i * 5 + 80]} for i in range(12)}
        for bx in range(0, 330, 7):
            ball = [bx - 2, 88, bx + 2, 92]
            expected = []
            for pid, p in players.items():
                x1, y1, x2, y2 = p["bbox"]
                distance = min(math.hypot(x1 - bx, y2 - 90), math.hypot(x2 - bx, y2 - 90))
                if distance < 70:
                    expected.append({"player_id": pid, "distance": distance})
            expected.sort(key=lambda row: row["distance"])
            self.assertEqual(assigner.get_nearby_players(players, ball), expected)

    def test_height_rule_is_invariant_under_uniform_resizing(self):
        players = {1: {"bbox": [100, 100, 130, 200]}, 2: {"bbox": [190, 70, 230, 200]}}
        ball = [162, 193, 168, 199]
        expected = None
        for scale in (0.5, 1, 2):
            scaled = {pid: {"bbox": [v * scale for v in p["bbox"]]} for pid, p in players.items()}
            assigner = PlayerBallAssigner(distance_mode="height_ratio")
            result = assigner.assign_ball_to_player(scaled, [v * scale for v in ball])
            expected = result if expected is None else expected
            self.assertEqual(result, expected)

    def test_normalized_ranking_can_differ_from_raw_distance(self):
        players = {1: {"bbox": [90, 160, 100, 200]},
                   2: {"bbox": [30, 0, 60, 200]}}
        ball = [118, 198, 122, 202]
        self.assertEqual(PlayerBallAssigner().assign_ball_to_player(players, ball), 1)
        self.assertEqual(PlayerBallAssigner(distance_mode="height_ratio").assign_ball_to_player(players, ball), 2)

    def test_invalid_and_missing_boxes_abstain_without_crash(self):
        bad = [None, [], [0, 0, 1], [0, 0, float("nan"), 1], [0, 0, 1, 0], [1, 0, 0, 1], [0, 0, "bad", 1]]
        valid = {1: {"bbox": [10, 10, 20, 100]}}
        for box in bad:
            with self.subTest(box=box):
                self.assertEqual(PlayerBallAssigner().assign_ball_to_player(valid, box), -1)
                self.assertEqual(PlayerBallAssigner().assign_ball_to_player({1: {"bbox": box}}, [18, 98, 22, 102]), -1)

    def test_normalized_rule_rejects_image_edge_truncation(self):
        players = {1: {"bbox": [0, 20, 20, 100]}}
        ball = [18, 98, 22, 102]
        self.assertEqual(PlayerBallAssigner().assign_ball_to_player(players, ball), 1)
        assigner = PlayerBallAssigner(distance_mode="height_ratio", image_size=(300, 200))
        self.assertEqual(assigner.assign_ball_to_player(players, ball), -1)

    def test_explicit_crouched_or_occluded_scale_flag_abstains(self):
        player = {"bbox": [20, 40, 50, 100], "scale_reliable": False}
        ball = [48, 98, 52, 102]
        self.assertEqual(PlayerBallAssigner().assign_ball_to_player({1: player}, ball), 1)
        self.assertEqual(PlayerBallAssigner(distance_mode="height_ratio").assign_ball_to_player({1: player}, ball), -1)
        player = {"bbox": [20, 40, 50, 100], "bbox_truncated": True}
        self.assertEqual(PlayerBallAssigner(distance_mode="height_ratio").assign_ball_to_player({1: player}, ball), -1)

    def test_optional_ambiguity_abstention_does_not_label_ball_free(self):
        players = {1: {"bbox": [20, 10, 40, 100]}, 2: {"bbox": [60, 10, 80, 100]}}
        ball = [48, 98, 52, 102]
        result = PlayerBallAssigner(abstain_ambiguous=True).ownership_observation(players, ball)
        self.assertEqual(result["status"], "ambiguous_proximity")
        self.assertIsNone(result["player_id"])
        self.assertFalse(result["physical_control_verified"])
        self.assertEqual(PlayerBallAssigner(abstain_ambiguous=True).assign_ball_to_player(players, ball), -1)
        self.assertEqual(PlayerBallAssigner().assign_ball_to_player(players, ball), 1)

    def test_optional_contest_requires_two_known_opposing_teams(self):
        assigner = PlayerBallAssigner(require_opposing_teams_for_contest=True)
        players = {1: {"bbox": [20, 10, 40, 100], "team": 1},
                   2: {"bbox": [60, 10, 80, 100], "team": "1"}}
        ball = [48, 98, 52, 102]
        self.assertIsNone(assigner.detect_divided_ball(players, ball))
        players[2]["team"] = 0
        self.assertIsNone(assigner.detect_divided_ball(players, ball))
        players[2]["team"] = 2
        self.assertIsNotNone(assigner.detect_divided_ball(players, ball))

    def test_legacy_contest_observation_remains_compatible(self):
        players = {1: {"bbox": [20, 10, 40, 100], "team": 1},
                   2: {"bbox": [60, 10, 80, 100], "team": 1}}
        self.assertEqual(PlayerBallAssigner().detect_divided_ball(players, [48, 98, 52, 102]),
                         {"player_ids": [1, 2], "distances": [10.0, 10.0]})

    def test_configuration_rejects_unknown_keys_and_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            PlayerBallAssigner.from_configuration({"max_distance": 7})
        for value in (0, -1, None, True, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PlayerBallAssigner(max_player_ball_height_ratio=value)
        with self.assertRaises(ValueError):
            PlayerBallAssigner(distance_mode="metres")

    def test_serializable_configuration_distinguishes_experimental_mode(self):
        import json
        assigner = PlayerBallAssigner.from_configuration({"distance_mode": "height_ratio"})
        result = json.loads(json.dumps(assigner.configuration))
        self.assertTrue(result["normalized_rule_experimental"])
        self.assertFalse(result["physical_control_verified"])

    def test_invalid_image_dimensions_fail_with_clear_configuration_error(self):
        for size in (7, [], [100], [100, "bad"], [100, None], [100, float("inf")], [100, True]):
            with self.subTest(size=size), self.assertRaises(ValueError):
                PlayerBallAssigner(image_size=size)


if __name__ == "__main__":
    unittest.main()
