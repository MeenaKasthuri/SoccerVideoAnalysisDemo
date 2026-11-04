"""Offline logic tests: no detector, OCR model download, or source mutations."""
from pathlib import Path
import copy
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "colab_support"))
from identity_evidence import EvidenceTeamAssigner, JerseyNumberRecognizer, StableTeamAssigner


def team_frame():
    frame = np.zeros((120, 140, 3), dtype=np.uint8)
    frame[10:110, 5:60] = (0, 0, 230)
    frame[10:110, 75:130] = (230, 30, 0)
    return frame


def players():
    return {1: {"bbox": [5, 10, 60, 110], "role": "player"},
            2: {"bbox": [75, 10, 130, 110], "role": "player"}}


class FakeReader:
    def __init__(self, texts=None):
        self.texts = iter(texts or [("12", .98)] * 20)
        self.calls = 0

    def readtext(self, crop, **kwargs):
        self.calls += 1
        value, confidence = next(self.texts)
        return [([[4, 4], [23, 4], [23, 24], [4, 24]], value, confidence)]


class IdentityEvidenceTests(unittest.TestCase):
    def test_team_reuses_original_and_corrects_display_color(self):
        a = EvidenceTeamAssigner()
        red = a.get_player_color(team_frame(), [5, 10, 60, 110])
        blue = a.get_player_color(team_frame(), [75, 10, 130, 110])
        self.assertTrue(a.assign_team_color_from_colors([red] * 10 + [blue] * 10))
        displays = list(a.team_colors.values())
        self.assertTrue(any(np.linalg.norm(c - [0, 0, 230]) < 5 for c in displays))
        self.assertTrue(any(np.linalg.norm(c - [230, 30, 0]) < 5 for c in displays))

    def test_two_chunks_keep_consensus_and_do_not_rename_ids(self):
        all_players = [players() for _ in range(12)]
        a = StableTeamAssigner(all_players, min_colors=10, sample_gap=1)
        a.process_chunk(0, [team_frame()] * 6)
        a.process_chunk(6, [team_frame()] * 6)
        a.finish()
        self.assertTrue(a.fitted)
        self.assertEqual(set(all_players[-1]), {1, 2})
        self.assertIn(all_players[-1][1]["team"], {1, 2})
        self.assertNotEqual(all_players[-1][1]["team"], all_players[-1][2]["team"])
        self.assertEqual(all_players[5][1]["team"], all_players[-1][1]["team"])
        self.assertEqual(all_players[0][1]["team"], 0)

    def test_goalkeeper_and_invalid_crops_remain_unknown(self):
        data = [{1: {"bbox": [0, 0, 0, 0]}, 2: {"bbox": [5, 10, 60, 110], "role": "goalkeeper"}}]
        a = StableTeamAssigner(data, max_pending=4)
        a.process_chunk(0, [team_frame()])
        a.finish()
        self.assertEqual(data[0][1]["team"], 0)
        self.assertEqual(data[0][2]["team"], 0)
        self.assertIsNone(a.assigner.get_player_color(team_frame(), [float("nan"), 0, 30, 40]))

    def test_weak_detections_do_not_train_team_clusters(self):
        data = [players() for _ in range(12)]
        for frame_players in data:
            for player in frame_players.values():
                player["detection_confidence"] = .24
        a = StableTeamAssigner(data, min_colors=10, sample_gap=1)
        a.process_chunk(0, [team_frame()] * len(data))
        a.finish()
        self.assertFalse(a.fitted)
        self.assertEqual(len(a.colors), 0)
        self.assertEqual(a.histories, {})
        self.assertTrue(all(player["team"] == 0 for frame in data for player in frame.values()))
        self.assertTrue(all(set(frame) == {1, 2} for frame in data))

    def test_weak_continuations_do_not_add_color_votes(self):
        data = [players() for _ in range(14)]
        for frame_players in data[12:]:
            # Even a clear opposing crop cannot change evidence when its
            # detection is too weak to support a reliable shirt observation.
            frame_players[1].update(bbox=[75, 10, 130, 110], detection_confidence=.15)
        a = StableTeamAssigner(data, min_colors=10, sample_gap=1)
        a.process_chunk(0, [team_frame()] * 12)
        previous = copy.deepcopy(data[11][1])
        a.process_chunk(12, [team_frame()] * 2)
        self.assertEqual(data[13][1]["team"], previous["team"])
        self.assertEqual(data[13][1]["team_evidence"]["votes"], previous["team_evidence"]["votes"])
        self.assertFalse(data[13][1]["team_evidence"]["current_observation_eligible"])
        self.assertEqual(set(data[13]), {1, 2})

    def test_latest_opposing_color_suppresses_established_majority_immediately(self):
        data = [players() for _ in range(14)]
        data[12][1]["bbox"] = [75, 10, 130, 110]
        data[13][1].update(bbox=[75, 10, 130, 110], detection_confidence=.15)
        a = StableTeamAssigner(data, min_colors=10, sample_gap=1)
        a.process_chunk(0, [team_frame()] * 12)
        original_team = data[11][1]["team"]
        self.assertIn(original_team, {1, 2})
        a.process_chunk(12, [team_frame()] * 2)
        self.assertGreater(data[12][1]["team_evidence"]["agreement"], .8)
        self.assertEqual(data[12][1]["team"], 0)
        self.assertEqual(data[12][1]["team_status"], "conflicting_colors")
        # A weak next frame supplies no evidence resolving the contradiction.
        self.assertEqual(data[13][1]["team"], 0)
        self.assertEqual(data[11][1]["team"], original_team)
        self.assertEqual(set(data[12]), {1, 2})

    def test_matching_shirt_on_new_id_does_not_inherit_old_identity_evidence(self):
        data = [players() for _ in range(12)]
        data.extend({31: players()[1]} for _ in range(3))
        a = StableTeamAssigner(data, min_colors=10, sample_gap=1)
        a.process_chunk(0, [team_frame()] * 12)
        old_team = data[11][1]["team"]
        a.process_chunk(12, [team_frame()] * 3)
        self.assertEqual(set(data[12]), {31})
        self.assertEqual(data[12][31]["team"], 0)
        self.assertEqual(data[12][31]["team_evidence"]["votes"], 1)
        self.assertEqual(data[13][31]["team"], 0)
        self.assertEqual(data[14][31]["team"], old_team)
        self.assertFalse(data[14][31]["team_evidence"]["identity_verified"])

    def test_unresolved_warmup_is_bounded(self):
        data = [{1: {"bbox": [5, 10, 60, 110]}} for _ in range(40)]
        a = StableTeamAssigner(data, min_colors=10, sample_gap=1, max_pending=4, max_colors=12)
        a.process_chunk(0, [team_frame()] * 40)
        self.assertFalse(a.fitted)
        self.assertLessEqual(len(a.colors), 12)
        self.assertLessEqual(len(a.pending), 4)

    def test_team_assignment_can_correct_and_keeps_previous_frame_results(self):
        data = [players() for _ in range(36)]
        for frame_players in data[6:]:
            frame_players[1]["bbox"] = [75, 10, 130, 110]
        a = StableTeamAssigner(data, min_colors=10, sample_gap=1)
        a.process_chunk(0, [team_frame()] * 6)
        initial = data[5][1]["team"]
        old_frame = copy.deepcopy(data[5][1])
        a.process_chunk(6, [team_frame()] * 30)
        self.assertIn(initial, {1, 2})
        self.assertIn(data[-1][1]["team"], {1, 2})
        self.assertNotEqual(data[-1][1]["team"], initial)
        self.assertEqual(data[5][1]["team"], old_frame["team"])
        self.assertEqual(data[5][1]["team_evidence"], old_frame["team_evidence"])
        np.testing.assert_array_equal(data[5][1]["team_color"], old_frame["team_color"])
        self.assertTrue(any(f[1]["team"] == 0 for f in data[6:15]))

    def test_team_history_expires_and_requires_new_evidence(self):
        data = [players() for _ in range(6)] + [{} for _ in range(8)] + [players()]
        a = StableTeamAssigner(data, min_colors=10, sample_gap=1, history_frames=5)
        a.process_chunk(0, [team_frame()] * 6)
        self.assertNotEqual(data[5][1]["team"], 0)
        a.process_chunk(6, [team_frame()] * 8)
        self.assertEqual(a.histories, {})
        a.process_chunk(14, [team_frame()])
        self.assertEqual(data[14][1]["team"], 0)
        self.assertEqual(data[14][1]["team_evidence"]["votes"], 1)

    def test_ocr_default_disabled(self):
        a = JerseyNumberRecognizer(reader=FakeReader())
        p = players()
        a.process(team_frame(), p, 0)
        self.assertEqual(p[1]["jersey_number_status"], "disabled")
        self.assertEqual(a.reader.calls, 0)

    def test_weak_detections_supply_no_ocr_votes_or_backend_calls(self):
        a = JerseyNumberRecognizer(enabled=True, reader=FakeReader(),
                                   sample_gap=5, min_sharpness=0)
        p = {1: players()[1]}
        p[1]["detection_confidence"] = .24
        for g in [0, 5, 10]:
            a.process(team_frame(), p, g)
            self.assertIsNone(p[1]["jersey_number"])
            self.assertEqual(p[1]["jersey_number_status"], "low_detection_confidence")
            self.assertEqual(p[1]["jersey_number_evidence"]["observations"], 0)
        self.assertEqual(a.reader.calls, 0)
        self.assertEqual(a.summary["readable_observations"], 0)
        # The threshold is inclusive; only these eligible frames may vote.
        p[1]["detection_confidence"] = .25
        for g in [15, 20, 25]:
            a.process(team_frame(), p, g)
        self.assertEqual(p[1]["jersey_number"], 12)
        self.assertEqual(a.reader.calls, 3)
        self.assertEqual(p[1]["jersey_number_evidence"]["frames"], [15, 20, 25])

    def test_weak_ocr_continuation_preserves_only_existing_consensus(self):
        a = JerseyNumberRecognizer(enabled=True, reader=FakeReader(), min_sharpness=0)
        p = {1: players()[1]}
        for g in [0, 5, 10]:
            a.process(team_frame(), p, g)
        p[1]["detection_confidence"] = .15
        # A new track at the same place cannot inherit the old track's number.
        p[31] = {**players()[1], "detection_confidence": .15}
        a.process(team_frame(), p, 15)
        self.assertEqual(set(p), {1, 31})
        self.assertEqual(p[1]["jersey_number"], 12)
        self.assertEqual(p[1]["jersey_number_evidence"]["frames"], [0, 5, 10])
        self.assertEqual(p[1]["jersey_number_evidence"]["current_observation_status"], "low_detection_confidence")
        self.assertIsNone(p[31]["jersey_number"])
        self.assertEqual(p[31]["jersey_number_evidence"]["observations"], 0)
        self.assertEqual(a.reader.calls, 3)

    def test_color_conflict_withholds_old_number_and_requires_three_fresh_reads(self):
        class AmbiguousOnceReader(FakeReader):
            def readtext(self, crop, **kwargs):
                findings = super().readtext(crop, **kwargs)
                if self.calls == 4:
                    findings.append(([[25, 4], [38, 4], [38, 24], [25, 24]], "13", .99))
                return findings

        a = JerseyNumberRecognizer(
            enabled=True, reader=AmbiguousOnceReader([("12", .98)] * 4 + [("13", .99)] * 3),
            min_sharpness=0)
        p = {1: players()[1]}
        for g in [0, 5, 10]:
            a.process(team_frame(), p, g)
        self.assertEqual(p[1]["jersey_number"], 12)
        a.process(team_frame(), p, 15)
        self.assertEqual(p[1]["jersey_number_status"], "conflicting_observations")
        # Current digits are unreadable. Color contradiction alone must
        # withhold the old number, including on subsequent weak detections.
        p[1].update(team_status="conflicting_colors", bbox=[0, 0, 12, 20])
        a.process(team_frame(), p, 16)
        self.assertIsNone(p[1]["jersey_number"])
        self.assertEqual(p[1]["jersey_number_status"], "association_evidence_conflict")
        self.assertEqual(p[1]["jersey_number_evidence"]["observations"], 0)
        p[1]["detection_confidence"] = .15
        a.process(team_frame(), p, 17)
        self.assertIsNone(p[1]["jersey_number"])
        self.assertEqual(a.reader.calls, 4)
        # A matching raw ID stays unchanged, but its previous OCR votes cannot
        # be reused once a new color consensus resolves the contradiction.
        p[1].update(team_status="color_consensus", bbox=players()[1]["bbox"],
                    detection_confidence=.95)
        for g in [18, 23]:
            a.process(team_frame(), p, g)
            self.assertIsNone(p[1]["jersey_number"])
        a.process(team_frame(), p, 28)
        self.assertEqual(set(p), {1})
        self.assertEqual(p[1]["jersey_number"], 13)
        self.assertEqual(p[1]["jersey_number_evidence"]["frames"], [18, 23, 28])
        self.assertEqual(a.reader.calls, 7)

    def test_ocr_requires_three_distinct_spaced_observations(self):
        a = JerseyNumberRecognizer(enabled=True, reader=FakeReader(), sample_gap=5, min_sharpness=0)
        p = {1: players()[1]}
        for g in [0, 0, 1, 5]:
            a.process(team_frame(), p, g)
            self.assertIsNone(p[1]["jersey_number"])
        a.process(team_frame(), p, 10)
        self.assertEqual(p[1]["jersey_number"], 12)
        self.assertEqual(a.reader.calls, 3)
        self.assertEqual(p[1]["jersey_number_evidence"]["frames"], [0, 5, 10])

    def test_ocr_conflicts_expire_and_do_not_merge_matching_numbers(self):
        a = JerseyNumberRecognizer(enabled=True, reader=FakeReader([("12", .98)] * 3 + [("13", .99)]),
                                   min_sharpness=0, sample_gap=5, history_frames=15)
        p = {1: players()[1]}
        for g in [0, 5, 10, 15]:
            a.process(team_frame(), p, g)
        self.assertIsNone(p[1]["jersey_number"])
        self.assertEqual(p[1]["jersey_number_status"], "conflicting_observations")
        a.process(team_frame(), {}, 31)
        self.assertEqual(a.history, {})

    def test_same_number_on_two_tracks_never_merges_them(self):
        a = JerseyNumberRecognizer(enabled=True, reader=FakeReader(), min_sharpness=0)
        p = players()
        for g in [0, 5, 10]:
            a.process(team_frame(), p, g)
        self.assertEqual(set(p), {1, 2})
        self.assertEqual([p[pid]["jersey_number"] for pid in p], [12, 12])

    def test_ocr_backend_missing_is_unknown_and_bounded(self):
        a = JerseyNumberRecognizer(enabled=True, min_sharpness=0, max_tracks=2)
        a.backend_error = "model not installed"
        p = {i: players()[1].copy() for i in range(5)}
        a.process(team_frame(), p, 0)
        self.assertTrue(all(v["jersey_number_status"] == "backend_unavailable" for v in p.values()))
        self.assertLessEqual(len(a.history), 2)

    def test_ocr_runtime_error_does_not_report_successful_consensus(self):
        a = JerseyNumberRecognizer(enabled=True, reader=FakeReader([("12", .98)] * 3), min_sharpness=0)
        p = {1: players()[1]}
        for g in [0, 5, 10]:
            a.process(team_frame(), p, g)
        self.assertEqual(p[1]["jersey_number"], 12)
        a.process(team_frame(), p, 15)
        self.assertIsNone(p[1]["jersey_number"])
        self.assertEqual(p[1]["jersey_number_status"], "ocr_error")
        self.assertEqual(a.summary["backend_status"], "backend_error")
        self.assertEqual(a.summary["error_count"], 1)
        self.assertFalse(a.summary["accuracy_validated"])

    def test_ocr_rejects_small_blurred_low_confidence_and_non_digits(self):
        a = JerseyNumberRecognizer(enabled=True, reader=FakeReader())
        p = {1: {"bbox": [0, 0, 12, 20]}}
        a.process(team_frame(), p, 0)
        self.assertEqual(p[1]["jersey_number_status"], "too_small")
        a.process(team_frame(), {2: players()[1]}, 1)
        self.assertEqual(a.reader.calls, 0)
        for text, confidence in [("12", .4), ("abc", .99), ("123", .99)]:
            b = JerseyNumberRecognizer(enabled=True, reader=FakeReader([(text, confidence)]), min_sharpness=0)
            p = {1: players()[1]}
            b.process(team_frame(), p, 0)
            self.assertIsNone(p[1]["jersey_number"])


if __name__ == "__main__":
    unittest.main()
