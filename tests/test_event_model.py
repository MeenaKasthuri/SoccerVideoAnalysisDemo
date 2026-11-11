"""Event semantics from explicit frame observations; no detector is involved."""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics.event_model import EventModel


def owner(player_id, team, **extra):
    return {player_id: {"has_ball": True, "team": team, **extra}}


def contest(ids=(1, 2), teams=(1, 2)):
    return {"player_ids": list(ids), "teams": list(teams), "distances": [10, 20],
            "coordinate_valid": False}


def of_type(events, kind):
    return [event for event in events if event["event_type"] == kind]


class EventModelTests(unittest.TestCase):
    def test_continuous_contest_is_one_bounded_episode(self):
        observations = [None, contest(), contest(), contest(), None]
        events = EventModel(25).build_events({"players": [{}] * 5, "divided_ball": observations})
        episode, = of_type(events, "divided_ball")
        self.assertEqual((episode["frame"], episode["start_frame"], episode["end_frame"]), (1, 1, 3))
        self.assertEqual(episode["duration_frames"], 3)
        self.assertAlmostEqual(episode["duration_seconds"], 3 / 25)
        self.assertEqual((episode["timestamp"], episode["end_timestamp"]), (0.04, 0.12))
        self.assertFalse(episode["details"]["left_censored"])
        self.assertFalse(episode["details"]["right_censored"])
        self.assertFalse(episode["coordinate_valid"])

    def test_participant_order_and_identity_changes_do_not_split_episode(self):
        observations = [contest(), contest((2, 1)), contest((2, 3))]
        episode, = EventModel(30).build_events({"players": [{}] * 3, "divided_ball": observations})
        self.assertEqual(episode["involved_player_ids"], [1, 2, 3])
        self.assertEqual(episode["duration_frames"], 3)
        self.assertTrue(episode["details"]["left_censored"])
        self.assertTrue(episode["details"]["right_censored"])

    def test_missing_contest_observation_ends_episode(self):
        events = EventModel(30).build_events({"players": [{}] * 5,
                                             "divided_ball": [contest(), None, contest()]})
        episodes = of_type(events, "divided_ball")
        self.assertEqual([(e["start_frame"], e["end_frame"]) for e in episodes], [(0, 0), (2, 2)])
        self.assertFalse(episodes[-1]["details"]["right_censored"])

    def test_same_team_transfer_is_touch_without_team_possession_change(self):
        frames = [owner(1, 1)] + [owner(2, 1)] * 3
        events = EventModel(25).build_events({"players": frames})
        self.assertEqual(len(of_type(events, "touch")), 2)
        self.assertEqual(of_type(events, "possession_change"), [])
        self.assertEqual(events[-1]["details"]["ownership_transition"], "same_team")
        self.assertEqual(events[-1]["related_player_id"], 1)
        self.assertEqual(events[-1]["frame"], 3)

    def test_opponent_transfer_has_one_team_change_after_stability(self):
        frames = [owner(1, 1)] + [owner(2, 2, pitch_x=i) for i in range(1, 4)]
        frames += [owner(2, 2)] * 4
        change, = of_type(EventModel(25).build_events({"players": frames}), "possession_change")
        self.assertEqual(change["frame"], 3)
        self.assertEqual(change["pitch_x"], 3)  # confirmation-frame data, not stale candidate data
        self.assertEqual(change["details"]["previous_team_id"], 1)
        self.assertEqual(change["details"]["new_team_id"], 2)

    def test_unknown_teams_do_not_create_team_changes(self):
        for team_a, team_b in [(None, 1), (1, None), (0, 1), (1, 0), (-1, 2), (1, float("nan"))]:
            with self.subTest(team_a=team_a, team_b=team_b):
                frames = [owner(1, team_a)] + [owner(2, team_b)] * 3
                events = EventModel(25).build_events({"players": frames})
                self.assertEqual(of_type(events, "possession_change"), [])
                self.assertEqual(events[-1]["details"]["ownership_transition"], "unknown_team")

    def test_flicker_and_gap_reset_stability(self):
        frames = [owner(1, 1), owner(2, 2), owner(2, 2), {}, owner(2, 2), owner(2, 2), owner(2, 2)]
        events = EventModel(25).build_events({"players": frames})
        change, = of_type(events, "possession_change")
        self.assertEqual(change["frame"], 6)
        self.assertEqual(change["details"]["unobserved_frames"], 1)

    def test_return_to_accepted_owner_cancels_candidate(self):
        frames = [owner(1, 1), owner(2, 2), owner(2, 2), owner(1, 1), owner(2, 2), owner(2, 2)]
        events = EventModel(25).build_events({"players": frames})
        self.assertEqual(len(events), 1)

    def test_contested_nearest_owner_flags_do_not_establish_exclusive_possession(self):
        frames = [owner(1, 1)] + [owner(2, 2)] * 6
        divided = [None] + [contest()] * 3 + [None] * 3
        events = EventModel(25).build_events({"players": frames, "divided_ball": divided})
        change, = of_type(events, "possession_change")
        self.assertEqual(change["frame"], 6)
        self.assertEqual(change["details"]["unobserved_frames"], 3)
        self.assertEqual(len(of_type(events, "divided_ball")), 1)

    def test_ambiguous_multiple_owner_flags_are_not_order_dependent(self):
        frames = [{**owner(1, 1), **owner(2, 2)}] * 4
        self.assertEqual(EventModel(25).build_events({"players": frames}), [])

    def test_same_identity_team_correction_does_not_emit_transition(self):
        frames = [owner(1, 0), owner(1, 1)] + [owner(2, 1)] * 3
        events = EventModel(25).build_events({"players": frames})
        self.assertEqual(of_type(events, "possession_change"), [])
        self.assertEqual(events[-1]["details"]["ownership_transition"], "same_team")

    def test_chunk_assembled_frames_and_repeated_calls_are_deterministic(self):
        chunk_a = {"players": [owner(1, 1), owner(1, 1)], "divided_ball": [None, contest()]}
        chunk_b = {"players": [owner(2, 2)] * 4, "divided_ball": [contest(), None, None, None]}
        tracks = {key: chunk_a[key] + chunk_b[key] for key in chunk_a}
        snapshot = copy.deepcopy(tracks)
        model = EventModel(25)
        first = model.build_events(tracks)
        self.assertEqual(first, model.build_events(tracks))
        self.assertEqual(tracks, snapshot)
        episode, = of_type(first, "divided_ball")
        self.assertEqual((episode["start_frame"], episode["end_frame"]), (1, 2))
        self.assertEqual([e["event_id"] for e in first], [f"EVT-{i:06d}" for i in range(1, len(first) + 1)])
        self.assertEqual([e["frame"] for e in first], sorted(e["frame"] for e in first))

    def test_empty_and_invalid_contest_observations(self):
        self.assertEqual(EventModel(25).build_events({"players": []}), [])
        tracks = {"players": [{}] * 3, "divided_ball": [{}, contest((1, 1)), {"player_ids": None}]}
        self.assertEqual(EventModel(25).build_events(tracks), [])

    def test_configuration_validation(self):
        for fps in (None, 0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                EventModel(fps)
        for stability in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                EventModel(25, stability)


if __name__ == "__main__":
    unittest.main()
