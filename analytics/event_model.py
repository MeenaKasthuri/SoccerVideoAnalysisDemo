"""Build observed football episodes from frame-level tracking observations.

``build_events`` consumes the complete, ordered frame sequence, including tracks
assembled across decoding chunks. Separate calls are independent and must not
be used to count events in separate chunks of a match.
"""

import math


class EventModel:
    def __init__(self, fps, stability_frames=3):
        if fps is None or not math.isfinite(fps) or fps <= 0:
            raise ValueError("FPS must be finite and greater than 0")
        if (not isinstance(stability_frames, int)
                or isinstance(stability_frames, bool) or stability_frames < 1):
            raise ValueError("stability_frames must be a positive integer")
        self.fps = fps
        self.stability_frames = stability_frames

    @staticmethod
    def _known_team(team):
        """Zero is the pipeline's unassigned team, never an opposing team."""
        if team is None or team in (0, -1, "", "unknown", "unassigned"):
            return None
        if isinstance(team, float) and not math.isfinite(team):
            return None
        return team

    @staticmethod
    def _contest_ids(observation):
        if not isinstance(observation, dict):
            return []
        ids = observation.get("player_ids", [])
        if not isinstance(ids, (list, tuple)):
            return []
        return list(dict.fromkeys(ids))

    def build_events(self, tracks):
        """Return one record per observed episode or stabilized owner transition.

        A divided-ball episode is a contiguous run of contested observations.
        Participant changes within that run do not create another event. Bounds
        are inclusive; duration is observed-frame count divided by FPS. Missing
        observations end the run. Boundary flags identify episodes whose actual
        beginning/end may lie outside this input.

        Exclusive ownership is unknown during a contest, with multiple owners,
        or with no owner. Gaps reset candidate stabilization but retain the last
        accepted owner/team. Later transitions compare observed ownership, with
        the gap count exposing intervening uncertainty. Touch is an ownership
        proxy, not proof of a completed pass. Only transitions between known,
        different teams are team possession_change events.
        """
        events = []
        previous_owner = None
        previous_team = None
        candidate_owner = None
        candidate_count = 0
        gap_frames = 0
        episode = None
        frames = tracks.get("players", [])
        divided_frames = tracks.get("divided_ball", [])

        def emit(event):
            event["event_id"] = f"EVT-{len(events) + 1:06d}"
            events.append(event)

        def close_episode(end_frame, right_censored=False):
            nonlocal episode
            episode["end_frame"] = end_frame
            episode["end_timestamp"] = round(end_frame / self.fps, 3)
            episode["duration_frames"] = end_frame - episode["start_frame"] + 1
            episode["duration_seconds"] = episode["duration_frames"] / self.fps
            episode["details"]["right_censored"] = right_censored
            emit(episode)
            episode = None

        for frame_num, frame_players in enumerate(frames):
            divided = divided_frames[frame_num] if frame_num < len(divided_frames) else None
            contest_ids = self._contest_ids(divided)
            contested = len(contest_ids) >= 2

            # A corrected label on the accepted identity is not itself an event.
            if previous_owner in frame_players:
                team = self._known_team(frame_players[previous_owner].get("team"))
                if team is not None:
                    previous_team = team

            if contested:
                if episode is None:
                    first_player = frame_players.get(contest_ids[0], {})
                    episode = {
                        "event_type": "divided_ball",
                        "frame": frame_num,
                        "start_frame": frame_num,
                        "timestamp": round(frame_num / self.fps, 3),
                        "player_id": contest_ids[0],
                        "related_player_id": contest_ids[1],
                        "involved_player_ids": [],
                        "team_id": first_player.get("team"),
                        "team_ids": [],
                        "pitch_x": divided.get("pitch_x"),
                        "pitch_y": divided.get("pitch_y"),
                        "coordinate_valid": divided.get("coordinate_valid", True),
                        "details": {
                            "distances": list(divided.get("distances", [])),
                            "left_censored": frame_num == 0,
                        },
                    }
                for player_id in contest_ids:
                    if player_id not in episode["involved_player_ids"]:
                        episode["involved_player_ids"].append(player_id)
                for team in divided.get("teams", []):
                    if team not in episode["team_ids"]:
                        episode["team_ids"].append(team)
                candidate_owner = None
                candidate_count = 0
                gap_frames += 1
                continue

            if episode is not None:
                close_episode(frame_num - 1)

            owners = [(track_id, player) for track_id, player in frame_players.items()
                      if player.get("has_ball", False)]
            current_owner, current_player = owners[0] if len(owners) == 1 else (None, None)
            if current_owner is None:
                candidate_owner = None
                candidate_count = 0
                gap_frames += 1
                continue
            if current_owner == previous_owner:
                candidate_owner = None
                candidate_count = 0
                gap_frames = 0
                continue

            if previous_owner is not None:
                if current_owner == candidate_owner:
                    candidate_count += 1
                else:
                    candidate_owner = current_owner
                    candidate_count = 1
                if candidate_count < self.stability_frames:
                    continue

            current_team = self._known_team(current_player.get("team"))
            transition = "initial"
            if previous_owner is not None:
                transition = ("unknown_team" if previous_team is None or current_team is None
                              else "same_team" if previous_team == current_team else "opposing_team")
            common = {
                "frame": frame_num,
                "timestamp": round(frame_num / self.fps, 3),
                "player_id": current_owner,
                "team_id": current_player.get("team"),
                "pitch_x": current_player.get("pitch_x"),
                "pitch_y": current_player.get("pitch_y"),
                "coordinate_valid": current_player.get("coordinate_valid", True),
                "related_player_id": previous_owner,
            }
            emit({**common, "event_type": "touch", "details": {
                "has_ball": True,
                "ownership_transition": transition,
                "unobserved_frames": gap_frames,
            }})
            if transition == "opposing_team":
                emit({**common, "event_type": "possession_change", "details": {
                    "previous_player_id": previous_owner,
                    "new_player_id": current_owner,
                    "previous_team_id": previous_team,
                    "new_team_id": current_team,
                    "unobserved_frames": gap_frames,
                }})
            previous_owner = current_owner
            previous_team = current_team
            candidate_owner = None
            candidate_count = 0
            gap_frames = 0

        if episode is not None:
            close_episode(len(frames) - 1, right_censored=True)
        return events
