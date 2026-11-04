"""Image-space proximity assignment with an explicit experimental scale rule.

Neither rule observes physical ball control. The default preserves the legacy
70-pixel nearest-foot heuristic on valid boxes. Height normalization is opt-in
and requires reference-based calibration before it can replace the baseline.
"""
import math


class PlayerBallAssigner:
    def __init__(self, max_player_ball_distance=70, distance_mode="pixels",
                 max_player_ball_height_ratio=0.75, image_size=None,
                 abstain_ambiguous=False, require_opposing_teams_for_contest=False):
        if distance_mode not in {"pixels", "height_ratio"}:
            raise ValueError("distance_mode must be pixels or height_ratio")
        for value in (max_player_ball_distance, max_player_ball_height_ratio):
            try:
                valid = not isinstance(value, bool) and math.isfinite(value) and value > 0
            except (TypeError, ValueError, OverflowError):
                valid = False
            if not valid:
                raise ValueError("Ownership distance thresholds must be finite and positive")
        if not isinstance(abstain_ambiguous, bool) or not isinstance(require_opposing_teams_for_contest, bool):
            raise ValueError("Ownership switches must be booleans")
        if image_size is not None:
            try:
                valid_size = len(image_size) == 2 and all(
                    not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in image_size)
            except (TypeError, ValueError, OverflowError):
                valid_size = False
            if not valid_size:
                raise ValueError("image_size must contain positive width and height")
            image_size = tuple(map(float, image_size))
        self.max_player_ball_distance = max_player_ball_distance
        self.distance_mode = distance_mode
        self.max_player_ball_height_ratio = max_player_ball_height_ratio
        self.image_size = image_size
        self.abstain_ambiguous = abstain_ambiguous
        self.require_opposing_teams_for_contest = require_opposing_teams_for_contest

    @classmethod
    def from_configuration(cls, configuration=None, image_size=None):
        config = dict(configuration or {})
        allowed = {"distance_mode", "max_player_ball_distance", "max_player_ball_height_ratio",
                   "abstain_ambiguous", "require_opposing_teams_for_contest"}
        if set(config) - allowed:
            raise ValueError("Unknown ball_ownership configuration: " + ", ".join(sorted(set(config) - allowed)))
        return cls(image_size=image_size, **config)

    @property
    def configuration(self):
        return {"distance_mode": self.distance_mode,
                "max_player_ball_distance": self.max_player_ball_distance,
                "max_player_ball_height_ratio": self.max_player_ball_height_ratio,
                "abstain_ambiguous": self.abstain_ambiguous,
                "require_opposing_teams_for_contest": self.require_opposing_teams_for_contest,
                "normalized_rule_experimental": self.distance_mode == "height_ratio",
                "physical_control_verified": False}

    @staticmethod
    def _bbox(value):
        try:
            if len(value) != 4:
                return None
            box = tuple(map(float, value))
            if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                return None
            return box
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _team(value):
        if value is None or str(value).strip().lower() in {"0", "-1", "", "unknown", "unassigned", "none", "nan", "inf", "-inf"}:
            return None
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return str(value)

    def get_nearby_players(self, players, ball_bbox):
        ball = self._bbox(ball_bbox)
        if ball is None:
            return []
        bx, by = (ball[0] + ball[2]) / 2, (ball[1] + ball[3]) / 2
        nearby = []
        for player_id, player in players.items():
            box = self._bbox(player.get("bbox"))
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if self.distance_mode == "height_ratio":
                # Height is unreliable for explicitly flagged occlusion/poses or
                # image-edge truncation. Missing flags do not prove reliability.
                if player.get("scale_reliable") is False or player.get("bbox_truncated") is True:
                    continue
                if self.image_size is not None:
                    width, height = self.image_size
                    if x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height:
                        continue
            distance = min(math.hypot(x1 - bx, y2 - by), math.hypot(x2 - bx, y2 - by))
            score = distance if self.distance_mode == "pixels" else distance / (y2 - y1)
            threshold = (self.max_player_ball_distance if self.distance_mode == "pixels"
                         else self.max_player_ball_height_ratio)
            if score < threshold:
                candidate = {"player_id": player_id, "distance": distance}
                if self.distance_mode == "height_ratio":
                    candidate["normalized_distance"] = score
                nearby.append(candidate)
        return sorted(nearby, key=lambda c: c.get("normalized_distance", c["distance"]))

    def ownership_observation(self, players, ball_bbox):
        nearby = self.get_nearby_players(players, ball_bbox)
        ambiguous = len(nearby) > 1
        assigned = bool(nearby) and not (ambiguous and self.abstain_ambiguous)
        return {"player_id": nearby[0]["player_id"] if assigned else None,
                "status": ("ambiguous_proximity" if ambiguous and self.abstain_ambiguous
                           else "proximity_assigned" if assigned else "unassigned"),
                "candidate_player_ids": [c["player_id"] for c in nearby],
                "distance_mode": self.distance_mode,
                "physical_control_verified": False}

    def detect_divided_ball(self, players, ball_bbox):
        nearby = self.get_nearby_players(players, ball_bbox)
        if len(nearby) < 2:
            return None
        if self.require_opposing_teams_for_contest:
            teams = {self._team(players[c["player_id"]].get("team")) for c in nearby}
            teams.discard(None)
            if len(teams) < 2:
                return None
        return {"player_ids": [c["player_id"] for c in nearby],
                "distances": [c["distance"] for c in nearby]}

    def assign_ball_to_player(self, players, ball_bbox):
        player_id = self.ownership_observation(players, ball_bbox)["player_id"]
        return player_id if player_id is not None else -1
