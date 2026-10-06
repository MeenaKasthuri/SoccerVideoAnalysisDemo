import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance
# PlayeR_Ball_Assigner
class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def get_nearby_players(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)
        nearby_players = []

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                nearby_players.append({
                    "player_id": player_id,
                    "distance": distance
                })

        return sorted(nearby_players, key=lambda candidate: candidate["distance"])

    def detect_divided_ball(self, players, ball_bbox):
        nearby_players = self.get_nearby_players(players, ball_bbox)

        if len(nearby_players) < 2:
            return None

        return {
            "player_ids": [candidate["player_id"] for candidate in nearby_players],
            "distances": [candidate["distance"] for candidate in nearby_players]
        }

    def assign_ball_to_player(self, players, ball_bbox):
        nearby_players = self.get_nearby_players(players, ball_bbox)
        return nearby_players[0]["player_id"] if nearby_players else -1
