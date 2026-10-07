import math


class TeamShapeAnalyzer:
    def calculate_frame_metrics(self, frame_players):
        teams = {}

        for player_id, player in frame_players.items():
            team_id = player.get("team")
            pitch_x = player.get("pitch_x")
            pitch_y = player.get("pitch_y")

            if team_id is None or pitch_x is None or pitch_y is None:
                continue

            teams.setdefault(team_id, []).append(
                {
                    "player_id": player_id,
                    "x": float(pitch_x),
                    "y": float(pitch_y),
                }
            )

        results = {}

        for team_id, players in teams.items():
            if len(players) < 2:
                continue

            xs = [p["x"] for p in players]
            ys = [p["y"] for p in players]

            centroid_x = sum(xs) / len(xs)
            centroid_y = sum(ys) / len(ys)

            width = max(xs) - min(xs)
            depth = max(ys) - min(ys)

            distances = [
                math.sqrt(
                    (p["x"] - centroid_x) ** 2
                    + (p["y"] - centroid_y) ** 2
                )
                for p in players
            ]

            compactness = sum(distances) / len(distances)

            results[team_id] = {
                "player_count": len(players),
                "width": round(width, 2),
                "depth": round(depth, 2),
                "compactness": round(compactness, 2),
                "centroid_x": round(centroid_x, 2),
                "centroid_y": round(centroid_y, 2),
            }

        return results

    def calculate_video_metrics(self, tracks):
        frame_results = []

        for frame_num, frame_players in enumerate(tracks["players"]):
            frame_results.append(
                {
                    "frame": frame_num,
                    "teams": self.calculate_frame_metrics(frame_players),
                }
            )

        return frame_results
