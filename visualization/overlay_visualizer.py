import numpy as np
import cv2
from utils import draw_clamped_label
class OverlayVisualizer:
    def __init__(self):
        pass

    def draw_team_hulls(self, frame, team_hulls):
        overlay = frame.copy()

        for team, data in team_hulls.items():
            hull = data["hull"].astype(
                np.int32
            )

            color = tuple(
                int(c)
                for c in data["team_color"]
            )

            print("Hull:")
            print(hull)
            print("Shape:", hull.shape)
            print("Dtype:", hull.dtype)

            cv2.fillPoly(
                overlay,
                [hull],
                color
            )

            cv2.polylines(
                frame,
                [hull],
                True,
                color,
                3
            )

            center_x = int(np.mean(hull[:, 0, 0]))
            center_y = int(np.mean(hull[:, 0, 1]))

            cv2.putText(
                frame,
                f"Team {team}",
                (center_x - 25, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.addWeighted(
            overlay,
            0.25,
            frame,
            0.75,
            0,
            frame
        )

        return frame

    def draw_team_heatmaps(
            self,
            frame,
            team_heatmaps
    ):

        overlay = frame.copy()

        # Fixed colors for readability
        team_colors = {

            1: (0, 0, 255),  # Red
            2: (255, 0, 0)  # Blue

        }

        for team, data in team_heatmaps.items():

            heatmap = data["heatmap"]

            color = team_colors.get(
                team,
                (0, 255, 0)
            )

            heatmap = cv2.resize(

                heatmap,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_CUBIC

            )

            heatmap = np.clip(
                heatmap,
                0,
                1
            )

            colored_heatmap = np.zeros_like(
                frame,
                dtype=np.uint8
            )

            for i in range(3):
                colored_heatmap[:, :, i] = (
                        heatmap * color[i]
                ).astype(np.uint8)

            cv2.addWeighted(

                colored_heatmap,
                0.45,

                overlay,
                0.55,

                0,

                overlay

            )

        return overlay

    def draw_players(self, frame, frame_players):

        for track_id, player in frame_players.items():
            bbox = player["bbox"]

            team = player.get("team")

            color = (
                (0, 0, 255)
                if team == 1
                else
                (255, 0, 0)
            )

            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            draw_clamped_label(
                frame,
                track_id,
                (x1, y1 - 8),
                color,
                font_scale=0.5,
                thickness=2,
                background=False
            )

        return frame

    def draw_referees(self, frame, referees):
        for track_id, referee in referees.items():
            bbox = referee["bbox"]

            color = (0,255,255)

            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )


