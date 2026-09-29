import csv
import os


def export_tracking_csv(tracks, output_path):
    # Create output directory if it does not exist
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "frame",
        "object_type",
        "track_id",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "pitch_x",
        "pitch_y",
        "team",
        "team_color",
        "has_ball"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for frame_num in range(len(tracks["players"])):

            # Players
            for track_id, player in tracks["players"][frame_num].items():

                bbox = player.get("bbox", [None, None, None, None])

                team_color = player.get("team_color")

                if team_color is not None:
                    try:
                        team_color = team_color.tolist()
                    except AttributeError:
                        pass

                    team_color = str(team_color)

                writer.writerow({
                    "frame": frame_num,
                    "object_type": "player",
                    "track_id": track_id,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "pitch_x": player.get("pitch_x"),
                    "pitch_y": player.get("pitch_y"),
                    "team": player.get("team"),
                    "team_color": team_color,
                    "has_ball": player.get("has_ball", False)
                })

            # Referees
            for track_id, referee in tracks["referees"][frame_num].items():

                bbox = referee.get("bbox", [None, None, None, None])

                writer.writerow({
                    "frame": frame_num,
                    "object_type": "referee",
                    "track_id": track_id,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "pitch_x": referee.get("pitch_x"),
                    "pitch_y": referee.get("pitch_y"),
                    "team": None,
                    "team_color": None,
                    "has_ball": False
                })

            # Ball
            if 1 in tracks["ball"][frame_num]:

                ball = tracks["ball"][frame_num][1]

                bbox = ball.get("bbox", [None, None, None, None])

                writer.writerow({
                    "frame": frame_num,
                    "object_type": "ball",
                    "track_id": 1,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y2": bbox[3],
                    "pitch_x": ball.get("pitch_x"),
                    "pitch_y": ball.get("pitch_y"),
                    "team": None,
                    "team_color": None,
                    "has_ball": False
                })

    print(f"CSV WRITE COMPLETE: {output_path}")