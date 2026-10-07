import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analytics.team_shape_analyzer import TeamShapeAnalyzer

INPUT_JSON = "JSON_data/tracking_output.json"
OUTPUT_JSON = "docs/analytics_validation/team_shape_metrics.json"
OUTPUT_CSV = "docs/analytics_validation/team_shape_metrics.csv"

with open(INPUT_JSON, encoding="utf-8") as f:
    exported_frames = json.load(f)

tracks = {
    "players": [
        {
            player["track_id"]: player
            for player in frame["players"]
        }
        for frame in exported_frames
    ]
}

analyzer = TeamShapeAnalyzer()
results = analyzer.calculate_video_metrics(tracks)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "frame",
        "team_id",
        "player_count",
        "width",
        "depth",
        "compactness",
        "centroid_x",
        "centroid_y",
    ])

    for frame in results:
        for team_id, metrics in frame["teams"].items():
            writer.writerow([
                frame["frame"],
                team_id,
                metrics["player_count"],
                metrics["width"],
                metrics["depth"],
                metrics["compactness"],
                metrics["centroid_x"],
                metrics["centroid_y"],
            ])

totals = defaultdict(
    lambda: {
        "width": [],
        "depth": [],
        "compactness": [],
    }
)

for frame in results:
    for team_id, metrics in frame["teams"].items():
        totals[team_id]["width"].append(metrics["width"])
        totals[team_id]["depth"].append(metrics["depth"])
        totals[team_id]["compactness"].append(metrics["compactness"])

print("TEAM SHAPE VALIDATION")
print("=" * 60)
print("Frames analyzed:", len(results))

for team_id, values in sorted(totals.items()):
    print(f"\nTeam {team_id}")

    print(
        "Average width:",
        round(sum(values["width"]) / len(values["width"]), 2)
    )

    print(
        "Average depth:",
        round(sum(values["depth"]) / len(values["depth"]), 2)
    )

    print(
        "Average compactness:",
        round(
            sum(values["compactness"]) / len(values["compactness"]),
            2
        )
    )

print("\nJSON:", OUTPUT_JSON)
print("CSV :", OUTPUT_CSV)
