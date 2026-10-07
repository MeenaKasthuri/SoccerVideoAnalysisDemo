import csv
import matplotlib.pyplot as plt

CSV_PATH = "docs/analytics_validation/team_shape_metrics.csv"
OUTPUT_PATH = "docs/analytics_validation/team_width_over_time.png"

team_1_frames = []
team_1_width = []

team_2_frames = []
team_2_width = []

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        frame = int(row["frame"])
        team_id = int(row["team_id"])
        width = float(row["width"])

        if team_id == 1:
            team_1_frames.append(frame)
            team_1_width.append(width)

        elif team_id == 2:
            team_2_frames.append(frame)
            team_2_width.append(width)

plt.figure(figsize=(10, 5))

plt.plot(
    team_1_frames,
    team_1_width,
    label="Team 1"
)

plt.plot(
    team_2_frames,
    team_2_width,
    label="Team 2"
)

plt.xlabel("Frame")
plt.ylabel("Team X-axis span")
plt.title("Team Spatial Width Over Time")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=150
)

print("Saved:", OUTPUT_PATH)
