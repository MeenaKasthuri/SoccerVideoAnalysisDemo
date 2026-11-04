"""Streaming evidence exports; unavailable pitch coordinates remain explicit."""
import csv
import json
import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def object_record(track_id, obj):
    return {"track_id": int(track_id), **obj,
            "coordinate_valid": bool(obj.get("coordinate_valid", False))}


def export_tracking_json(tracks, path):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write("[")
        for i in range(len(tracks["players"])):
            frame = {"frame": i,
                     "players": [object_record(k, v) for k, v in tracks["players"][i].items()],
                     "referees": [object_record(k, v) for k, v in tracks["referees"][i].items()],
                     "ball": object_record(1, tracks["ball"][i][1]) if 1 in tracks["ball"][i] else None}
            stream.write(("," if i else "") + "\n" + json.dumps(frame, default=json_value))
        stream.write("\n]")


def export_tracking_csv(tracks, path):
    fields = ["frame", "object_type", "track_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
              "pitch_x", "pitch_y", "coordinate_valid", "team", "team_color", "team_status",
              "team_confidence", "team_evidence", "role", "detection_confidence", "has_ball", "jersey_number",
              "jersey_number_status", "jersey_number_evidence"]
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for i in range(len(tracks["players"])):
            for category, name in (("players", "player"), ("referees", "referee"), ("ball", "ball")):
                for track_id, obj in tracks[category][i].items():
                    record = {k: obj.get(k) for k in fields if k in obj}
                    record.update(frame=i, object_type=name, track_id=track_id,
                                  coordinate_valid=bool(obj.get("coordinate_valid", False)),
                                  has_ball=bool(obj.get("has_ball", False)))
                    record.update(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), obj["bbox"]))
                    for key in ("team_color", "team_evidence", "jersey_number_evidence"):
                        if record.get(key) is not None:
                            record[key] = json.dumps(record[key], default=json_value)
                    writer.writerow(record)
