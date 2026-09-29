import json
import os

from exporters.json_exporter import json_converter

def export_events_json(events, output_path):
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            events,
            file,
            indent=4,
            default=json_converter
        )

    print(f"EVENT JSON WRITE COMPLETE: {output_path}")
