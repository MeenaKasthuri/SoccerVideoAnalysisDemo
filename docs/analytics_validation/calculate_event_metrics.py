import csv
from collections import defaultdict

CSV_PATH = "docs/analytics_validation/manual_events_100frames.csv"

stats = defaultdict(
    lambda: {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "unclear": 0,
        "system_events": 0,
    }
)

with open(CSV_PATH, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        event_type = row["event_type"].strip()
        system_detected = row["system_detected"].strip().lower()
        manual_valid = row["manual_valid"].strip().lower()

        if system_detected == "true":
            stats[event_type]["system_events"] += 1

        if manual_valid == "unclear":
            stats[event_type]["unclear"] += 1
            continue

        if system_detected == "true" and manual_valid == "true":
            stats[event_type]["tp"] += 1

        elif system_detected == "true" and manual_valid == "false":
            stats[event_type]["fp"] += 1

        elif system_detected == "false" and manual_valid == "true":
            stats[event_type]["fn"] += 1


def safe_divide(a, b):
    return a / b if b else None


print("\nEVENT VALIDATION RESULTS")
print("=" * 70)

for event_type, values in stats.items():
    tp = values["tp"]
    fp = values["fp"]
    fn = values["fn"]
    unclear = values["unclear"]

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    if precision is not None and recall is not None and precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    print(f"\nEvent type: {event_type}")
    print(f"System detections : {values['system_events']}")
    print(f"True positives    : {tp}")
    print(f"False positives   : {fp}")
    print(f"False negatives   : {fn}")
    print(f"Unclear           : {unclear}")

    if precision is not None:
        print(f"Precision         : {precision:.2%}")
    else:
        print("Precision         : N/A")

    if recall is not None:
        print(f"Recall            : {recall:.2%}")
    else:
        print("Recall            : N/A")

    if f1 is not None:
        print(f"F1 score          : {f1:.2%}")
    else:
        print("F1 score          : N/A")
