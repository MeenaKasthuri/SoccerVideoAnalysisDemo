"""Evaluate event records against explicitly labeled, complete time windows.

Reference review status is carried into the result. Incomplete or absent labels
produce unavailable metrics, never a claim that zero mistakes were measured.
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def _frame(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("Event frames must be nonnegative integers")
    return int(value)


def event_bounds(event):
    if event.get("event_type") == "divided_ball":
        if "end_frame" not in event:
            raise ValueError("Divided-ball evaluation requires episode end_frame")
        start = _frame(event.get("start_frame", event.get("frame")))
        end = _frame(event["end_frame"])
        if end < start:
            raise ValueError("Event end precedes start")
        return start, end
    frame = _frame(event.get("frame"))
    return frame, frame


def interval_iou(a, b):
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = max(a[1], b[1]) - min(a[0], b[0]) + 1
    return intersection / union


def _quality(predicted, reference, tolerance_frames, minimum_interval_iou):
    if predicted["event_type"] != reference["event_type"]:
        return None
    a, b = event_bounds(predicted), event_bounds(reference)
    if predicted["event_type"] == "divided_ball":
        quality = interval_iou(a, b)
        return quality if quality >= minimum_interval_iou else None
    delta = abs(a[0] - b[0])
    if delta > tolerance_frames:
        return None
    return 1 - delta / (tolerance_frames + 1)


def evaluate_events(predictions, reference, *, source_sha256=None,
                    tolerance_seconds=0.2, minimum_interval_iou=0.5):
    """Return per-type interval/point correspondence, not inferred football truth.

    Required reference fields: fps, start_frame, end_frame, events, and an
    explicit complete_event_types list. Point events match within the timing
    tolerance; divided_ball episodes match by inclusive-frame IoU. Predictions
    crossing the labeled window are ignored and counted as boundary exclusions.
    Uncertain reference events are ignore regions after eligible matches.
    """
    fps = reference.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Reference FPS must be finite and positive")
    if not math.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("Timing tolerance must be nonnegative")
    if not 0 < minimum_interval_iou <= 1:
        raise ValueError("minimum_interval_iou must be greater than zero and at most one")
    if source_sha256 is not None and reference.get("source_sha256") != source_sha256:
        raise ValueError("Reference source video hash does not match predictions")
    complete = reference.get("complete_event_types", [])
    if (not isinstance(complete, list) or any(not isinstance(k, str) or not k for k in complete)
            or len(complete) != len(set(complete))):
        raise ValueError("complete_event_types must be a list of distinct event types")
    base = {"human_reviewed": reference.get("human_reviewed") is True,
            "status": "human_reviewed_reference" if reference.get("human_reviewed") is True else "provisional_reference_review_pending",
            "matching": "one-to-one maximum valid match count, then overlap/timing quality",
            "point_tolerance_seconds": tolerance_seconds,
            "interval_iou_threshold": minimum_interval_iou,
            "per_type": [], "accuracy_validated": False}
    if not complete:
        base.update(status="unavailable_incomplete_reference", reason="No event types have explicitly complete reference coverage")
        return base
    start, end = _frame(reference.get("start_frame")), _frame(reference.get("end_frame"))
    if end < start:
        raise ValueError("Reference coverage ends before it starts")
    truth = reference.get("events", [])
    if not isinstance(truth, list) or not isinstance(predictions, list):
        raise ValueError("Events must be lists")
    for event in predictions + truth:
        if not isinstance(event, dict) or not isinstance(event.get("event_type"), str):
            raise ValueError("Every event requires event_type")
        event_bounds(event)
    for event in truth:
        a, b = event_bounds(event)
        if a < start or b > end:
            raise ValueError("Reference events must lie inside labeled coverage")
    tolerance_frames = tolerance_seconds * fps
    for kind in complete:
        targets = [e for e in truth if e["event_type"] == kind and not e.get("uncertain", False)]
        ignored = [e for e in truth if e["event_type"] == kind and e.get("uncertain", False)]
        rows = [e for e in predictions if e["event_type"] == kind]
        eligible = [e for e in rows if start <= event_bounds(e)[0] <= event_bounds(e)[1] <= end]
        matches = []
        if eligible and targets:
            quality = np.full((len(eligible), len(targets)), -1.0)
            for i, p in enumerate(eligible):
                for j, t in enumerate(targets):
                    q = _quality(p, t, tolerance_frames, minimum_interval_iou)
                    if q is not None:
                        quality[i, j] = q
            bonus = min(len(eligible), len(targets)) + 1
            weights = np.where(quality >= 0, bonus + quality, 0)
            p_indices, t_indices = linear_sum_assignment(weights, maximize=True)
            matches = [{"prediction_index": int(i), "reference_index": int(j), "quality": float(quality[i, j])}
                       for i, j in zip(p_indices, t_indices) if quality[i, j] >= 0]
        matched = {m["prediction_index"] for m in matches}
        ignored_predictions = []
        for i, p in enumerate(eligible):
            if i in matched:
                continue
            bounds = event_bounds(p)
            if any(interval_iou(bounds, event_bounds(t)) > 0 for t in ignored):
                ignored_predictions.append(i)
        tp, fn = len(matches), len(targets) - len(matches)
        fp = len(eligible) - tp - len(ignored_predictions)
        base["per_type"].append({"event_type": kind, "tp": tp, "fp": fp, "fn": fn,
                                 "precision": tp / (tp + fp) if tp + fp else None,
                                 "recall": tp / (tp + fn) if tp + fn else None,
                                 "matches": matches,
                                 "uncertain_reference_events": len(ignored),
                                 "ignored_predictions": len(ignored_predictions),
                                 "outside_or_boundary_predictions": len(rows) - len(eligible)})
    return base
