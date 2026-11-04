"""Per-source, per-camera-segment ground-plane calibration with explicit gates.

Fitting a homography is independent from accepting it for metric analysis.
Unreviewed landmarks, nominal pitch dimensions, poor independent checkpoints,
unmatched source identity and unsupported camera intervals never yield valid
metric coordinates. No field polygon or in-bounds projection grants acceptance.
"""
import copy
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


class CalibrationError(ValueError):
    pass


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationError(f"{label} must be a nonnegative integer")
    return value


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(value) or value <= 0:
        raise CalibrationError(f"{label} must be finite and positive")
    return float(value)


def _project(matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    if not np.isfinite(homogeneous).all() or np.any(abs(homogeneous[:, 2]) < 1e-9):
        raise CalibrationError("Projection is nonfinite or lies at the projective horizon")
    return homogeneous[:, :2] / homogeneous[:, 2:]


def _landmarks(rows, label, image_size, pitch_size):
    if not isinstance(rows, list):
        raise CalibrationError(f"{label} must be a list")
    identifiers, image, pitch = [], [], []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            raise CalibrationError(f"{label} landmark needs a nonempty id")
        try:
            a = np.asarray(row["image"], dtype=np.float64)
            b = np.asarray(row["pitch"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"Invalid {label} coordinates") from exc
        if a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise CalibrationError(f"{label} coordinates must be finite 2D points")
        if not (0 <= a[0] < image_size[0] and 0 <= a[1] < image_size[1]):
            raise CalibrationError(f"{label} image point outside original image")
        if not (0 <= b[0] <= pitch_size[0] and 0 <= b[1] <= pitch_size[1]):
            raise CalibrationError(f"{label} pitch point outside configured pitch")
        identifiers.append(row["id"])
        image.append(a)
        pitch.append(b)
    if len(set(identifiers)) != len(identifiers):
        raise CalibrationError(f"Duplicate {label} landmark id")
    return identifiers, np.asarray(image).reshape(-1, 2), np.asarray(pitch).reshape(-1, 2)


class SpatialCalibrator:
    """Fit once; return projection validity separately from candidate coordinates.

    ``source_sha256`` and ``image_size`` are the identity of the actual input,
    supplied by the caller after hashing/inspecting the source. Review acceptance
    must come from an actual recorded reviewer, never from this fitter.
    ``keyframe_only`` segments cover exactly one source frame. A ``static``
    segment is bounded and must be reviewed as one fixed camera view.
    """

    def __init__(self, configuration, source_sha256, image_size):
        self.config = copy.deepcopy(configuration)
        if self.config.get("schema_version") != 1:
            raise CalibrationError("Unsupported calibration schema_version")
        source = self.config.get("source", {})
        declared_hash = source.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
            raise CalibrationError("Calibration must identify source SHA-256")
        if len(image_size) != 2:
            raise CalibrationError("Need original image width and height")
        self.image_size = tuple(_integer(v, "image dimension") for v in image_size)
        if min(self.image_size) < 1:
            raise CalibrationError("Image dimensions must be positive")
        declared_size = (source.get("width"), source.get("height"))
        self.source_matches = source_sha256 == declared_hash
        self.dimensions_match = self.image_size == declared_size
        pitch = self.config.get("pitch", {})
        self.pitch_size = (_positive(pitch.get("length_m"), "pitch length"),
                           _positive(pitch.get("width_m"), "pitch width"))
        self.pitch_verified = pitch.get("dimensions_verified") is True
        quality = self.config.get("quality", {})
        self.max_rmse = _positive(quality.get("max_checkpoint_rmse_px", 3), "checkpoint RMSE limit")
        self.max_error = _positive(quality.get("max_checkpoint_error_px", 5), "checkpoint error limit")
        self.min_checkpoints = _integer(quality.get("min_checkpoints", 2), "minimum checkpoints")
        if self.min_checkpoints < 2:
            raise CalibrationError("At least two independent checkpoints are required")
        segments = self.config.get("segments", [])
        if not isinstance(segments, list):
            raise CalibrationError("segments must be a list")
        self.segments = []
        seen_ids = set()
        for segment in segments:
            if not isinstance(segment, dict):
                raise CalibrationError("Camera segments must be objects")
            sid = segment.get("id")
            if not isinstance(sid, str) or not sid.strip() or sid in seen_ids:
                raise CalibrationError("Every camera segment needs a distinct id")
            seen_ids.add(sid)
            start = _integer(segment.get("start_frame"), "segment start")
            end = _integer(segment.get("end_frame"), "segment end")
            keyframe = _integer(segment.get("keyframe_frame"), "keyframe")
            if not start <= keyframe <= end:
                raise CalibrationError("Keyframe must lie within the inclusive camera interval")
            model = segment.get("camera_model")
            if model not in ("keyframe_only", "static"):
                raise CalibrationError("Camera model must be keyframe_only or static")
            if model == "keyframe_only" and not start == end == keyframe:
                raise CalibrationError("A keyframe-only calibration cannot propagate to other frames")
            if any(start <= other["end_frame"] and end >= other["start_frame"] for other in self.segments):
                raise CalibrationError("Overlapping calibration intervals are ambiguous")
            report, matrix = self._fit(segment)
            self.segments.append({"start_frame": start, "end_frame": end,
                                  "id": sid, "matrix": matrix, "report": report})

    def _fit(self, segment):
        fit_ids, image, pitch = _landmarks(segment.get("fit_landmarks", []), "fit", self.image_size, self.pitch_size)
        check_ids, check_image, check_pitch = _landmarks(segment.get("validation_landmarks", []), "checkpoint", self.image_size, self.pitch_size)
        if len(image) < 4:
            raise CalibrationError("Need at least four fit correspondences")
        if set(fit_ids) & set(check_ids):
            raise CalibrationError("Fit landmarks cannot be reused as independent checkpoints")
        for points in (check_image, check_pitch):
            if len(np.unique(points, axis=0)) != len(points):
                raise CalibrationError("Independent checkpoints must have distinct positions")
        for points in (image, pitch):
            if len(np.unique(points, axis=0)) != len(points) or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
                raise CalibrationError("Fit landmarks must be distinct and noncollinear")
        for point, field in zip(check_image, check_pitch):
            if any(np.allclose(point, old, atol=1e-7) or np.allclose(field, old_field, atol=1e-7)
                   for old, old_field in zip(image, pitch)):
                raise CalibrationError("Independent checkpoints cannot reuse fitted positions")
        matrix, _ = cv2.findHomography(image, pitch, method=0)
        if matrix is None or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e14:
            raise CalibrationError("Degenerate or ill-conditioned homography")
        inverse = np.linalg.inv(matrix)
        fit_errors = np.linalg.norm(_project(inverse, pitch) - image, axis=1)
        check_errors = np.linalg.norm(_project(inverse, check_pitch) - check_image, axis=1) if len(check_image) else np.array([])
        reasons = []
        review = segment.get("review", {})
        if review.get("status") != "approved" or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or not review.get("reviewed_at"):
            reasons.append("review_pending")
        if not self.pitch_verified:
            reasons.append("pitch_dimensions_unverified")
        if not self.source_matches:
            reasons.append("source_hash_mismatch")
        if not self.dimensions_match:
            reasons.append("source_image_dimensions_mismatch")
        rmse = float(np.sqrt(np.mean(check_errors ** 2))) if len(check_errors) else None
        maximum = float(np.max(check_errors)) if len(check_errors) else None
        if len(check_errors) < self.min_checkpoints:
            reasons.append("insufficient_independent_checkpoints")
        if rmse is not None and (rmse > self.max_rmse or maximum > self.max_error):
            reasons.append("independent_checkpoint_error_exceeds_limit")
        area = abs(float(cv2.contourArea(cv2.convexHull(image.astype(np.float32)))))
        if area / (self.image_size[0] * self.image_size[1]) < 0.002:
            reasons.append("localized_fit_support")
        report = {"segment_id": segment["id"], "keyframe_frame": segment["keyframe_frame"],
                  "fit_landmarks": len(image), "independent_checkpoints": len(check_image),
                  "fit_image_rmse_px": float(np.sqrt(np.mean(fit_errors ** 2))),
                  "checkpoint_image_rmse_px": rmse, "checkpoint_image_max_error_px": maximum,
                  "checkpoint_errors_px": dict(zip(check_ids, map(float, check_errors))),
                  "homography": matrix.tolist(), "coordinate_valid": not reasons,
                  "quality_gate_reasons": reasons,
                  "interval": [segment["start_frame"], segment["end_frame"]],
                  "camera_model": segment["camera_model"],
                  "note": "Image reprojection checks do not independently verify pitch dimensions, reviewer approval or temporal camera validity."}
        return report, matrix

    def reports(self):
        return copy.deepcopy([s["report"] for s in self.segments])

    def transform(self, source_frame, image_point):
        source_frame = _integer(source_frame, "source frame")
        try:
            point = np.asarray(image_point, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise CalibrationError("Invalid image point") from exc
        if point.shape != (2,) or not np.isfinite(point).all():
            raise CalibrationError("Image point must be finite and 2D")
        unavailable = {"pitch_x": None, "pitch_y": None, "coordinate_valid": False,
                       "calibration_segment": None, "calibration_reasons": ["no_calibration_for_source_frame"]}
        segment = next((s for s in self.segments if s["start_frame"] <= source_frame <= s["end_frame"]), None)
        if segment is None:
            return unavailable
        try:
            x, y = map(float, _project(segment["matrix"], [point])[0])
        except CalibrationError:
            return {**unavailable, "calibration_segment": segment["id"],
                    "calibration_reasons": ["projection_unavailable"]}
        reasons = list(segment["report"]["quality_gate_reasons"])
        if not (0 <= point[0] < self.image_size[0] and 0 <= point[1] < self.image_size[1]):
            reasons.append("image_point_outside_source")
        if not (0 <= x <= self.pitch_size[0] and 0 <= y <= self.pitch_size[1]):
            reasons.append("projection_outside_pitch")
        return {"pitch_x": x, "pitch_y": y, "coordinate_valid": not reasons,
                "calibration_segment": segment["id"], "calibration_reasons": reasons}


def load_calibration(path, source_sha256, image_size):
    return SpatialCalibrator(json.loads(Path(path).read_text(encoding="utf-8")), source_sha256, image_size)
