"""Conservative, explicitly calibrated field filtering and duplicate suppression.

No polygon means no exclusions. Polygons use normalized image coordinates and
global source-frame indices. Optical flow carries a calibration only while
background support remains reliable; uncertainty disables exclusions until the
next keyframe. No grass-color guess is used as a field boundary.
"""

import cv2
import numpy as np


class FieldFilter:
    def __init__(self, config=None):
        self.config = dict(config or {})
        self.keyframes = {int(k): v for k, v in self.config.get("field_keyframes", {}).items()}
        self.max_frames = int(self.config.get("max_propagation_frames", 120))
        self.margin = max(0.0, float(self.config.get("boundary_margin_px", 15)))
        self.box_margin = max(0.0, float(self.config.get("boundary_margin_box_ratio", 0.15)))
        self.min_features = int(self.config.get("field_min_features", 12))
        self.min_inlier_ratio = float(self.config.get("field_min_inlier_ratio", 0.7))
        self._polygon = None
        self._gray = None
        self._boxes = []
        self._frame_index = None
        self._seed_index = None
        self._shape = None
        self._status = "uncalibrated"
        self._reason = "no_manual_field_keyframe"
        self._confidence = 0.0
        self._support = 0

    @property
    def state(self):
        return {
            "status": self._status,
            "reason": self._reason,
            "confidence": float(self._confidence),
            "frame_index": self._frame_index,
            "keyframe_index": self._seed_index,
            "supporting_features": self._support,
            "polygon": self._polygon.tolist() if self._polygon is not None else None,
            "filtering_active": self._polygon is not None,
        }

    def _invalidate(self, reason):
        self._polygon = None
        self._status = "uncertain"
        self._reason = reason
        self._confidence = 0.0
        self._support = 0

    @staticmethod
    def _background_mask(shape, boxes):
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        for box in boxes:
            if len(box) != 4 or not np.isfinite(box).all():
                continue
            x1, y1, x2, y2 = box
            pad = max(5, int(0.1 * max(y2 - y1, x2 - x1)))
            left, top = max(0, int(x1) - pad), max(0, int(y1) - pad)
            right, bottom = min(w, int(x2) + pad), min(h, int(y2) + pad)
            if right > left and bottom > top:
                mask[top:bottom, left:right] = 0
        return mask

    def _seed(self, value, gray, index):
        try:
            polygon = np.asarray(value, dtype=np.float32)
        except (ValueError, TypeError):
            self._invalidate("invalid_manual_keyframe")
            return
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all()
                or np.any(polygon < 0) or np.any(polygon > 1)):
            self._invalidate("invalid_manual_keyframe")
            return
        h, w = gray.shape
        polygon = polygon * np.array([w - 1, h - 1], dtype=np.float32)
        # Require an ordered, convex polygon, so crossed vertices never silently
        # produce a wrong field mask. Visible pitch intersections are convex.
        if not cv2.isContourConvex(polygon) or abs(cv2.contourArea(polygon)) < 100:
            self._invalidate("invalid_manual_keyframe")
            return
        self._polygon = polygon
        self._seed_index = index
        self._status = "calibrated"
        self._reason = "manual_keyframe"
        self._confidence = 1.0
        self._support = 0

    def _propagate(self, gray, boxes):
        h, w = gray.shape
        mask = self._background_mask(gray.shape, self._boxes)
        points = cv2.goodFeaturesToTrack(self._gray, maxCorners=400,
                                        qualityLevel=0.02, minDistance=7, mask=mask)
        if points is None or len(points) < self.min_features:
            self._invalidate("insufficient_background_features")
            return
        moved, ok, _ = cv2.calcOpticalFlowPyrLK(
            self._gray, gray, points, None, winSize=(21, 21), maxLevel=3)
        if moved is None or ok is None:
            self._invalidate("optical_flow_failed")
            return
        back, back_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self._gray, moved, None, winSize=(21, 21), maxLevel=3)
        if back is None or back_ok is None:
            self._invalidate("optical_flow_failed")
            return
        src, dst = points.reshape(-1, 2), moved.reshape(-1, 2)
        valid = ((ok.ravel() == 1) & (back_ok.ravel() == 1)
                 & (np.linalg.norm(src - back.reshape(-1, 2), axis=1) <= 1.5)
                 & np.isfinite(dst).all(axis=1))
        valid &= ((dst[:, 0] >= 0) & (dst[:, 0] < w)
                  & (dst[:, 1] >= 0) & (dst[:, 1] < h))
        current_mask = self._background_mask(gray.shape, boxes)
        in_bounds = np.flatnonzero(valid)
        if len(in_bounds):
            xy = dst[in_bounds].astype(int)
            valid[in_bounds] &= current_mask[xy[:, 1], xy[:, 0]] > 0
        src, dst = src[valid], dst[valid]
        if len(src) < self.min_features or len(src) < 0.35 * len(points):
            self._invalidate("insufficient_flow_support_or_scene_cut")
            return
        affine, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
            maxIters=1000, confidence=0.995, refineIters=10)
        if affine is None or inliers is None or not np.isfinite(affine).all():
            self._invalidate("camera_transform_failed")
            return
        inliers = inliers.ravel().astype(bool)
        ratio = float(np.mean(inliers))
        support = src[inliers]
        if len(support) < self.min_features or ratio < self.min_inlier_ratio:
            self._invalidate("unreliable_camera_transform")
            return
        # Spatial coverage guards against following one group of moving players.
        span = np.ptp(support, axis=0)
        coverage = cv2.contourArea(cv2.convexHull(support)) / (w * h)
        if span[0] < 0.2 * w or span[1] < 0.15 * h or coverage < 0.03:
            self._invalidate("localized_feature_support")
            return
        scale = float(np.linalg.norm(affine[:, 0]))
        rotation = abs(float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0]))))
        if (scale < 0.85 or scale > 1.18 or rotation > 8
                or np.linalg.norm(affine[:, 2]) > 0.18 * np.hypot(w, h)):
            self._invalidate("implausible_camera_transform")
            return
        warped = cv2.warpAffine(self._gray, affine, (w, h))
        overlap = cv2.warpAffine(mask, affine, (w, h)) > 250
        overlap &= current_mask > 0
        overlap = cv2.erode(overlap.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if np.count_nonzero(overlap) < 0.15 * w * h:
            self._invalidate("insufficient_visible_background")
            return
        # Compare the aligned image, allowing motion but rejecting appearance cuts.
        difference = cv2.absdiff(warped, gray)[overlap]
        if float(np.mean(difference)) > 45 or float(np.median(difference)) > 35:
            self._invalidate("scene_cut_or_appearance_change")
            return
        self._polygon = cv2.transform(self._polygon[None, :, :], affine)[0]
        self._status = "propagated"
        self._reason = "background_optical_flow_affine"
        self._confidence = min(self._confidence, ratio)
        self._support = int(len(support))

    def update(self, frame, frame_index, boxes=None):
        """Update once per consecutive source frame, before rejecting detections."""
        frame_index = int(frame_index)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = np.asarray(gray, dtype=np.uint8)
        boxes = np.asarray(boxes if boxes is not None else [], dtype=float).reshape(-1, 4)
        if frame_index in self.keyframes:
            self._seed(self.keyframes[frame_index], gray, frame_index)
        elif self._polygon is not None:
            if gray.shape != self._shape:
                self._invalidate("image_dimensions_changed")
            elif frame_index != self._frame_index + 1:
                self._invalidate("nonconsecutive_source_frames")
            elif frame_index - self._seed_index > self.max_frames:
                self._invalidate("calibration_expired")
            else:
                self._propagate(gray, boxes)
        self._gray = gray.copy()
        self._boxes = boxes.copy()
        self._shape = gray.shape
        self._frame_index = frame_index
        return self.state

    def reject(self, bbox, role="player"):
        """Return (reject, reason); uncertainty and boundary bands always keep."""
        if self._polygon is None:
            return False, "field_unavailable:" + self._reason
        box = np.asarray(bbox, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            return False, "invalid_bbox_fail_open"
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return False, "invalid_bbox_fail_open"
        # Clip feet at the image edge: partial players can extend below the image.
        foot = (float((x1 + x2) / 2), float(np.clip(y2, 0, self._shape[0] - 1)))
        distance = cv2.pointPolygonTest(self._polygon, foot, measureDist=True)
        margin = self.margin + self.box_margin * (y2 - y1)
        if str(role).lower() in {"goalkeeper", "goalie", "referee", "ref"}:
            margin *= 1.5
        if distance < -margin:
            return True, "outside_calibrated_field"
        return False, "inside_field" if distance >= 0 else "boundary_safety_band"


def duplicate_mask(boxes, scores, roles, iou_threshold=0.92):
    """Keep mask removing only same-role, almost identical detection geometry.

    This is detection deduplication, never track-ID or player identity merging.
    Mere overlap is insufficient: every corner must agree within 3.5% of the
    smaller box's width/height, with a matching foot point.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    roles = np.asarray(roles, dtype=object).reshape(-1)
    if len(boxes) != len(scores) or len(boxes) != len(roles):
        raise ValueError("boxes, scores and roles must have equal lengths")
    keep = np.ones(len(boxes), dtype=bool)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="stable")
    for pos, i in enumerate(order):
        if not keep[i]:
            continue
        a = boxes[i]
        aw, ah = a[2:] - a[:2]
        if not np.isfinite(a).all() or min(aw, ah) <= 0:
            continue
        for j in order[pos + 1:]:
            if not keep[j] or roles[i] != roles[j]:
                continue
            b = boxes[j]
            bw, bh = b[2:] - b[:2]
            if not np.isfinite(b).all() or min(bw, bh) <= 0:
                continue
            overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
            intersection = float(overlap[0] * overlap[1])
            iou = intersection / (aw * ah + bw * bh - intersection)
            if iou < iou_threshold:
                continue
            norm = np.array([min(aw, bw), min(ah, bh)] * 2)
            if np.max(np.abs(a - b) / norm) > 0.035:
                continue
            foot_delta = np.array([(a[0] + a[2] - b[0] - b[2]) / 2, a[3] - b[3]])
            if np.linalg.norm(foot_delta / norm[:2]) > 0.04:
                continue
            keep[j] = False
    return keep
