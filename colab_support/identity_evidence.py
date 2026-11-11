"""Conservative jersey evidence, independent of person identity association.

Team features reuse Andrew Brandenburg's torso/LAB TeamAssigner. Neither
team labels nor OCR numbers merge or rename ByteTrack identities. Results are
evidence, not ground truth. All image processing/history storage is bounded.
"""
from collections import Counter, deque
import re

import cv2
import numpy as np
from sklearn.cluster import KMeans
from team_assigner import TeamAssigner


def _clip_bbox(frame, bbox):
    values = np.asarray(bbox, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = values
    x1, x2 = max(0, int(x1)), min(w, int(np.ceil(x2)))
    y1, y2 = max(0, int(y1)), min(h, int(np.ceil(y2)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


class EvidenceTeamAssigner(TeamAssigner):
    """Reuse the inherited torso crop/LAB estimator, with safe deterministic fit."""

    def get_clustering_model(self, image):
        pixels = image.reshape(-1, 3)
        if len(pixels) > 2048:
            pixels = pixels[np.linspace(0, len(pixels) - 1, 2048, dtype=int)]
        unique = np.unique(pixels, axis=0)
        return KMeans(n_clusters=min(2, len(unique)), random_state=0,
                      n_init=10).fit(pixels)

    def get_player_color(self, frame, bbox):
        clipped = _clip_bbox(frame, bbox)
        if clipped is None or clipped[2] - clipped[0] < 8 or clipped[3] - clipped[1] < 16:
            return None
        color = super().get_player_color(frame, clipped)
        if color is None or not np.isfinite(color).all():
            return None
        return np.asarray(color, dtype=float)

    def assign_team_color_from_colors(self, player_colors):
        colors = np.asarray(player_colors, dtype=float)
        if len(colors) < 10 or len(np.unique(colors.round(1), axis=0)) < 2:
            return False
        model = KMeans(n_clusters=2, random_state=0, n_init=10).fit(colors)
        centers = model.cluster_centers_
        counts = np.bincount(model.labels_, minlength=2)
        if np.linalg.norm(centers[0] - centers[1]) < 18 or counts.min() < 3:
            return False
        # Deterministic display IDs, including when KMeans swaps cluster ordering.
        order = sorted(range(2), key=lambda i: tuple(centers[i]))
        self.lab_centers = centers[order]
        self.kmeans = model
        self.cluster_team = {cluster: team + 1 for team, cluster in enumerate(order)}
        self.team_colors = {
            team + 1: cv2.cvtColor(
                np.clip(np.rint(color), 0, 255).astype(np.uint8).reshape(1, 1, 3),
                cv2.COLOR_LAB2BGR,
            )[0, 0].astype(float)
            for team, color in enumerate(self.lab_centers)
        }
        return True


class StableTeamAssigner:
    """Chunk-compatible, bounded multi-frame team evidence.

    ``all_players`` is the growing tracks['players'] list. process_chunk uses
    global contiguous frame indices. Warmup stores LAB vectors and output
    references only; old unfinished warmup observations remain unknown.
    ``team=0`` always means unknown; team 1/2 do not name real clubs.
    """

    def __init__(self, all_players, min_colors=20, min_votes=3, sample_gap=3,
                 max_colors=128, max_pending=512, history_frames=300,
                 max_tracks=256, max_color_distance=60.0, min_margin=0.25,
                 min_detection_confidence=0.25):
        self.assigner = EvidenceTeamAssigner()
        self.all_players = all_players
        self.min_colors = max(10, int(min_colors))
        self.min_votes = max(3, int(min_votes))
        self.sample_gap = max(1, int(sample_gap))
        self.colors = deque(maxlen=max(self.min_colors, int(max_colors)))
        self.pending = deque(maxlen=max(1, int(max_pending)))
        self.history_frames = max(1, int(history_frames))
        self.max_tracks = max(1, int(max_tracks))
        self.max_color_distance = float(max_color_distance)
        self.min_margin = float(min_margin)
        self.min_detection_confidence = float(min_detection_confidence)
        self.histories = {}
        self.last_seen = {}
        self.last_sample = {}
        self.fitted = False
        self.next_frame = 0
        self._last_fit_attempt = -1

    @property
    def team_colors(self):
        return self.assigner.team_colors

    @staticmethod
    def _unknown(player, status="warming_up"):
        player.update(team=0, team_color=np.array([128., 128., 128.]),
                      team_confidence=0.0, team_status=status,
                      team_evidence={"votes": 0})

    def _expire(self, frame_index):
        stale = [pid for pid, last in self.last_seen.items()
                 if frame_index - last > self.history_frames]
        if len(self.last_seen) - len(stale) > self.max_tracks:
            remaining = sorted((last, pid) for pid, last in self.last_seen.items()
                               if pid not in stale)
            stale.extend(pid for _, pid in remaining[:len(remaining) - self.max_tracks])
        for pid in stale:
            self.last_seen.pop(pid, None)
            self.last_sample.pop(pid, None)
            self.histories.pop(pid, None)

    def _classify(self, color):
        if color is None:
            return None
        distances = np.linalg.norm(self.assigner.lab_centers - color, axis=1)
        best = int(np.argmin(distances))
        margin = float((distances[1 - best] - distances[best]) /
                       max(distances[1 - best], 1.0))
        if distances[best] > self.max_color_distance or margin < self.min_margin:
            return None
        return best + 1, margin

    def _assign(self, pid, player, color, frame_index, sampled):
        history = self.histories.setdefault(pid, deque(maxlen=24))
        while history and frame_index - history[0][0] > self.history_frames:
            history.popleft()
        prediction = self._classify(color) if sampled else None
        if prediction is not None:
            history.append((frame_index, prediction[0], prediction[1]))
        votes = Counter(item[1] for item in history)
        winner, count = votes.most_common(1)[0] if votes else (0, 0)
        total = sum(votes.values())
        agreement = count / total if total else 0.0
        # A newly observed opposing shirt must not inherit a confident old team
        # while the rolling majority catches up (possible association switch).
        contradictory = bool(history and history[-1][1] != winner)
        if count >= self.min_votes and agreement >= 0.8 and not contradictory:
            confidence = agreement * float(np.mean([h[2] for h in history if h[1] == winner]))
            player.update(team=winner, team_color=self.team_colors[winner].copy(),
                          team_status="color_consensus", team_confidence=confidence)
        else:
            self._unknown(player, "conflicting_colors" if len(votes) > 1 else "insufficient_evidence")
        player["team_evidence"] = {"votes": total, "agreement": agreement,
                                   "current_observation_eligible": bool(sampled and color is not None),
                                   "source": "torso_lab", "identity_verified": False}

    def _try_fit(self):
        if self.fitted or len(self.colors) < self.min_colors:
            return
        self.fitted = self.assigner.assign_team_color_from_colors(list(self.colors))
        if self.fitted:
            for g, pid, player, color, sampled in self.pending:
                self._assign(pid, player, color, g, sampled)
            self.pending.clear()

    def process_chunk(self, start, frames):
        if start != self.next_frame:
            raise ValueError("StableTeamAssigner needs consecutive global chunks")
        for offset, frame in enumerate(frames):
            g = start + offset
            self._expire(g)
            for pid, player in self.all_players[g].items():
                self.last_seen[pid] = g
                self._unknown(player)
                if player.get("role", "player") in {"goalkeeper", "referee"}:
                    self._unknown(player, "role_requires_independent_team_evidence")
                    continue
                # Weak detections may sustain motion tracking, but cannot add
                # color votes or train the shared jersey-color clusters.
                eligible = (float(player.get("detection_confidence", 1.0))
                            >= self.min_detection_confidence)
                sampled = eligible and g - self.last_sample.get(pid, -self.sample_gap) >= self.sample_gap
                color = self.assigner.get_player_color(frame, player["bbox"]) if sampled else None
                if sampled:
                    self.last_sample[pid] = g
                if not self.fitted:
                    if color is not None:
                        self.colors.append(color)
                    self.pending.append((g, pid, player, color, sampled))
                else:
                    self._assign(pid, player, color, g, sampled)
            # Try at most every sample gap; degenerate single-team bootstrap
            # stays unknown and the rolling colors can later include both teams.
            if g - self._last_fit_attempt >= self.sample_gap:
                self._last_fit_attempt = g
                self._try_fit()
            self._expire(g)
            self.next_frame = g + 1

    def finish(self):
        if not self.fitted and len(self.colors) >= 10:
            self.min_colors = 10
            self._try_fit()
        if not self.fitted:
            for _, _, player, _, _ in self.pending:
                self._unknown(player, "unresolved_team_clusters")
            self.pending.clear()


class JerseyNumberRecognizer:
    """Optional OCR; three spaced, agreeing readable observations are required.

    EasyOCR is loaded only for an eligible crop with enabled=True. Model
    downloads are disabled: install/prefetch the model explicitly in Colab.
    A missing backend produces an explicit unknown status, never a guessed
    number. ``reader`` injection supports lightweight offline unit tests.
    """

    def __init__(self, enabled=False, reader=None, gpu=True, min_votes=3,
                 sample_gap=5, history_frames=150, max_tracks=256,
                 min_width=28, min_height=44, min_sharpness=40.0,
                 min_confidence=0.85, min_detection_confidence=0.25):
        self.enabled = bool(enabled)
        self.reader = reader
        self.gpu = gpu
        self.min_votes = max(3, int(min_votes))
        self.sample_gap = max(1, int(sample_gap))
        self.history_frames = max(1, int(history_frames))
        self.max_tracks = max(1, int(max_tracks))
        self.min_width, self.min_height = int(min_width), int(min_height)
        self.min_sharpness, self.min_confidence = float(min_sharpness), float(min_confidence)
        self.min_detection_confidence = float(min_detection_confidence)
        self.history = {}
        self.ambiguous_frames = {}
        self.last_seen = {}
        self.last_sample = {}
        self.backend_error = None
        self.read_attempts = 0
        self.error_count = 0
        self.readable_observations = 0

    @property
    def summary(self):
        if not self.enabled:
            status = "disabled"
        elif self.backend_error:
            status = "backend_error"
        elif self.reader is None:
            status = "not_loaded_no_eligible_crop"
        else:
            status = "ready"
        return {"enabled": self.enabled, "backend_status": status,
                "backend_error": self.backend_error, "read_attempts": self.read_attempts,
                "error_count": self.error_count,
                "readable_observations": self.readable_observations,
                "accuracy_validated": False}

    def _load_reader(self):
        if self.reader is None and self.backend_error is None:
            try:
                import easyocr
                self.reader = easyocr.Reader(["en"], gpu=self.gpu, download_enabled=False)
            except Exception as exc:
                self.backend_error = f"{type(exc).__name__}: {exc}"
                self.error_count += 1
        return self.reader

    def _expire(self, g):
        stale = [pid for pid, last in self.last_seen.items() if g - last > self.history_frames]
        survivors = sorted((last, pid) for pid, last in self.last_seen.items() if pid not in stale)
        stale.extend(pid for _, pid in survivors[:max(0, len(survivors) - self.max_tracks)])
        for pid in stale:
            self.history.pop(pid, None)
            self.ambiguous_frames.pop(pid, None)
            self.last_seen.pop(pid, None)
            self.last_sample.pop(pid, None)

    def _read(self, frame, bbox):
        clipped = _clip_bbox(frame, bbox)
        if clipped is None:
            return None, "invalid_crop"
        x1, y1, x2, y2 = clipped
        h, w = y2 - y1, x2 - x1
        torso = frame[y1 + int(h * .15):y1 + int(h * .65),
                      x1 + int(w * .15):x1 + int(w * .85)]
        if torso.shape[0] < self.min_height or torso.shape[1] < self.min_width:
            return None, "too_small"
        sharpness = float(cv2.Laplacian(cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            return None, "blurred"
        reader = self._load_reader()
        if reader is None:
            return None, "backend_unavailable"
        try:
            self.read_attempts += 1
            findings = reader.readtext(torso, allowlist="0123456789", detail=1,
                                       paragraph=False, decoder="greedy")
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            self.error_count += 1
            return None, "ocr_error"
        self.backend_error = None
        candidates = []
        for region, text, confidence in findings:
            text = str(text).strip()
            if re.fullmatch(r"[0-9]{1,2}", text) is None or float(confidence) < self.min_confidence:
                continue
            coords = np.asarray(region, dtype=float)
            if coords.shape != (4, 2) or not np.isfinite(coords).all():
                continue
            digit_height = float(np.ptp(coords[:, 1]))
            digit_width = float(np.ptp(coords[:, 0]))
            # OCR confidence cannot rescue tiny text or a whole-torso false read.
            if digit_height < 12 or digit_width < 5 or digit_height > torso.shape[0] * .95:
                continue
            candidates.append((int(text), float(confidence)))
        numbers = {n for n, _ in candidates}
        if len(numbers) != 1:
            return None, "ambiguous_digits" if numbers else "unreadable"
        self.readable_observations += 1
        return max(candidates, key=lambda item: item[1]), "readable_observation"

    def process(self, frame, players, frame_index):
        g = int(frame_index)
        self._expire(g)
        for pid, player in players.items():
            player.update(jersey_number=None, jersey_number_status="disabled",
                          jersey_number_evidence={"observations": 0, "identity_verified": False})
            if not self.enabled:
                continue
            self.last_seen[pid] = g
            # Reliable opposing shirt evidence can indicate an association
            # switch. Do not carry an old number onto the newly observed
            # shirt, or revive that consensus after the color conflict clears.
            if player.get("team_status") == "conflicting_colors":
                self.history.pop(pid, None)
                self.ambiguous_frames.pop(pid, None)
                self.last_sample.pop(pid, None)
                player["jersey_number_status"] = "association_evidence_conflict"
                player["jersey_number_evidence"].update(
                    frames=[], distinct_numbers=[],
                    current_observation_status="association_evidence_conflict",
                    reset_reason="conflicting_colors")
                continue
            if player.get("role") == "referee":
                player["jersey_number_status"] = "role_not_supported"
                continue
            history = self.history.setdefault(pid, deque(maxlen=24))
            while history and g - history[0][0] > self.history_frames:
                history.popleft()
            status = "between_samples"
            eligible = (float(player.get("detection_confidence", 1.0))
                        >= self.min_detection_confidence)
            if not eligible:
                status = "low_detection_confidence"
            if eligible and g - self.last_sample.get(pid, -self.sample_gap) >= self.sample_gap:
                self.last_sample[pid] = g
                observation, status = self._read(frame, player["bbox"])
                if observation is not None:
                    history.append((g, observation[0], observation[1]))
                if status == "ambiguous_digits":
                    self.ambiguous_frames[pid] = g
            observation_status = status
            numbers = {item[1] for item in history}
            ambiguous_recent = g - self.ambiguous_frames.get(pid, -self.history_frames - 1) <= self.history_frames
            if self.backend_error:
                status = "ocr_error" if status == "ocr_error" else "backend_unavailable"
            elif len(numbers) > 1 or ambiguous_recent:
                status = "conflicting_observations"
            elif len(history) >= self.min_votes:
                player["jersey_number"] = int(history[0][1])
                status = "multi_frame_consensus"
            elif history:
                status = "insufficient_observations"
            player["jersey_number_status"] = status
            player["jersey_number_evidence"] = {
                "observations": len(history), "distinct_numbers": sorted(numbers),
                "frames": [int(item[0]) for item in history],
                "mean_confidence": float(np.mean([item[2] for item in history])) if history else None,
                "current_observation_status": observation_status,
                "identity_verified": False,
            }
            if self.backend_error:
                player["jersey_number_evidence"]["backend_error"] = self.backend_error
        self._expire(g)
