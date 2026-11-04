import cv2
import numpy as np

class CameraMotionEstimator:
    def __init__(self):
        self.total_dx = 0.0
        self.total_dy = 0.0
        self.prev_gray = None

    def update_camera(self, frame):
        """Return incremental image displacement since the previous call.

        The first frame, a changed image size, or unavailable/invalid feature
        matches return (0.0, 0.0). Zero on failure means no measured increment,
        not proof of a stationary camera. Every accepted frame becomes the next
        reference, so recovery never bridges a failed interval. total_dx/dy sum
        only valid estimates; they are never returned as a per-call increment.

        Callers that sample every N frames receive displacement across those N
        frames, rather than a displacement normalized to one source-video frame.
        """
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        previous = self.prev_gray
        self.prev_gray = gray
        if previous is None or previous.shape != gray.shape:
            return 0.0, 0.0

        #Detect Features
        try:
            prev_points = cv2.goodFeaturesToTrack(
                previous,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=20
            )
        except cv2.error:
            return 0.0, 0.0
        if prev_points is None or prev_points.size == 0:
            return 0.0, 0.0

        #Track features
        try:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                gray,
                prev_points,
                None
            )
        except cv2.error:
            return 0.0, 0.0

        if next_points is None or status is None:
            return 0.0, 0.0
        if (len(next_points) != len(prev_points)
                or status.size != len(prev_points)):
            return 0.0, 0.0

        #Keep good matches
        good_old = prev_points[
            status.flatten() == 1
            ]

        good_new = next_points[
            status.flatten() == 1
            ]

        # Compute and accumulate only finite, valid increments.
        motion = (good_new - good_old).reshape(-1, 2)
        motion = motion[np.isfinite(motion).all(axis=1)]
        if len(motion) == 0:
            return 0.0, 0.0

        dx = float(np.median(motion[:, 0]))
        dy = float(np.median(motion[:, 1]))

        self.total_dx += dx
        self.total_dy += dy

        return dx, dy
