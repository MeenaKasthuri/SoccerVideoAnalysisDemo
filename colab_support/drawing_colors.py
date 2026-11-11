"""Convert team evidence to OpenCV colors only at the drawing boundary."""
import math
from numbers import Real

import numpy as np


UNKNOWN_BGR = (0, 0, 255)
_COLOR_ERROR = "BGR color must contain exactly three finite real numeric channels"


def normalize_bgr(color):
    """Return three clipped, rounded Python ints without changing the input.

    NumPy integer scalars are valid evidence values, but OpenCV's color parser
    may reject them even inside a tuple. ``None`` is not a color; drawing
    methods alone choose the explicit unknown-team fallback.
    """
    if isinstance(color, np.ndarray):
        if color.shape != (3,):
            raise ValueError(_COLOR_ERROR)
    elif not isinstance(color, (tuple, list)) or len(color) != 3:
        raise ValueError(_COLOR_ERROR)
    for channel in color:
        if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, Real):
            raise ValueError(_COLOR_ERROR)
        if channel != channel or channel == math.inf or channel == -math.inf:
            raise ValueError(_COLOR_ERROR)
    return tuple(int(round(min(255, max(0, channel)))) for channel in color)


def _drawing_color(color):
    return UNKNOWN_BGR if color is None else normalize_bgr(color)


class ColorSafeDrawingMixin:
    """Keep tracker/classifier data unchanged while supplying native BGR ints."""
    def draw_ellipse(self, frame, bbox, color, track_id=None, draw_id=True):
        return super().draw_ellipse(frame, bbox, _drawing_color(color),
                                    track_id=track_id, draw_id=draw_id)

    def draw_trajectory(self, frame, track_id, bbox, color=(0, 255, 255),
                        history_length=50, offset_x=0, offset_y=0):
        return super().draw_trajectory(
            frame, track_id, bbox, color=_drawing_color(color),
            history_length=history_length, offset_x=offset_x, offset_y=offset_y)

    def draw_triangle(self, frame, bbox, color):
        return super().draw_triangle(frame, bbox, _drawing_color(color))

    def draw_ball_trail(self, frame, color=(0, 255, 255)):
        return super().draw_ball_trail(frame, color=_drawing_color(color))
