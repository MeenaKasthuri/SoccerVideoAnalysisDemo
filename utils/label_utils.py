import cv2


def draw_clamped_label(
        frame,
        label,
        origin,
        color,
        text_color=(0, 0, 0),
        font_scale=0.6,
        thickness=2,
        padding=3,
        background=True
):
    """Draw a label while keeping its text and optional background in-frame."""
    label = str(label)
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )

    frame_height, frame_width = frame.shape[:2]
    max_x = max(padding, frame_width - text_width - padding)
    min_baseline = text_height + padding
    max_baseline = max(min_baseline, frame_height - baseline - padding)

    x = max(padding, min(int(origin[0]), max_x))
    text_baseline = max(
        min_baseline,
        min(int(origin[1]), max_baseline)
    )

    if background:
        cv2.rectangle(
            frame,
            (x - padding, text_baseline - text_height - padding),
            (
                x + text_width + padding,
                text_baseline + baseline + padding
            ),
            color,
            cv2.FILLED
        )

    cv2.putText(
        frame,
        label,
        (x, text_baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness
    )
    return frame
