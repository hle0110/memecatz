import cv2
import numpy as np


def compute_fit(source_shape, target_width, target_height):
    src_h, src_w = source_shape[:2]
    scale = min(target_width / src_w, target_height / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    return scale, x_offset, y_offset, new_w, new_h


def fit_to_panel(image, target_width, target_height, placeholder_text="no image"):
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    if image is None:
        cv2.putText(
            canvas,
            placeholder_text,
            (20, target_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    scale, x_offset, y_offset, new_w, new_h = compute_fit(image.shape, target_width, target_height)
    resized = cv2.resize(image, (new_w, new_h))
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas
