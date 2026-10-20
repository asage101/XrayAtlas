from .constants import SITE_POSITIONS, SITE_POSITIONS_RIGHT
import numpy as np
import cv2


def draw_highlighted_template(predicted_labels, template_image=None, canvas_size=(2048, 1024), box_size=(128, 128),
                              border_color=(255, 0, 0), thickness=3):
    """
    Draws red borders around predicted anatomical regions

    """
    if template_image is None:
        # Create canvas
        template = np.zeros(canvas_size, dtype=np.uint8)
        for site_dict in [SITE_POSITIONS, SITE_POSITIONS_RIGHT]:
            for _, (x, y) in site_dict.items():
                cv2.rectangle(template, (x, y), (x + box_size[0], y + box_size[1]), 255, thickness=-1)
        template_color = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    else:
        template_color = template_image.copy()
        if len(template_color.shape) == 2:  # convert grayscale to color
            template_color = cv2.cvtColor(template_color, cv2.COLOR_GRAY2BGR)

    # Draw red borders on predicted areas
    for label in predicted_labels:
        if label in SITE_POSITIONS:
            x, y = SITE_POSITIONS[label]
        elif label in SITE_POSITIONS_RIGHT:
            x, y = SITE_POSITIONS_RIGHT[label]
        else:
            print(f"Unknown label: {label}")
            continue

        cv2.rectangle(template_color, (x, y), (x + box_size[0], y + box_size[1]), border_color, thickness)

    return template_color
