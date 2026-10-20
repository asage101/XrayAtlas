import cv2
import numpy as np
from .constants import *

def stitch_templates(site_templates):
    canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)
    h_canvas, w_canvas = CANVAS_SIZE
    for site, img in site_templates.items():
        if site not in SITE_POSITIONS:
            continue
        x, y = SITE_POSITIONS[site]
        img = cv2.resize(img, (128, 128))
        h, w = img.shape[:2]
        end_y = min(y + h, h_canvas)
        end_x = min(x + w, w_canvas)

        crop_h = end_y - y
        crop_w = end_x - x
        if crop_h > 0 and crop_w > 0:
            patch = img[:crop_h, :crop_w]
            canvas[y:end_y, x:end_x] = patch
        else:
            print(f"skipping site {site}")
    return canvas