import cv2
from .constants import *

def flip_image_horizontally(img):
    return cv2.flip(img, 1)

def rotate_image_180(img):
    return cv2.rotate(img, cv2.ROTATE_180)

def mirror_arm_leg_templates(site_templates):
    if "Hand" in site_templates:
        site_templates["Hand"] = rotate_image_180(site_templates["Hand"])

    if "Feet" in site_templates:
        site_templates["Feet"] = rotate_image_180(site_templates["Feet"])

    mirrored_templates = {}
    for left_label, right_label in MIRROR_MAP.items():
        if left_label in site_templates:
            flipped = flip_image_horizontally(site_templates[left_label])
            mirrored_templates[right_label] = flipped
    site_templates.update(mirrored_templates)
    return site_templates