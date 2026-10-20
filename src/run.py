import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from xray_template import *

DICOM_DIR = "../data/train"
# PNG_DIR = "../data/png/train/"
PNG_DIR = "../data/images/train/"
TRAIN_CSV = "../data/unifesp-x-ray-body-part-classifier/train.csv"
SORTED_DIR = "../data/sorted_by_site"
# TEMPLATE_OUTPUT_PATH = "./xray_template/template.png"
# TEST_IMG_PATH = os.path.join(PNG_DIR, "1.2.826.0.1.3680043.8.498.11990760348755488730051066538791533229-c.png")
# TEST_IMG_PATH = os.path.join(PNG_DIR, "1.2.826.0.1.3680043.8.498.11517871159945297695228776765479996926-c.png")
# TEST_IMG_PATH = os.path.join(PNG_DIR, "1.2.826.0.1.3680043.8.498.11232365649155304259072062868516202248-c.png")
# TEST_IMG_PATH = os.path.join(PNG_DIR, "1.2.826.0.1.3680043.8.498.10093718510761734264315711288650185574-c.png")
TEST_IMG_PATH = os.path.join(PNG_DIR, "1.2.826.0.1.3680043.8.498.10151099395979621308892699787301311930-c.png")
# TEST_IMG_PATH = os.path.join(PNG_DIR, "1.2.826.0.1.3680043.8.498.24493226107909011198872263168825685069-c.png")
PREDICTED_LABELS = ["Chest"]
# PREDICTED_LABELS = ["Feet"]
# PREDICTED_LABELS = ["Hand"]
# PREDICTED_LABELS = ["Abdomen", "Hip", "Pelvis", "Thigh"]
TEMPLATE_PATH = "./xray_template/template.jpg"

if __name__ == "__main__":
    import matplotlib

    matplotlib.use('TkAgg')
    #   if not os.path.exists(PNG_DIR) or not os.listdir(PNG_DIR):
    #   dcm_to_png(DICOM_DIR, PNG_DIR)
    #   sort_images_by_label(SORTED_DIR, TRAIN_CSV, PNG_DIR)

    # 1. Generate per-site templates in memory
    # print("Generating per-site templates...")
    # site_templates = generate_site_templates(SORTED_DIR)
    #
    # # 2. Stitch per-site templates into one whole-body template
    # print("Stitching templates into whole-body template...")
    # whole_body_template = stitch_templates(site_templates)
    # # Save the template
    # cv2.imwrite(TEMPLATE_OUTPUT_PATH, whole_body_template)
    # print(f"Whole-body template saved at {TEMPLATE_OUTPUT_PATH}")

    # img = np.zeros((2048, 1024), dtype=np.uint8)
    #
    # for label, (x, y) in SITE_POSITIONS.items():
    #     cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    #     cv2.rectangle(img, (x, y), (x + 64, y + 64), 255, 1)
    #
    # for label, (x, y) in SITE_POSITIONS_RIGHT.items():
    #     cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
    #     cv2.rectangle(img, (x, y), (x + 64, y + 64), 255, 1)
    #
    # plt.figure(figsize=(12, 18))
    # plt.imshow(img, cmap='gray')
    # plt.title("Template layout with label bounding boxes")
    # plt.axis('off')
    # plt.show()

    # 3. Register and overlay a new image on the whole-body template
    # print("Registering and overlaying new image...")
    # register_and_overlay_new_image(PREDICTED_LABELS, TEST_IMG_PATH, site_templates, whole_body_template)


    # Convert String to List in order to separate different labels
    # if isinstance(PREDICTED_LABELS,str):
    #     PREDICTED_LABELS = [PREDICTED_LABELS]
    # separated_labels = []
    # for body_part in PREDICTED_LABELS:
    #     if isinstance(body_part, str):
    #         separated_labels.extend([label.strip() for label in body_part.split(",")])
    # # Highlight the predicted labels
    # template_img = draw_highlighted_template(separated_labels, template_image=whole_body_template)
    # display_image(template_img)

    manual_region_map = get_region_map()
    regions = extract_regions(TEMPLATE_PATH, manual_region_map)
    new_xray = cv2.imread(TEST_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    if new_xray is None:
        print("Error: Could not load new x-ray image.")
    else:
        # result_overlay = overlay(new_xray, PREDICTED_LABELS, manual_region_map, TEMPLATE_PATH, white_thresh=220, deskew=False, alpha=0.8)
        result_overlay = overlay_cc_normed(new_xray, PREDICTED_LABELS, manual_region_map, TEMPLATE_PATH, white_thresh=220, alpha=0.8)
        if result_overlay is not None:
            display_image(result_overlay, title="Overlay Result")
