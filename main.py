import os
from src.classification_save import *
from src.classification_test import classification_model_predict_one
from src.xray_template.registration import overlay_cc_sitk
from src.xray_template import *
import argparse
import sys

DATA_PATH = './data'
TRAIN_CSV_PATH = os.path.join(DATA_PATH, 'unifesp-x-ray-body-part-classifier', 'train.csv')
TEST_CSV_PATH = os.path.join(DATA_PATH, 'unifesp-x-ray-body-part-classifier', 'sample_submission.csv')
TRAIN_IMAGES_PATH = os.path.join(DATA_PATH, 'png', 'train')
TEMPLATE_PATH = "src/xray_template/template.jpg"
P = os.path.join(DATA_PATH, 'unifesp-x-ray-body-part-classifier', 'test')
OUTPUT_PATH = 'output'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XrayAtlas Prediction Pipeline")
    parser.add_argument('--input', type=str, required=True, help="Path to input X-ray image for prediction")
    args = parser.parse_args()
    predict_image_path = args.input

    if not os.path.exists(predict_image_path):
        print(f"Error: The specified input image '{predict_image_path}' does not exist or is not a valid file.")
        sys.exit(1)

    if os.path.isdir('./data/png/train'):
        print("PNG images already converted")
    else:
        print("Png folder not found.")
        DICOM_DIR = "./data/unifesp-x-ray-body-part-classifier/train/train"
        dcm_to_png(DICOM_DIR, TRAIN_IMAGES_PATH)
    if os.path.exists('xray_classifier.keras'):
        print("Classification model exists!")
    else:
        print("Classification model not found.")
        classification_model_create(TRAIN_CSV_PATH, TRAIN_IMAGES_PATH)


    img, part_labels = classification_model_predict_one(predict_image_path)
    print(part_labels)
    print(f"img: shape={img.shape}, dtype={img.dtype}")
    manual_region_map = get_region_map()
    result_overlay = overlay_cc_sitk(img, part_labels, manual_region_map, TEMPLATE_PATH, white_thresh=220, alpha=0.8)
    if result_overlay is not None:
        display_image(result_overlay, title="Overlay Result")