import os
import numpy as np
import cv2
from tqdm import tqdm
import pydicom

# Define paths (relative to src directory)
# DICOM_DIR = "../data/train/train"  # Adjusted path from src
# OUTPUT_DIR = "../data/images/train"  # Output directory for PNGs# Ensure output directory exists

def dcm_to_png(dicom_dir, output_dir):
    """
    Converts DICOM images to PNG format.

    """
    os.makedirs(output_dir, exist_ok=True)  # Find all DICOM files recursively
    dicom_files = []
    for root, _, files in os.walk(dicom_dir):
        for file in files:
            if file.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, file))
    print(f"Found {len(dicom_files)} DICOM files.")  # Convert each DICOM file to PNG

    for dicom_path in tqdm(dicom_files, desc="Converting DICOM to PNG"):
        try:
            # Read DICOM
            dicom_data = pydicom.dcmread(dicom_path)
            image = dicom_data.pixel_array.astype(np.float32)  # Normalize to float

            if hasattr(dicom_data, "RescaleScope") and hasattr(dicom_data, "RescaleIntercept"):
                image *= dicom_data.RescaleScope + dicom_data.RescaleIntercept # Unit correction

            image -= image.min()
            image /= image.max()
            image *= 255.0
            image = np.uint8(image) # scale to 8-bit

            dicom_id = os.path.basename(dicom_path).replace(".dcm", ".png")
            output_path = os.path.join(output_dir, dicom_id)
            cv2.imwrite(output_path, image)

            os.remove(dicom_path)

        except Exception as e:
            print(f"Error processing {dicom_path}: {e}")
    print(f"Conversion completed! PNG images saved in {output_dir}")
