import os
import shutil
import pandas as pd

LABEL_MAP = {
    0: 'Abdomen', 1: 'Ankle', 2: 'Cervical Spine', 3: 'Chest', 4: 'Clavicles',
    5: 'Elbow', 6: 'Feet', 7: 'Finger', 8: 'Forearm', 9: 'Hand',
    10: 'Hip', 11: 'Knee', 12: 'Lower Leg', 13: 'Lumbar Spine', 14: 'Others',
    15: 'Pelvis', 16: 'Shoulder', 17: 'Sinus', 18: 'Skull', 19: 'Thigh',
    20: 'Thoracic Spine', 21: 'Wrist'
}

def sort_images_by_label(sorted_dir, train_csv, png_dir):
    os.makedirs(sorted_dir, exist_ok=True)
    df = pd.read_csv(train_csv)

    for _, row in df.iterrows():
        uid = row["SOPInstanceUID"]
        labels = str(row["Target"]).strip().split()

        matching_files = [f for f in os.listdir(png_dir) if f.startswith(uid)]
        if not matching_files:
            print(f"Image not found for UID: {uid}")
            continue

        src_img_path = os.path.join(png_dir, matching_files[0])

        for label_id in labels:
            label_id = int(label_id)
            if label_id not in LABEL_MAP:
                continue
            label_name = LABEL_MAP[label_id]
            dest_dir = os.path.join(sorted_dir, label_name)
            os.makedirs(dest_dir, exist_ok=True)

            dest_path = os.path.join(dest_dir, f"{uid}.png")
            shutil.copy(src_img_path, dest_path)

    print("Finished sorting images by label.")
