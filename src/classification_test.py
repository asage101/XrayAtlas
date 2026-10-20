import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.models import load_model

names_col = [
    ['Abdomen'],
    ['Abdomen', 'Chest'],
    ['Abdomen', 'Pelvis'],
    ['Ankle'],
    ['Ankle', 'Feet'],
    ['Ankle', 'Knee', 'Lower Leg'],
    ['Ankle', 'Lower Leg'],
    ['Cervical', 'Spine'],
    ['Cervical Spine', 'Lumbar Spine', 'Thoracic Spine'],
    ['Cervical Spine', 'Thoracic Spine'],
    ['Chest'],
    ['Chest', 'Feet'],
    ['Chest', 'Finger'],
    ['Chest', 'Pelvis'],
    ['Clavicles'],
    ['Elbow'],
    ['Elbow', 'Forearm'],
    ['Feet'],
    ['Finger'],
    ['Finger', 'Hand'],
    ['Forearm'],
    ['Forearm', 'Shoulder'],
    ['Hand'],
    ['Hand', 'Wrist'],
    ['Hip'],
    ['Hip', 'Knee', 'Thigh'],
    ['Hip', 'Thigh'],
    ['Knee'],
    ['Knee', 'Lower Leg'],
    ['Lower Leg'],
    ['Lumbar Spine'],
    ['Lumbar Spine', 'Thoracic Spine'],
    ['Others'],
    ['Pelvis'],
    ['Pelvis', 'Thigh'],
    ['Shoulder'],
    ['Sinus'],
    ['Skull'],
    ['Thigh'],
    ['Thoracic Spine'],
    ['Wrist']
]


def read_first_png(path):
    if os.path.isfile(path):
        img = plt.imread(path, cv2.IMREAD_GRAYSCALE)
        return img
    elif os.path.isdir(path):
        files = os.listdir(path)
        png_files = [f for f in files if f.lower().endswith('.png')]
        if not png_files:
            raise FileNotFoundError("No PNG images found in the folder.")
        first_png = png_files[0]
        image_path = os.path.join(path, first_png)

        return plt.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
def classification_model_predict_one (PREDICT_IMAGE_PATH):
    model = load_model("xray_classifier.keras")
    img = read_first_png(PREDICT_IMAGE_PATH)
    img_resized = cv2.resize(img,dsize=(128,128))
    test_inp = []
    test_inp.append(np.expand_dims(img_resized,axis=-1))
    test_inp = np.array(test_inp)
    pred = model.predict(test_inp)

    parts = names_col[np.argmax(pred[0])]
    return img, parts