import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def load_xray_images(folder_path, target_size=(128, 128)):
    images = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        path = os.path.join(folder_path, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    return images

def cluster_images(images, resize_dim=(64, 64), n_clusters=2):
    if len(images) == 0:
        return []
    features = []
    for img in images:
        small = cv2.resize(img, resize_dim)
        features.append(small.flatten())
    features = np.array(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    dominant_cluster = np.argmax(counts)
    indices = np.where(labels == dominant_cluster)[0]
    return indices
