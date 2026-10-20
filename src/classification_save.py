import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, InputLayer
import cv2
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam

OUTPUT_PATH = 'output'

def Names(label):
    if label == '0':
        return 'Abdomen'
    elif label == '1':
        return 'Ankle'
    elif label == '2':
        return 'Cervical Spine'
    elif label == '3':
        return 'Chest'
    elif label == '4':
        return 'Clavicles'
    elif label == '5':
        return 'Elbow'
    elif label == '6':
        return 'Feet'
    elif label == '7':
        return 'Finger'
    elif label == '8':
        return 'Forearm'
    elif label == '9':
        return 'Hand'
    elif label == '10':
        return 'Hip'
    elif label == '11':
        return 'Knee'
    elif label == '12':
        return 'Lower Leg'
    elif label == '13':
        return 'Lumbar Spine'
    elif label == '14':
        return 'Others'
    elif label == '15':
        return 'Pelvis'
    elif label == '16':
        return 'Shoulder'
    elif label == '17':
        return 'Sinus'
    elif label == '18':
        return 'Skull'
    elif label == '19':
        return 'Thigh'
    elif label == '20':
        return 'Thoracic Spine'
    elif label == '21':
        return 'Wrist'
    
def classification_model_create (TRAIN_CSV_PATH, TRAIN_IMAGES_PATH):
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    train_data = pd.read_csv(TRAIN_CSV_PATH)
    train_data['Target'] = train_data['Target'].apply(lambda x: x.strip(' '))
    multi = []
    multi_target = []
    for i, val in enumerate(train_data.Target):
        if len(val) > 2:
            multi.append(train_data.iloc[i].SOPInstanceUID)
            multi_target.append(train_data.iloc[i].Target)

    train_dff = pd.DataFrame()
    train_dff['SOPInstanceUID'] = train_data['SOPInstanceUID'].copy()
    train_dff['Target'] = train_data['Target'].copy()
    for i, val in enumerate(train_dff.Target):
        if len(val) > 2 and len(val) <= 6:
            x, x1 = train_dff.iloc[i]['Target'].split(' ', 1)
            x_name = Names(x)
            x1_name = Names(x1)
            total = x_name + ' ' + x1_name
            train_dff.at[i, 'label'] = total
        elif len(val) >= 7:
            x, x1, x2 = train_dff.iloc[i]['Target'].split(' ', 2)
            x_name = Names(x)
            x1_name = Names(x1)
            x2_name = Names(x2)
            total = x_name + ' ' + x1_name + ' ' + x2_name
            train_dff.at[i, 'label'] = total
        else:
            x = train_dff.iloc[i]['Target']
            x_name = Names(x)
            train_dff.at[i, 'label'] = x_name
    
    train_dff = pd.get_dummies(train_dff['label'])
    train_dff['SOPInstanceUID'] = train_data['SOPInstanceUID']
    train_dff['Target'] = train_data['Target']
    new_train_data = pd.merge(train_data, train_dff, on='SOPInstanceUID')
    new_train_data.drop('Target_y', axis=1, inplace=True)

    X = []
    y = []

    for val in tqdm(new_train_data.SOPInstanceUID):
        idd = val
        img_path = os.path.join(TRAIN_IMAGES_PATH, f"{val}-c.png")

        try:
            img = plt.imread(img_path)
            img = cv2.resize(img, dsize=(128, 128))
            img_norm = cv2.normalize(img, None, alpha=0, beta=500, norm_type=cv2.NORM_MINMAX)
            y.append(np.ndarray.flatten(np.array(new_train_data.loc[new_train_data['SOPInstanceUID'] == idd][new_train_data.columns[3:]])))
            X.append(np.expand_dims(img, axis=-1))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])


    x_train_augmented = np.copy(x_train)
    for i in tqdm(range(len(x_train))):
        if np.random.random() > 0.5:
            img_tensor = tf.convert_to_tensor(x_train[i])
            augmented = data_augmentation(img_tensor)
            x_train_augmented[i] = augmented.numpy()

    model = tf.keras.models.Sequential()

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same', input_shape=(128, 128, 1)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2048, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation="softmax"))

    optimizer = Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='BinaryCrossentropy',
        metrics=['accuracy']
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1)
    ])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_PATH, "best_model.h5"),
            monitor='val_loss',
            save_best_only=True
        )
    ]

    class_weights = {}
    for i in range(y_train.shape[1]):
        pos_samples = np.sum(y_train[:, i])
        total = len(y_train)
        weight = total / (2 * pos_samples) if pos_samples > 0 else 1.0
        class_weights[i] = weight

    history = model.fit(
        x_train, y_train,
        batch_size=16,
        callbacks=callbacks,
        epochs=25,
        validation_data=(x_test, y_test)
    )

    model.evaluate(x_test, y_test)

    model.save("xray_classifier.keras")