import numpy as np
import cv2
import os
import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.utils import to_categorical
from keras import layers
from keras.metrics import MeanIoU, IoU

img_dir = r"Data\train_images\train"
mask_dir = r"Data\train_mask\train"

train_img = os.listdir(img_dir)
train_mask = os.listdir(mask_dir)

train_x = []
train_y = []

for img in train_img:
    train_x.append(cv2.imread(os.path.join(img_dir, img),1))

train_x = np.array(train_x)

for mask in train_mask:
    mask_img =np.array(np.expand_dims(cv2.imread(os.path.join(mask_dir, mask),0), axis=-1), dtype=bool)
    train_y.append(mask_img)

train_y = np.array(train_y)

train_y = np.array(train_y)

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

inputs = layers.Input((128,128,3))

s = layers.Lambda(lambda x: x / 255)(inputs)
c1 = layers.Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(s)
c1 = layers.Dropout(0.1)(c1)
p1 = layers.MaxPooling2D((2,2))(c1)


c2 = layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
c2= layers.Dropout(0.1)(c2)
c2 = layers.Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
p2 = layers.MaxPooling2D((2,2))(c2)

c3 = layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
c3 = layers.Dropout(0.1)(c3)
c3 = layers.Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
p3 = layers.MaxPooling2D((2,2))(c3)

c4 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
c4 = layers.Dropout(0.2)(c4)
c4 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
p4 = layers.MaxPooling2D((2,2))(c4)

c5 = layers.Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
c5 = layers.Dropout(0.3)(c5)
c5 = layers.Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)


u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding="same")(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
c6 = layers.Dropout(0.2)(c6)
c6 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

u7 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding="same")(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
c7 = layers.Dropout(0.2)(c7)
c7 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding="same")(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
c8 = layers.Dropout(0.2)(c8)
c8 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

u9 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding="same")(c8)
u9 = layers.concatenate([u9, c1])
c9 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
c9 = layers.Dropout(0.1)(c9)
c9 = layers.Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c9)

model = Model(inputs=[inputs], outputs=(outputs))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[IoU(num_classes=2, target_class_ids=[1])])
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(".h5", save_best_only=True),
    keras.callbacks.EarlyStopping(patience = 2, monitor="val_io_u")
]
epochs = 1000
model.fit(train_x, train_y, epochs=epochs, validation_split=0.1, callbacks=callbacks)