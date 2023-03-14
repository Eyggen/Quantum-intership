import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.layers import Dense

df = pd.read_csv("internship_train.csv")

y = df.pop("target")
train_target = np.array(y)
train_data = np.array(df)

std = train_data.std(axis=0)
train_data /= std

x = train_data
y = train_target

model = keras.Sequential()
model.add(Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(loss="mse", metrics=[keras.metrics.RootMeanSquaredError()], optimizer=keras.optimizers.Adam(learning_rate=0.01))

callbacks = [
    keras.callbacks.ModelCheckpoint("3_q_regresion_model.h5", save_best_only=True),
    keras.callbacks.EarlyStopping(patience = 3, monitor="val_root_mean_squared_error")
]

history = model.fit(x,y, validation_split = 0.1,  epochs=40, callbacks = callbacks)