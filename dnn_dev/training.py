# Reference: https://github.com/elisim/piven

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from model_builder import build_model
import tensorflow as tf
from lossfunction import piven_loss
import matplotlib.pyplot as plt

# hyper parameters
dropout_rate = 0.1
learning_rate = 0.001
epochs = 200
batch_size = 128

# General Settings
seed = 123
log_dir = './logs'
model_dir = './models/dnn_{epoch:02d}-{val_loss:.2f}.h5'


# Load training data
df = pd.read_csv("etl_data.csv")
X = df.drop('median_ces', axis=1)
Y = df['median_ces']

# convert to numpy array
X_array = np.array(X)
Y_array = np.array(Y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size=0.2, random_state=seed)

# Load model and model settings
model = build_model(dropout_rate)

# compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    optimizer=opt,
    loss=piven_loss()
)
model.summary()


# Define call back functions
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          histogram_freq=1, write_graph=True, write_images=True)

save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_dir, monitor='val_loss', save_best_only=True)

lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                   patience=10,
                                                   verbose=1,
                                                   factor=0.4,
                                                   min_lr=0.0001)

call_back = [tensorboard_callback, save_callback, lr_callback]

result = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=call_back)

# Plot training validation curve
plt.figure(figsize=(10,5))
plt.plot(result.epoch, result.history['loss'], color='r', label='Training Loss')
plt.plot(result.epoch, result.history['val_loss'], color='b', label='Validation Loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()