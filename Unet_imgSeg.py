from gc import callbacks
import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from data_preparation import training_data_preparation, test_data_preparation
from unet_model import unet_model, train_model

seed = 42
np.random.seed = seed
# Input Image Shape:
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3 

# Read Data:
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

# Building X_train and Y_train:
X_train, Y_train = training_data_preparation(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TRAIN_PATH)
print('Xtrain: ', X_train)
# Building X_test:
X_test = test_data_preparation(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, TEST_PATH)
print('X_test: ', X_test)

image_x = 100
imshow(X_train[image_x])
plt.show()

imshow(np.squeeze(Y_train[image_x]))
plt.show()
# Build the model:
model  = unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
results = train_model(model, X_train, Y_train)
# inputs:
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x : x / 255.0)(inputs)

# First Layer:  
C1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
C1 = tf.keras.layers.Dropout(0.1)(C1)
C1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C1)
P1 = tf.keras.layers.MaxPooling2D((2,2))(C1)
# Second Layer:
C2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P1)
C2 = tf.keras.layers.Dropout(0.1)(C2)
C2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C2)
P2 = tf.keras.layers.MaxPooling2D((2,2))(C2)
# Third Layer:
C3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P2)
C3 = tf.keras.layers.Dropout(0.2)(C3)
C3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C3)
P3 = tf.keras.layers.MaxPooling2D((2,2))(C3)
# Forth Layer:
C4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P3)
C4 = tf.keras.layers.Dropout(0.2)(C4)
C4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C4)
P4 = tf.keras.layers.MaxPooling2D((2,2))(C4)
# Fifth Layer:
C5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(P4)
C5 = tf.keras.layers.Dropout(0.3)(C5)
C5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C5)

# Expansive Path:
# Sixth Layer
U6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(C5)
U6 = tf.keras.layers.concatenate([U6, C4])
C6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U6)
C6 = tf.keras.layers.Dropout(0.2)(C6)
C6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C6)
# Seventh Layer:
U7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(C6)
U7 = tf.keras.layers.concatenate([U7, C3])
C7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U7)
C7 = tf.keras.layers.Dropout(0.2)(C7)
C7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C7)
# Eighth Layer:
U8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(C7)
U8 = tf.keras.layers.concatenate([U8, C2])
C8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U8)
C8 = tf.keras.layers.Dropout(0.1)(C8)
C8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C8)
# Ninth Layer:
U9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(C8)
U9 = tf.keras.layers.concatenate([U9, C1], axis=3)
C9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(U9)
C9 = tf.keras.layers.Dropout(0.1)(C9)
C9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(C9)

# Output Layer:
outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(C9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
callbacks = [
tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
tf.keras.callbacks.TensorBoard(log_dir='logs')]

# train the model:
results = model.fit(X_train,Y_train, batch_size=16, epochs=50, validation_split=0.1, callbacks=callbacks)

####################################

idx = random.randint(0, len(X_train))


preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
imshow(X_train[ix])
plt.show()
imshow(np.squeeze(Y_train[ix]))
plt.show()
imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_val_t[ix]))
plt.show()