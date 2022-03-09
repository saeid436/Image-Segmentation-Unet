# unet_model function is the architecure of the model...
# train_model function trains the model on training data...

import tensorflow as tf

def unet_model(_imgHeight,  _imgWidth, _imgChannel):
    # Build the model:
    # inputs:
    inputs = tf.keras.layers.Input((_imgHeight,  _imgWidth, _imgChannel))
    s = tf.keras.layers.Lambda(lambda x : x / 255.0)(inputs)

    # Contraction path
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
    return model

def train_model( _model, _xTrain, _yTrain):
    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
    callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]

    # train the model:
    results = _model.fit(_xTrain,_yTrain, batch_size=16, epochs=50, validation_split=0.1, callbacks=callbacks)
    return results