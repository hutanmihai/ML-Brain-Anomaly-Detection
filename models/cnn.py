import os
from time import time

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ReLU
from keras.losses import binary_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

from utils.write import write

# DATAFRAMES
test_dataframe = pd.read_csv('../data_frames/test_dataframe.csv', sep=',', names=['id']).astype(str)
validation_dataframe = pd.read_csv('../data_frames/validation_dataframe.csv', sep=',', names=['id', 'class']).astype(
    str)
train_dataframe = pd.read_csv('../data_frames/train_dataframe.csv', sep=',', names=['id', 'class']).astype(str)

# PLOT THE UNBALANCED DATA
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
train_dataframe['class'].value_counts().plot(kind='bar', ax=ax[0], title='Train')
validation_dataframe['class'].value_counts().plot(kind='bar', ax=ax[1], title='Validation')
plt.show()

# CALCUALATE CLASS WEIGHTS
train = train_dataframe['class'].values
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train), y=train)

class_weights = {0: class_weights[0], 1: class_weights[1]}
print(class_weights)

# CONSTANTS
data_dir_path = os.path.join(os.getcwd(), '../data/data/')
batch_size = 35
image_resize = (224, 224)


# DATA AUGMENTATION FUNCTION
def contrast_stretching(image, min_output=0, max_output=255):
    # Find the minimum and maximum pixel values in the image
    min_input = np.min(image)
    max_input = np.max(image)

    # Check if the input image has any variation in pixel values
    if min_input == max_input:
        return image

    # Calculate the scaling factor
    scale_factor = (max_output - min_output) / (max_input - min_input)

    # Apply contrast stretching to the image
    stretched_image = (image - min_input) * scale_factor + min_output
    return stretched_image


# DATA GENERATORS
train_data_gen = ImageDataGenerator(
    rescale=1. / 255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    rotation_range=15,
    fill_mode='nearest',
    horizontal_flip=True,
    preprocessing_function=contrast_stretching,
)

validation_data_gen = ImageDataGenerator(
    rescale=1. / 255.0,
    preprocessing_function=contrast_stretching,
)

test_data_gen = ImageDataGenerator(
    rescale=1. / 255.0,
    preprocessing_function=contrast_stretching,
)

train_generator = train_data_gen.flow_from_dataframe(
    dataframe=train_dataframe,
    directory=data_dir_path,
    x_col='id',
    y_col='class',
    target_size=image_resize,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
)

validation_generator = validation_data_gen.flow_from_dataframe(
    dataframe=validation_dataframe,
    directory=data_dir_path,
    x_col='id',
    y_col='class',
    target_size=image_resize,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False,
)

test_generator = test_data_gen.flow_from_dataframe(
    dataframe=test_dataframe,
    directory=data_dir_path,
    x_col='id',
    y_col=None,
    target_size=image_resize,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
)


# F1 SCORE FUNCTION
def score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


# CALLBACKS
learn_rate = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto', min_delta=1e-7,
                               cooldown=1, min_lr=1e-7)

early_stop = EarlyStopping(monitor='val_score', min_delta=1e-4, patience=60, verbose=1, mode='max', baseline=None,
                           restore_best_weights=True)

checkpoint_path = "saved_models/model_weights.{epoch:02d}-{val_loss:.2f}--{val_score:.2f}.h5"
checkpoint_callback = ModelCheckpoint(checkpoint_path, verbose=0)

tensorboard = TensorBoard(log_dir=f"logs/{time()}")

# MODEL
model = Sequential(
    [
        Conv2D(filters=32, kernel_size=3, padding='same',
               input_shape=(image_resize[0], image_resize[1], 1)),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters=32, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Dropout(0.2),

        Conv2D(filters=64, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters=64, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Dropout(0.2),

        Conv2D(filters=128, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters=128, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Dropout(0.2),

        Conv2D(filters=256, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters=256, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        Conv2D(filters=256, kernel_size=3, padding='same'),
        BatchNormalization(),
        ReLU(),
        MaxPooling2D(),
        Dropout(0.2),

        Flatten(),
        Dense(256),
        ReLU(),
        BatchNormalization(),
        Dense(256),
        ReLU(),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ]
)

model.summary()

start = time()

model.compile(optimizer=Adam(learning_rate=0.001), loss=binary_crossentropy, metrics=[score])

history = model.fit(train_generator, epochs=500, batch_size=batch_size, verbose=1,
                    validation_data=validation_generator, class_weight=class_weights,
                    callbacks=[early_stop, tensorboard, learn_rate, checkpoint_callback])

print(f"--------------------------- {time() - start} ---------------------------")

plt.figure()
plt.plot(history.history['score'])
plt.plot(history.history['val_score'])
plt.title('Model Score')
plt.ylabel('F1 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# WRITE TEST PREDICTIONS
predicted_labels = model.predict(test_generator, verbose=1)
predicted_labels = np.round(predicted_labels).astype(int).reshape(-1, )

write(predicted_labels)

print(f"-----------------------------------------------------")
