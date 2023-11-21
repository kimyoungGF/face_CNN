import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential

os.environ['KMP_DUPLICATE_LIB_OK']='True'


import pathlib


data_dir = './face'
data_dir = pathlib.Path(data_dir)

num_classes = 3

batch_size = 32
img_height = 240
img_width = 200

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset='training',
    color_mode='grayscale',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)



val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    color_mode='grayscale',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


class_names = train_ds.class_names
print(class_names)


model = Sequential([
  layers.Conv2D(256, 1, padding='same', activation='relu',input_shape=(img_height,img_width,1)),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 1, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 1, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 1, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(64, activation='relu'),
  layers.Dense(16, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#folder_directory = "./callback1"
#checkPoint_path = folder_directory+"/model_{epoch}.ckpt"

#my_period =100
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkPoint_path,save_weights_only=True, verbose=1, period=my_period)

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
    #callbacks=[cp_callback]
)

model.save('./my_model')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


