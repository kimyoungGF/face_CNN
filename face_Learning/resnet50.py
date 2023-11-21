import numpy as np
import os
import PIL
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.models import Sequential
from keras.applications import ResNet50




os.environ['KMP_DUPLICATE_LIB_OK']='True'


import pathlib


data_dir = './face'
data_dir = pathlib.Path(data_dir)



batch_size = 32
img_height = 240
img_width = 200

resnet50 = ResNet50(weights=None, include_top=False, input_shape=(img_width, img_height, 1))

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
with open('./class_names.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(class_names)
num_classes = len(class_names)


model = Sequential()
model.add(resnet50)
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))


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
plt.savefig('./savefig_default.png')
plt.show()


