import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

Datadirectory = "C:/Users/ASUS/Music/SE Semester Project/trains/"
Classes = ["0", "1", "2", "3", "4", "5", "6"]

def convert_to_rgb(img):
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=convert_to_rgb
)

#  load images in batches
train_data = datagen.flow_from_directory(
    Datadirectory,  # Directory containing subdirectories of classes
    target_size=(224, 224),  # Resize images to the required size
    batch_size=64,  # Batch size for loading data
    class_mode='sparse',  # For sparse labels (integer encoded classes)
    shuffle=True  # Shuffle the data for randomness
)

# Define the model
model = tf.keras.applications.ResNet50(
    include_top=False, 
    weights='imagenet',  
    input_shape=(224, 224, 3)  
)

base_input = model.input
base_output = model.output

final_output = layers.GlobalAveragePooling2D()(base_output)
final_output = layers.Dense(128)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output)


new_model = models.Model(inputs=base_input, outputs=final_output)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

new_model.fit(train_data, epochs=50)

new_model.save('Emoton AI Model.h5')
