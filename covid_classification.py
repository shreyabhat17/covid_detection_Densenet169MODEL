# -*- coding: utf-8 -*-
"""Untitled10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MbJ9neYpS1eqAJkTltOqQd-8DsD5q1vt
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet169
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Assuming you have a directory containing two subdirectories: 'COVID' and 'NonCOVID'
data_dir = '/content/sarscov2-ctscan-dataset'
image_size = (224, 224)  # Adjust image size according to the model's input requirements

# Create a dataframe to store the file paths and labels
data = []
labels = []

for label in ['COVID', 'non-COVID']:
    label_dir = os.path.join(data_dir, label)
    file_names = os.listdir(label_dir)
    file_paths = [os.path.join(label_dir, file_name) for file_name in file_names]
    data.extend(file_paths)
    labels.extend([label] * len(file_names))

df = pd.DataFrame({'data': data, 'labels': labels})

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

batch_size = 32

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Only rescale the testing set
test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches from the train directory
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='data',
    y_col='labels',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Flow testing images in batches from the test directory
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col='data',
    y_col='labels',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Load pre-trained DenseNet169 without the top (fully connected) layers
base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = 10

# Train the model
model.fit(train_generator,
          epochs=epochs,
          validation_data=test_generator)

# Evaluate the model on the test set
_, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

model.save('covid_classification_model.h5')

