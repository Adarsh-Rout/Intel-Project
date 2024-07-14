import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the paths
base_path = 'idd20k_lite'
train_img_path = os.path.join(base_path, 'leftImg8bit', 'train')
val_img_path = os.path.join(base_path, 'leftImg8bit', 'val')
train_ann_path = os.path.join(base_path, 'gtFine', 'train')
val_ann_path = os.path.join(base_path, 'gtFine', 'val')

# Function to load images and annotations
def load_data(img_path, ann_path):
    images = []
    annotations = []
    for folder in os.listdir(img_path):
        folder_path = os.path.join(img_path, folder)
        for img_file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, img_file))
            images.append(img)
            
            ann_file = img_file.replace('_image.jpg', '_label.png')
            ann = cv2.imread(os.path.join(ann_path, folder, ann_file), 0)
            annotations.append(ann)
    return images, annotations

train_images, train_annotations = load_data(train_img_path, train_ann_path)
val_images, val_annotations = load_data(val_img_path, val_ann_path)

# Data generator for data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generate augmented data
train_gen = datagen.flow(np.array(train_images), batch_size=32)
val_gen = datagen.flow(np.array(val_images), batch_size=32)


import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load pre-trained YOLO model
yolo_model = load_model('yolov3.h5')

# Compile the model
yolo_model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
yolo_model.fit(train_gen, epochs=10, validation_data=val_gen)
