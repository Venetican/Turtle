from Model_B_dataprep import df_all
print(df_all)

import os
import skimage.io
import skimage
import cv2
import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19

#resize images to needed format
loaded_dir = ("C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/images/")
target_dir = ("C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/images_resized/")

for imageName in df_all["filename"][1:]:
    load_path = loaded_dir + imageName
    save_path = target_dir + imageName
    img = mpimg.imread(load_path)
    img_resize = cv2.resize(img, (224, 224))
    mpimg.imsave(save_path, img_resize)


width = 224
length = 224
df_all['resize_upper_x'] = df_all.apply(lambda row: round(row['yolo_x'] * width - 0.5 * row['yolo_width'] * width), axis = 1)
df_all['resize_left_y'] = df_all.apply(lambda row:  round(row['yolo_y'] * length - 0.5 * row['yolo_length'] * length), axis = 1)
df_all['resize_lower_x'] = df_all.apply(lambda row: round(row['yolo_x'] * width + 0.5 * row['yolo_width'] * width), axis = 1)
df_all['resize_right_y'] = df_all.apply(lambda row: round(row['yolo_y'] * length + 0.5 * row['yolo_length'] * length), axis = 1)
print(df_all)

print(df_all)

def box_corner_to_center_arr(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner_arr(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)


def box_center_to_corner(box, width, length):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    x1 = round(cx * width - 0.5 * w * width)
    y1 = round(cy * length - 0.5 * h * length)
    x2 = round(cx * width + 0.5 * w * width)
    y2 = round(cy * length + 0.5 * h * length)
    bbox = [x1, y1, x2, y2]
    return bbox

def plot_images(imgs, rows=5):
    # Set figure to 15 inches x 8 inches
    figure = plt.figure(figsize=(15, 8))
    cols = len(imgs) // rows + 1
    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        plt.imshow(imgs[i])

def plot_images_for_filenames(filenames, rows=5):
    imgs = [plt.imread(f'{filename}') for filename in filenames]
    return plot_images(imgs, rows)

def displayTurtle(turtleNum):
    dataPath = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/"
    image_path = dataPath + "images/Image_" + str(turtleNum) + ".jpg"
    lable_path = dataPath + "labels/Image_" + str(turtleNum) + ".txt"
    # display the image
    img = mpimg.imread(image_path)
    height, width, depth = img.shape
    plt.figure(figsize = (15,15))
    fig = plt.imshow(img)
    # read the yolo cordinates.  read_csv stores the numbers in the column names
    label = pd.read_csv(lable_path, sep=" ")
    yolobbox = list(label.columns[1:5])
    yolobbox = [float(i) for i in yolobbox]    # convert numbers from strings to floats
    print (yolobbox)
    # convert to corners
    bbox = box_center_to_corner(yolobbox, width, height)
    print (bbox)
    fig.axes.add_patch(bbox_to_rect(bbox, 'blue'))

def displayTurtle_resized(turtleNum):
    dataPath = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/"
    image_path = 'C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/images_resized/' + "Image_" + str(turtleNum) + ".jpg"
    lable_path = dataPath + "labels/Image_" + str(turtleNum) + ".txt"
    # display the image
    img = mpimg.imread(image_path)
    height, width, depth = img.shape
    plt.figure(figsize = (5,5))
    fig = plt.imshow(img)
    # read the yolo cordinates.  read_csv stores the numbers in the column names
    label = pd.read_csv(lable_path, sep=" ")
    yolobbox = list(label.columns[1:5])
    yolobbox = [float(i) for i in yolobbox]    # convert numbers from strings to floats
    print (yolobbox)
    # convert to corners
    bbox = box_center_to_corner(yolobbox, width, height)
    print (bbox)
    fig.axes.add_patch(bbox_to_rect(bbox, 'blue'))    

sample_images = glob("C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/images/image_3.jpg")

print ("C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/images/image_3.jpg")
print (sample_images)
plot_images_for_filenames(sample_images)
displayTurtle(3)
displayTurtle_resized(405)
plt.show()     


#
IMAGE_SIZE = (224, 224)

train_datagen = image.ImageDataGenerator(
    rescale = 1./255, 
    horizontal_flip = True
) 

# This version uses the resized images
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_all, 
    directory="C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B_transform/images_resized/",                                               
    x_col="filename", 
    #y_col=['resize_upper_x', 'resize_left_y', 'resize_lower_x', 'resize_right_y'], 
    y_col=['yolo_x', 'yolo_y', 'yolo_width', 'yolo_length'], 
    #has_ext=True, 
    class_mode="raw", 
    target_size=IMAGE_SIZE,
    batch_size=32
)

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                            patience=5, 
                                            verbose=2, 
                                            factor=0.5,                                            
                                            min_lr=0.000001)

early_stops = EarlyStopping(monitor='loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=2, 
                            mode='auto')

checkpointer = ModelCheckpoint(filepath = 'cis3115.{epoch:02d}-{accuracy:.6f}.hdf5',
                               verbose=2,
                               save_best_only=True, 
                               save_weights_only = True)

pretrained_model = VGG19(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
pretrained_model.trainable = False 

"""
model
"""
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='linear'))

print ("Final model created:")
model.summary()
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
history = model.fit(
        train_generator,
        # validation_data=(testImages, testTargets),
        batch_size=16,                  # Number of image batches to process per epoch 
        epochs=100,                      # Number of epochs
        callbacks=[learning_rate_reduction, early_stops],
        )

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])        

history = model.fit(
        train_generator,
        # validation_data=(testImages, testTargets),
        batch_size=16,                  # Number of image batches to process per epoch 
        epochs=100,                      # Number of epochs
        callbacks=[learning_rate_reduction, early_stops],
        )
model.save('C:/Users/Danie/Desktop/Grinding/data_projekt/ModelB/final_2.h5')                    