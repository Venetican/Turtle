"""
model sesbírán od https://www.kaggle.com/code/tgibbons/sea-turtle-face-detection
"""
"""
Import knihoven
"""

# First, we'll import pandas and numpy, two data processing libraries
import pandas as pd
import numpy as np

# We'll also import seaborn and matplot, twp Python graphing libraries
import seaborn as sns
import matplotlib.pyplot as plt
# Import the needed sklearn libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# The Keras library provides support for neural networks and deep learning
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, SpatialDropout2D, Activation, Lambda, Flatten, LSTM
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, RMSprop
#from tensorflow.keras.utils import np_utils
from tensorflow.keras import utils

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import h5py
from tensorflow.keras.models import load_model

# define path to the read-only input folder
dataPath = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/"
# define path to the writable working folder
working_dir = "C:/Users/Danie/Desktop/Grinding/dp_git/model_B/DSR_results/"

# read text file into pandas DataFrame
df_labels = pd.read_csv("C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/labels.csv", header=None)
df_labels.columns = ['species', 'upper_left_x', 'upper_left_y', 'bbwidth', 'bblength', 'filename', 'image_width', 'image_length' ]
# display DataFrame
df_labels.head


#duplicates remove
print("Shape before dropping duplicates " + str(df_labels.shape))
df_labels = df_labels.drop_duplicates(subset='filename', keep=False)
print("Shape after dropping duplicates " + str(df_labels.shape))

df_labels = df_labels[0:]
df_labels

IMAGE_SIZE = (224, 224)

# Get YOLO bounding boxes from text files
#df_yolo = pd.DataFrame.empty
labelName = dataPath + "labels/" + df_labels["filename"][0][:-3] + 'txt'
df_yolo = pd.read_csv(labelName, delim_whitespace=True, header=None)
df_yolo.columns = ['idd', 'yolo_x', 'yolo_y','yolo_width', 'yolo_length'] 

# Loop over all the file names, skipping the first one that is done above
for imageName in df_labels["filename"][1:]:
    if (imageName[-4] == '.'):
        # some images have ".jpeg" instead of ".jpg"
        labelName = dataPath + "labels/" + imageName[:-3] + 'txt'
    else:
        labelName = dataPath + "labels/" + imageName[:-4] + 'txt'
    #print (labelName)
    bb_txt = pd.read_csv(labelName, delim_whitespace=True, header=None)
    bb_txt.columns = ['idd', 'yolo_x', 'yolo_y','yolo_width', 'yolo_length'] 
    df_yolo = pd.concat([df_yolo, bb_txt.iloc[[0]] ], ignore_index=True)

    #print (df_yolo)
    
df_yolo.head(10)

df_labels.reset_index(drop=True, inplace=True)
df_yolo.reset_index(drop=True, inplace=True)

df_all = pd.concat([df_labels, df_yolo], axis=1)
print ("df_labels shape = "+ str(df_labels.shape))
print ("df_yolo shape = "+ str(df_yolo.shape))
print ("df_all shape = "+ str(df_all.shape))


import os
import skimage.io
import skimage
import cv2
from tqdm import tqdm    # needed for progress bar

load_dir = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/images/"
save_dir = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/images224/"
print ('Creating a new folder at '+save_dir)
os.makedirs(save_dir, exist_ok=True)

#resizing all images to 224
print ('Looping through all the images in the data frame')
#for imageName in df_labels["filename"][1:]:
for imageName in tqdm(df_labels["filename"]):

    load_path = load_dir + imageName
    save_path = save_dir + imageName
    img = mpimg.imread(load_path)
    img_resize = cv2.resize(img, (224, 224))
    mpimg.imsave(save_path, img_resize)

#add bounding box for resized images
width = 224
length = 224
df_all['resize_upper_x'] = df_all.apply(lambda row: round(row['yolo_x'] * width - 0.5 * row['yolo_width'] * width), axis = 1)
df_all['resize_left_y'] = df_all.apply(lambda row:  round(row['yolo_y'] * length - 0.5 * row['yolo_length'] * length), axis = 1)
df_all['resize_lower_x'] = df_all.apply(lambda row: round(row['yolo_x'] * width + 0.5 * row['yolo_width'] * width), axis = 1)
df_all['resize_right_y'] = df_all.apply(lambda row: round(row['yolo_y'] * length + 0.5 * row['yolo_length'] * length), axis = 1)

df_all

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

from glob import glob
from PIL import Image

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
    dataPath = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/"
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
    dataPath = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/"
    image_path = 'C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/images224/' + "Image_" + str(turtleNum) + ".jpg"
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

sample_images = glob("C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/images224/image_100*")

print ("Original Sized Images")
print (sample_images)
plot_images_for_filenames(sample_images) 
plt.show()

displayTurtle_resized(34)
plt.show()


"""
MODEL setup
"""
IMAGE_SIZE = (224, 224)

train_datagen = ImageDataGenerator(
    rescale = 1./255, 
    horizontal_flip = True
) 

# This version uses the resized images
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_all, 
    directory=save_dir,                                               
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

# Set up the Neural Network
IMAGE_SIZE = (224, 224)
# ==== Select one of the pre-trained models from Keras.  Samples are shown below
#pretrained_model = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.EfficientNetB4(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])
#pretrained_model = tf.keras.applications.Xception(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

# Set the following to False so that the pre-trained weights are not changed 
pretrained_model.trainable = False 

model = Sequential()
#  Start with the pretrained model defined above
model.add(pretrained_model)

# Flatten 2D images into 1D data for final layers like traditional neural network
model.add(Flatten())
# GlobalAveragePooling2D is an alternative to Flatter and reduces the size of the layer while flattening
#model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# The final output layer
# Use Sigmoid when predicting YOLO bounding box since that output is between 0 and 1
#model.add(Dense(4, activation='sigmoid'))
# Use relu when predicting corner pixels since the outputs are intergers larger than 1
#model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='linear'))


print ("Pretrained model used:")
pretrained_model.summary()

print ("Final model created:")
model.summary()

# Compile neural network model
#model.compile(optimizer='adam', loss='mean_squared_error')
model.compile(optimizer='adam', loss='mean_absolute_error')

# Train the model with the images in the folders
history = model.fit(
        train_generator,
        # validation_data=(testImages, testTargets), 
        batch_size=16,                  # Number of image batches to process per epoch 
        epochs=100,                      # Number of epochs
        callbacks=[learning_rate_reduction, early_stops],
        )

model.save('C:/Users/Danie/Desktop/final_2_b.h5')
model.predict(13)

testable = "C:/Users/Danie/Desktop/200410016385.jpg"



def predict_yolo(rowNum):
    row = df_all.iloc[[rowNum]]
    filename = row['filename']
    filename = filename.iloc[0]
    image_path = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/images224/" + str(filename)
    # display the image
    img = mpimg.imread(image_path)
    height, width, depth = img.shape
    print ('Read in image of size '+str(height) + ' by ' + str(width) + ' by '+ str(depth))
    img = img/255     # rescale the image values
    plt.figure(figsize = (5,5))
    fig = plt.imshow(img)
    
    # Display to original bounding box
    yolo_orig_bbox = row[['yolo_x', 'yolo_y', 'yolo_width', 'yolo_length']]
    yolo_orig_bbox = yolo_orig_bbox.values.tolist()
    yolo_orig_bbox = yolo_orig_bbox[0]
    print ('Actual YOLO Bounding Box')
    print (yolo_orig_bbox)
    orig_bbox = box_center_to_corner(yolo_orig_bbox, 224, 224)
    print ('Actual Pixel Corners Bounding Box')
    print (orig_bbox)
    fig.axes.add_patch(bbox_to_rect(orig_bbox, 'blue'))
    
    # Predict and display the predicted bounding box
    img = np.reshape(img,[1,224,224,3])
    yolo_pred_bbox = model.predict(img)
    yolo_pred_bbox = yolo_pred_bbox[0]
    print ('Predicted YOLO Bounding Box')
    print (yolo_pred_bbox)
    pred_bbox = box_center_to_corner(yolo_pred_bbox, 224, 224)
    print ('Predicted Pixel Corners Bounding Box')
    print (pred_bbox)
    fig.axes.add_patch(bbox_to_rect(pred_bbox, 'yellow'))

def predict_yolo_only_new(x):
    img  = x
    fig = plt.imshow(img) 
    img = img/255
    img = np.reshape(img,[1,224,224,3])
    yolo_pred_bbox = model.predict(img)
    yolo_pred_bbox = yolo_pred_bbox[0]
    print ('Predicted YOLO Bounding Box')
    print (yolo_pred_bbox)
    pred_bbox = box_center_to_corner(yolo_pred_bbox, 224, 224)
    print ('Predicted Pixel Corners Bounding Box')
    print (pred_bbox)
    fig.axes.add_patch(bbox_to_rect(pred_bbox, 'yellow'))


def prepare(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)


if loaded_model_a.predict(prepare('C:/Users/Danie/Desktop/280282566_387383293310529_1763940945273802682_n.jpg')) > 0.7:
    print("it is a turtle")
else:
    print("it is not a turtle")





predict_yolo_only_new(prepare("C:/Users/Danie/Desktop/200410016385.jpg"))
predict_yolo(127)
plt.show()