"""
MODEL A - přetrénování
"""
"""
Nahrání potřebných balíčků
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
import h5py
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

"""
úprava dat
"""
#Množina trénovacích dat
datadef_train = image.ImageDataGenerator(
    width_shift_range=0.15, #horizontal shift
    height_shift_range=0.15, #vertical shift
    rotation_range=30, #rotation
    rescale=1./255, #rescale Our original images consist in RGB coefficients in the 0-255, but such values would be too high for 
                    #our model to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255. 
    zoom_range=0.2, #zoom
    horizontal_flip=True, #horizontal flip
    brightness_range=[0.2,1.7], #brightness
    fill_mode='nearest', #fill the area in black color. You can change the color by giving a value for
)

#Množina testovacích dat
datadef_test = datadef_train = image.ImageDataGenerator(
    rescale=1./255, #rescale
)

train_img_gen = datadef_train.flow_from_directory(
        'C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_A_retrain/train',  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 250x250
        batch_size=32, #32 randomly selected images from across the classes in the dataset will be returned in each batch when training
        class_mode='binary') #detail of only binary classes

test_img_gen = datadef_test.flow_from_directory(
        'C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_A_retrain/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 150
batch_size = 16

img_width, img_height = 150, 150

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

"""
loadování modelu
"""

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

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

"""
fit modelu
"""

model.fit_generator(
    train_img_gen,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_img_gen,
    validation_steps=nb_validation_samples // batch_size)


"""
uložení modelu
"""
model.save_weights('C:/Users/Danie/Desktop/Grinding/data_projekt/ModelA/final_retrain.h5')

model.save('C:/Users/Danie/Desktop/final_2.h5')