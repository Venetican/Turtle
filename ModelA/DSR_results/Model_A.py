#import knihoven
from unicodedata import category
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.utils import shuffle 
import matplotlib.pyplot as plt
import cv2 
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import preprocessing

""""
legacy model 17-100
""""

#Pojmenování class_atributů a přiřazení indexů
class_names = ['Bear','Brown bear', 'Bull','Butterfly','Camel','Canary','Caterpillar','Cattle','Centipede','Cheetah','Chicken','Crab','Crocodile','Deer','Duck','Eagle','Elephant','Fish','Fox','Frog','Giraffe','Goat','Goldfish','Goose','Hamster','Harbor seal','Hedgehog','Hippopotamus','Horse','Jaguar','Jellyfish','Kangaroo','Koala','Ladybug','Leopard','Lion','Lizard','Lynx','Magpie','Monkey','Moths and butterflies','Mouse','Mule','Ostrich','Otter','Owl','Panda','Parrot','Penguin','Pig','Polar bear','Rabbit','Raccoon','Raven','Red panda','Rhinoceros','Scorpion','Sea lion','Sea turtle','Seahorse','Shark','Sheep','Shrimp','Snail','Snake','Sparrow','Spider','Squid','Squirrel','Starfish','Swan','Tick','Tiger','Tortoise','Turkey','Turtle','Whale','Woodpecker','Worm','Zebra']
class_label = {class_name:i for i, class_name in enumerate(class_names)}
print(class_label)

#loadování dat
def load_data() :
    input = r"C:\Users\Danie\Desktop\Grinding\data_projekt\Data\model_A"
    test_train_diff = ['train','test']

    output = []

    for cat in test_train_diff:
        path = os.path.join(input,cat)
        print(path)
        images = []
        labels = []

        print('načítám {}'.format(category))


        for folder in os.listdir(path):
            label = class_label[folder]

            for file in os.listdir(os.path.join(path,folder)):
                
                img_path = os.path.join(os.path.join(path,folder), file)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (150,150))

                images.append(image)
                labels.append(label)
                

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')
        
        output.append((images,labels))
    return output

(train_images, train_labels), (test_images,test_labels) = load_data()   
 
def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(30,30))
    for i in range(50):
        plt.subplot(15,15,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = cv2.resize(images[i], (30,30))
        plt.imshow(image.astype(np.uint8))
        plt.xlabel(class_names[labels[i]])
    plt.show()
display_examples(class_names, train_images, train_labels)        

model = Sequential([
    layers.Conv2D(32,(3,3), activation= 'relu', input_shape = (150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32,(3,3), activation= 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(80, activation=tf.nn.softmax)
])

model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=4, validation_split=0.2)


"""
model_2
transformace dat
    -> vytvoření pouze dvou podkategorií -> rest a turtle
    ->
"""

class_names = ['rest', 'turtle']
class_label = {class_name:i for i, class_name in enumerate(class_names)}
print(class_label)

def load_data() :
    input = r"C:\Users\Danie\Desktop\Grinding\data_projekt\Data\model_A_transform"
    test_train_diff = ['train','test']

    output = []

    for cat in test_train_diff:
        path = os.path.join(input,cat)
        print(path)
        images = []
        labels = []

        print('načítám {}'.format(category))


        for folder in os.listdir(path):
            label = class_label[folder]

            for file in os.listdir(os.path.join(path,folder)):
                
                img_path = os.path.join(os.path.join(path,folder), file)

                image = cv2.imread(img_path)
                image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (150,150))

                images.append(image)
                labels.append(label)
                

        images = np.array(images, dtype = 'float32')
        labels = np.array(labels, dtype = 'int32')
        
        output.append((images,labels))
    return output

(train_images, train_labels), (test_images,test_labels) = load_data()

def display_examples(class_names, images, labels):
    fig = plt.figure(figsize=(30,30))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        image = cv2.resize(images[i], (30,30))
        plt.imshow(image.astype(np.uint8))
        plt.xlabel(class_names[labels[i]])
    plt.show()
display_examples(class_names, train_images, train_labels)   

model = Sequential([
    layers.Conv2D(32,(3,3), activation= 'relu', input_shape = (150,150,3)),
    laye rs.MaxPooling2D(2,2),
    layers.Conv2D(32,(3,3), activation= 'relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.softmax)
])

model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=8, validation_split=0.2)

model = load_model('model.h5')

