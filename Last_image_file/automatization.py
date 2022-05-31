"""
import knihoven
"""
import pandas as pd
import numpy as np
import os
import h5py
import time
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
"""
definice proměných součástí loopu
"""
f = open(r'c:\Users\Danie\Desktop\Automatizace\list_of_images.txt', 'r')
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

loaded_model_b = load_model('C:/Users/Danie/Desktop/Grinding/dp_git/ModelA/DSR_results/final_retrained_b.h5')

"""
definice funkcí součástí loopu
"""
def prepare(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

def model_A_predict():
        if loaded_model_b.predict(prepare(z)) > 0.7:
            print("it is a turtle")
        else:
            print("it is not a turtle")

def del_rows_in_txt():
    a_file = open('c:/Users/Danie/Desktop/Automatizace/list_of_images.txt', "r")
    lines = a_file.readlines()
    a_file.close()
    new_file = open("c:/Users/Danie/Desktop/Automatizace/list_of_images.txt", "w")
    for line in lines:
        if line.strip("\n") != y:
            new_file.write(line)
    new_file.close()  

"""
automatizovaný loop
"""
while True:
    line = f.readline()
    if not line:
        time.sleep(60)
        print('Nothing New in time =', current_time)
    else:
        os.chdir('C:/Users/Danie/Desktop/Automatizace/new_image')
        result = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime, reverse=True)
        x = '\n'.join(map(str, result))
        y = x.partition('\n')[0]
        z = 'C:/Users/Danie/Desktop/Automatizace/new_image/'+ y
        model_A_predict()
        del_rows_in_txt()
        continue

