"""
Pouze model k dodatečnému testování
"""

"""
import balíčků
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import h5py
from tensorflow.keras.models import load_model

"""
load modelu
"""

loaded_model_a = load_model('C:/Users/Danie/Desktop/Grinding/dp_git/ModelA/DSR_results/final_retrained_a.h5')
loaded_model_b = load_model('C:/Users/Danie/Desktop/Grinding/dp_git/ModelA/DSR_results/final_retrained_b.h5')

"""
funkce k testování
"""

def prepare(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    x = image.img_to_array(img)
    x = x/255
    return np.expand_dims(x, axis=0)

"""
výsledný print
"""
loaded_model_a.predict(prepare('C:/Users/Danie/Desktop/er.jpeg'))
loaded_model_b.predict(prepare('C:/Users/Danie/Desktop/er.jpeg'))

#test modelu A
if loaded_model_a.predict(prepare('C:/Users/Danie/Desktop/280282566_387383293310529_1763940945273802682_n.jpg')) > 0.7:
    print("it is a turtle")
else:
    print("it is not a turtle")

#test modelu B
if loaded_model_b.predict(prepare('C:/Users/Danie/Desktop/280282566_387383293310529_1763940945273802682_n.jpg')) > 0.7:
    print("it is a turtle")
else:
    print("it is not a turtle")