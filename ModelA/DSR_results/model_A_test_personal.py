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

loaded_model = load_model('C:/Users/Danie/Desktop/Grinding/data_projekt/ModelA/final_2.h5')

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

if loaded_model.predict(prepare('<Zde vlož cestu k obrázku>')) > 0.7:
    print("it is a turtle")
else:
    print("it is not a turtle")