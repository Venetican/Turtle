"""
Nahrání potřebných balíčků k modelu B
"""
def import_libr():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import h5py
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
    from tensorflow.keras import backend as K
    import h5py
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

