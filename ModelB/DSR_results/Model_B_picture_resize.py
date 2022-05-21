"""
Model B - rámečkování hlavy - test ' ' ' 
"""

"""
nahrání potřebných balíčků
"""
import numpy as np
import pandas as pd
import cv2
import os
import fnmatch

"""
loadování dat
"""
df_index = pd.read_csv('C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/labels.csv', header=None)
df_index.columns = ['Turtle','left_x','left_y','bwidth','blenght','filename','imwidth','imlenght']
#což znamená (x,y,)
print(df_index.head())
print(df_index.count())

"""
Zbavení se duplicit
dle dokumentace jich má býti 2000, což znamená, že jich zde je 103 double, duplicity
"""
print(df_index.duplicated(subset='filename').sum()) #duplicity check 103
df_index = df_index.drop_duplicates(subset='filename', keep='last') #drop duplicit s podmínkou, že první nález bude zachován
print(df_index.duplicated(subset='filename').sum()) #duplicity check 0
print(df_index.count()) #v tuto chvíli přesně 2000 obrázků

"""
přidružení YOLO souřadnic k datasetu
"""
"""způsob 1"""
data_path = 'C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/'
print(df_index["filename"][0][:-3])

list = []
for file in os.listdir(data_path + "labels"):
    if fnmatch.fnmatch(file, '*.txt'):
        list.append(file)
list.sort()
print(list)

"""způsob 2"""   

labelname = "C:/Users/Danie/Desktop/Grinding/data_projekt/Data/model_B/labels/" + df_index["filename"][0][:-3] + 'txt'
df_yolo = pd.read_csv(labelname, delim_whitespace=True, header=None)
df_yolo.columns = ['idd', 'yolo_x', 'yolo_y','yolo_width', 'yolo_length']
print(df_yolo)

for imageName in df_index["filename"][1:]:
    if (imageName[-4] == '.'):
        # some images have ".jpeg" instead of ".jpg"
        labelName = data_path + "labels/" + imageName[:-3] + 'txt'
    else:
        labelName = data_path + "labels/" + imageName[:-4] + 'txt'
    #print (labelName)
    bb_txt = pd.read_csv(labelName, delim_whitespace=True, header=None)
    bb_txt.columns = ['idd', 'yolo_x', 'yolo_y','yolo_width', 'yolo_length'] 
    df_yolo = pd.concat([df_yolo, bb_txt.iloc[[0]]], ignore_index=True)

df_yolo.head(10)

df_index.reset_index(drop=True, inplace=True)
df_yolo.reset_index(drop=True, inplace=True)

df_all = pd.concat([df_index, df_yolo], axis=1)

df_all.head()




