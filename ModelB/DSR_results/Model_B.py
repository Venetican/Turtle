"""
Model B - rámečkování hlavy - test ' ' ' 
"""

"""
nahrání potřebných balíčků
"""
import numpy as np
import pandas as pd
import cv2

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
Stanovení podmínek pro velikost obrázků a formátů
v základním csv je rámeček v pixelech (x1,y1,x2,y2)
mechanismus YOLO -> je procentuální část z celkové velikosti obrázku
"""

target_size = (150,150) #target velikosti obrázků






