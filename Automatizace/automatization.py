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
import yagmail
from difPy import dif
import mysql.connector

"""
definice proměných součástí loopu
"""
f = open(r'c:\Users\Danie\Desktop\Automatizace\list_of_images.txt', 'r')
now = datetime.now()

current_time = datetime.now().strftime("%H:%M:%S")
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

def email_out_with_turtle():
    user = 'venetiwow@gmail.com'
    app_password = 'gjklmkqtzepscrfs'
    mezi = open("c:/Users/Danie/Desktop/Automatizace/list_of_emails.txt", "r")
    to = mezi.read()
    subject = 'Digi Turtle'
    contents = 'You Have just found unique turtle'
    with yagmail.SMTP(user, app_password) as yag:
        yag.send(to, subject, contents)
        print('Sent email successfully')

def email_out_without_turtle():
    user = 'venetiwow@gmail.com'
    app_password = 'gjklmkqtzepscrfs'
    mezi = open("c:/Users/Danie/Desktop/Automatizace/list_of_emails.txt", "r")
    to = mezi.read()
    subject = 'Digi Turtle'
    contents = 'We have not recognised your picture as a turtle.'
    with yagmail.SMTP(user, app_password) as yag:
        yag.send(to, subject, contents)
        print('Sent email successfully')

def email_out_with_duplicity_detected():
    user = 'venetiwow@gmail.com'
    app_password = 'gjklmkqtzepscrfs'
    mezi = open("c:/Users/Danie/Desktop/Automatizace/list_of_emails.txt", "r")
    to = mezi.read()
    subject = 'Digi Turtle'
    contents = ['We already have this photo in our database.']
    with yagmail.SMTP(user, app_password) as yag:
        yag.send(to, subject, contents,)
        print('Sent email successfully')

def model_A_predict():
        if loaded_model_b.predict(prepare(z)) > 0.7:
            email_out_with_turtle()
        else:
            email_out_without_turtle()
            

def same_photo_check():
    path_dir_to_compare = difPy.dif('C:/Users/Danie/Desktop/', show_output=False)
    dir_to_compare = path_dir_to_compare.result
    if bool(dir_to_compare) == True:
        email_out_with_duplicity_detected()
    else:
        model_A_predict()

def del_rows_in_txt_pic():
    a_file = open('c:/Users/Danie/Desktop/Automatizace/list_of_images.txt', "r")
    lines = a_file.readlines()
    a_file.close()
    new_file = open("c:/Users/Danie/Desktop/Automatizace/list_of_images.txt", "w")
    for line in lines:
        if line.strip("\n") != y:
            new_file.write(line)
    new_file.close()

def del_rows_in_txt_email():
    a_file = open('c:/Users/Danie/Desktop/Automatizace/list_of_emails.txt', "r")
    lines = a_file.readlines()
    a_file.close()
    new_file = open("c:/Users/Danie/Desktop/Automatizace/list_of_emails.txt", "w")
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
        time.sleep(40)
        print('Nothing New in time =', datetime.now().strftime("%H:%M:%S"))
    else:
        os.chdir('C:/Users/Danie/Desktop/Automatizace/new_image')
        result = sorted(filter(os.path.isfile, os.listdir('.')), key=os.path.getmtime, reverse=True)
        x = '\n'.join(map(str, result))
        y = x.partition('\n')[0]
        z = 'C:/Users/Danie/Desktop/Automatizace/new_image/'+ y
        same_photo_check()
        del_rows_in_txt_pic()
        continue

