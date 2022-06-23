"""
import relevantních knihoven pro funkčnost automatizačního skriptu auto.py
"""
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import difPy
#SQL part
import mysql.connector
from mysql.connector import Error
#downloading pictures
import urllib.request
#balíček správne WIN cesty
from pathlib import Path
import shutil
#balíčky pro zasílání emailu
import smtplib, ssl, email
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from html_unique import html_unique
from html_unique import replace_main_photo
from html_unique import replace_nft_pic
from html_unique import replace_nft_odkaz
from html_no_turtle import html_no_turtle
from html_non_unique import html_non_unique

"""
workaround pro lepší performance
"""
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
připojení k sql databázi, select relevantních záznamů pro jednotlivé fáze a jejich uložení do pd.dfs
"""
def db_conn():
    global mydb
    mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Xxsplit88",
    database="test_1")
    global cursor 
    cursor = mydb.cursor()
    global df
    df = pd.read_sql_query("SELECT * FROM test_1.form_responses ORDER BY entry_id ",mydb)
    #dfs nutné k identifikaci a zpracování záznamu
    global relevant_rows_initial 
    relevant_rows_initial = df[(df["processed"]=="0") & (df["email"].notnull()) & (df["upload_path"].notnull())]
    global relevant_rows_is_turtle_part
    relevant_rows_is_turtle_part = df[(df["processed"]=="1") & (df["email_sent"]=="0")]
    global relevant_rows_unique_part
    relevant_rows_unique_part = df[(df["is_turtle"]=="1") & (df["email_sent"]=="0")]
    global relevant_rows_path_of_turtles
    relevant_rows_path_of_turtles = df[(df["is_unique"]=="1") & (df["email_sent"]=="0")]


def db_conn_for_emails():
    global mydb
    mydb = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Xxsplit88",
    database="test_1")
    global cursor 
    cursor = mydb.cursor()
    global df
    df = pd.read_sql_query("SELECT * FROM test_1.form_responses ORDER BY entry_id ",mydb)
    global relevant_rows_for_unique_turtles
    relevant_rows_for_unique_turtles = pd.read_sql_query("""select * from test_1.nft_table a1 inner join test_1.form_responses a2 on a1.nft_id = a2.nft_id 
                                                        where email_sent=0""",mydb)
    global relevant_rows_for_non_unique_turtles
    relevant_rows_for_non_unique_turtles = df[(df["is_unique"]=="0") & (df["is_turtle"] =="1") & (df["email_sent"] =="0")]
    global relevant_rows_for_non_turtles
    relevant_rows_for_non_turtles = df[(df["is_unique"]=="0") & (df["is_turtle"] =="0") & (df["email_sent"] =="0")]
    
    

"""
funkce stahující fotky z databáze do složky určené k recognitionu
"""
def downloading_images_from_db():
    db_conn()
    number_of_rows_in_df = relevant_rows_initial.shape[0]
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
        log.write(str(f"PHASE 1 DOWNLOADING IMAGES\n{current_time} - {number_of_rows_in_df} records were found during regular itteration.\n"))
    for row in relevant_rows_initial.loc[:,"upload_path"]:
        time.sleep(3)
        urllib.request.urlretrieve(row,(f"C:/Users/Danie/Desktop/Pics_to_check/{row[47:]}"))
        db_conn()
        dir_with_pics = "C:/Users/Danie/Desktop/Pics_to_check/"
        cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.processed = 1 WHERE test_1.form_responses.upload_path='{row}'")
        time.sleep(3)
        cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.os_path_of_all_pics = '{dir_with_pics}{row[47:]}' WHERE test_1.form_responses.upload_path='{row}'")
        mydb.commit()
        cursor.close()
        mydb.close()
        with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
            log.write(str(f"-------New image {row[47:]} was downloaded to C:/Users/Danie/Desktop/Pics_to_check/\n"))
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
            log.write(str(f"END OF PHASE 1\n"))

"""
funkce obsahující identifikaci želvy
"""
#funkce pro transformaci obrázku nutné pro model identifikující želvu
def prepare():
    db_conn()
    loaded_model_b = load_model('C:/Users/Danie/Desktop/Grinding/dp_git/ModelA/DSR_results/final_retrained_b.h5')
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    number_of_rows_in_df = relevant_rows_is_turtle_part.shape[0]
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
        log.write(str(f"PHASE 2 TURTLE RECOGNITION\n{current_time} - {number_of_rows_in_df} are beining proccessed by Turtle recognition algorithm\n"))
    for row in relevant_rows_is_turtle_part.loc[:,"os_path_of_all_pics"]:
        img = image.load_img(row, target_size=(150,150))
        x = image.img_to_array(img)
        x = x/255
        if loaded_model_b.predict(np.expand_dims(x, axis=0)) > 0.7:
            cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.is_turtle = 1 WHERE test_1.form_responses.os_path_of_all_pics='{row}'")
            with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------image {row} was validated as a turtle image\n"))
        else:
            cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.is_turtle = 0 WHERE test_1.form_responses.os_path_of_all_pics='{row}'")
            with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------Image {row} was validated as a non turtle image\n"))
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
            log.write(str(f"END OF PHASE 2\n"))
    mydb.commit()
    cursor.close()
    mydb.close()

"""
funkce zaručující unikátnost jednotlivé fotky
"""

def unique():
    db_conn()
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    number_of_rows_in_df = relevant_rows_unique_part.shape[0]
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
        log.write(str(f"PHASE 3 UNIQUE TURTLE RECOGNITION\n{current_time} - {number_of_rows_in_df} are beining proccessed by Turtle unique recognition algorithm\n"))
    for row in relevant_rows_unique_part.loc[:,"os_path_of_all_pics"]:
        db_conn()
        shutil.copy(row, f"C:/Users/Danie/Desktop/tmp_folder/{row[37:]}")
        if pd.DataFrame(data = difPy.dif('C:/Users/Danie/Desktop/only_unique_turtles','C:/Users/Danie/Desktop/tmp_folder').result).empty:
            cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.is_unique = 1 WHERE test_1.form_responses.os_path_of_all_pics='{row}'")
            time.sleep(3)
            cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.os_path_of_turtles = 'C:/Users/Danie/Desktop/only_unique_turtles/{row[37:]}' WHERE test_1.form_responses.os_path_of_all_pics='{row}'")
            shutil.copy(row, f"C:/Users/Danie/Desktop/only_unique_turtles/{row[36:]}")
            with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------Image {row[37:]} in {row[:38]} was validated as unique and was placed to C:/Users/Danie/Desktop/tmp_folder/\n"))     
        else:
            cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.is_unique = 0 WHERE test_1.form_responses.os_path_of_all_pics='{row}'")
            with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------Image {row[37:]} in {row[:37]} was validated as non unique\n"))  
        mydb.commit()
        cursor.close()
        mydb.close()
        for root,dirs, files in os.walk("C:/Users/Danie/Desktop/tmp_folder"):
            for file in files:
                os.remove(os.path.join(root, file))
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
            log.write(str(f"END OF PHASE 3\n"))

"""
funkce přiřazující unikátní id_of_nft pro unikátní želvy
"""

def assign_nft_id_for_unique():
    db_conn()
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    number_of_rows_in_df = relevant_rows_path_of_turtles.shape[0]
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"PHASE 4 ASSIGNING UNIQUE NFT_ID TO UNIQUE TURTLES\n{current_time} - {number_of_rows_in_df} to be assigned\n"))
    for row in relevant_rows_path_of_turtles.loc[:,"os_path_of_turtles"]:
        db_conn()
        df = pd.read_sql_query("SELECT max(nft_id)+1 FROM test_1.form_responses",mydb)
        id_to_assign = int(df.iloc[0,0]) 
        cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.nft_id = {id_to_assign} WHERE test_1.form_responses.os_path_of_turtles='{row}'")
        with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------Image {row[37:]} got paired with nft_id={id_to_assign}\n"))
        mydb.commit()
        cursor.close()
        mydb.close()
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
            log.write(str(f"END OF PHASE 4\n"))

"""
funkce zasílající email
"""
def sending_email(html_pick,reciever):
  sender_email = "venetiwow@gmail.com"
  receiver_email = reciever
  password = "udshfytealeyfann"
  #Create MIMEMultipart object
  msg = MIMEMultipart("related")
  msg["Subject"] = "multipart test"
  msg["Fom"] = sender_email
  msg["To"] = receiver_email
  #HTML Message Part
  html = html_pick
  part = MIMEText(html, "html")
  msg.attach(part)
  
  # Create secure SMTP connection and send email
  context = ssl.create_default_context()
  with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
      server.login(sender_email, password)
      server.sendmail(
          sender_email, receiver_email, msg.as_string()
      )

"""
funkce automaticky zasílající maily
"""

def sending_unique_turtles():
    db_conn_for_emails()
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    number_of_rows_in_df = relevant_rows_for_unique_turtles.shape[0]
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"PHASE 5 Sending_emails\n{current_time} - {number_of_rows_in_df} emails to be sent as unique emails\n"))
    for index, row in relevant_rows_for_unique_turtles.iterrows():
        sending_email(html_unique.replace(replace_main_photo,row.values[10]).replace(replace_nft_pic,row.values[3]).replace(replace_nft_odkaz, row.values[2]),row.values[9])
        time.sleep(2)
        cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.email_sent = 1 WHERE test_1.form_responses.os_path_of_all_pics='{row.values[-2]}'")
        with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------Unique email was sent to {row.values[9]} with NFT_id{row.values[0]}\n"))
    time.sleep(2)
    cursor.execute("UPDATE test_1.nft_table a1, test_1.form_responses a2 SET a1.is_paired = 1 WHERE a1.nft_id = a2.nft_id")
    mydb.commit()
    cursor.close()
    mydb.close()

def sending_non_unique_turtles():
    db_conn_for_emails()
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    number_of_rows_in_df = relevant_rows_for_non_unique_turtles.shape[0]
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"{current_time} - {number_of_rows_in_df} emails to be sent as non-unique emails\n"))
    for index, row in relevant_rows_for_non_unique_turtles.iterrows():
        sending_email(html_non_unique.replace(replace_main_photo,row.values[6]),row.values[5])
        time.sleep(2)
        cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.email_sent = 1 WHERE test_1.form_responses.os_path_of_all_pics='{row.values[-2]}'")
        with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------Non-unique email was sent to {row.values[5]}\n"))
    time.sleep(2)
    mydb.commit()
    cursor.close()
    mydb.close()

def sending_non_turtles():
    db_conn_for_emails()
    current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    number_of_rows_in_df = relevant_rows_for_non_turtles.shape[0]
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"{current_time} - {number_of_rows_in_df} emails to be sent as non-turtle emails\n"))
    for index, row in relevant_rows_for_non_turtles.iterrows():
        sending_email(html_no_turtle,row.values[5])
        time.sleep(2)
        cursor.execute(f"UPDATE test_1.form_responses SET test_1.form_responses.email_sent = 1 WHERE test_1.form_responses.os_path_of_all_pics='{row.values[-2]}'")
        with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
                log.write(str(f"-------Non-turtle email was sent to {row.values[5]}\n"))
    mydb.commit()
    cursor.close()
    mydb.close()
    with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
            log.write(str(f"END OF PHASE 5\n"))


"""
automatizovaný select relevantních záznamů z db 
jestliže není objeven nový záznam, je do logu zaznamenána informace a čas s informací o nulovém záznamu
jestliže je objeven nový záznam, tak pro každý řádek df 
"""
   
while True:
    db_conn() #initial conect
    if relevant_rows_initial.empty:
        time.sleep(10)
        current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        with open("C:/Users/Danie/Desktop/event_log.txt", "a") as log:
            log.write(str(f"{current_time} - No new records where found in target table during regular itteration.\n"))
    else:
        time.sleep(10)
        downloading_images_from_db()
        time.sleep(5)
        prepare()
        time.sleep(5)
        unique()
        time.sleep(5)
        assign_nft_id_for_unique()
        time.sleep(5)
        sending_unique_turtles()
        time.sleep(5)
        sending_non_unique_turtles()
        time.sleep(5)
        sending_non_turtles()
        continue