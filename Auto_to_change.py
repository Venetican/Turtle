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
    host="localhost",
    user="69278_wp2",
    password="x0psu5j5ofr",
    database="nfturtle_io_wp2")
    global cursor 
    cursor = mydb.cursor()
    global df
    df = pd.read_sql_query("SELECT * FROM nfturtle_io_wp2.form_responses ORDER BY entry_id ",mydb)
    #dfs nutné k identifikaci a zpracování záznamu
    global relevant_rows_initial 
    relevant_rows_initial = df[(df["processed"]=="0") & (df["email"].notnull()) & (df["upload_path"].notnull())]
    global relevant_rows_is_turtle_part
    relevant_rows_is_turtle_part = df[(df["processed"]=="1") & (df["os_path_of_all_pics"].notnull())]
    global relevant_rows_unique_part 
    relevant_rows_unique_part = df[(df["is_turtle"]=="1") & (df["os_path_of_all_pics"].notnull())]
    global relevant_rows_path_of_turtles
    relevant_rows_path_of_turtles = df[(df["is_unique"]=="1") & (df["os_path_of_turtles"].notnull())]

def db_conn_for_emails():
    global mydb
    mydb = mysql.connector.connect(
    host="localhost",
    user="69278_wp2",
    password="x0psu5j5ofr",
    database="nfturtle_io_wp2")
    global cursor 
    cursor = mydb.cursor()
    global df
    df = pd.read_sql_query("SELECT * FROM nfturtle_io_wp2.form_responses ORDER BY entry_id ",mydb)
    #dfs součástí rozesílky emailů
    global relevant_rows_for_unique_turtles
    relevant_rows_for_unique_turtles = df[(df["is_unique"]=="1") & (df["email_sent"] =="0")]
    global relevant_rows_for_non_unique_turtles
    relevant_rows_for_non_unique_turtles = df[(df["is_unique"]=="0") & (df["is_turtle"] =="1") & (df["email_sent"] =="0")]
    global relevant_rows_for_non_turtles
    relevant_rows_for_non_turtles = df[(df["is_unique"]=="0") & (df["is_turtle"] =="0") & (df["email_sent"] =="0")]

"""
funkce stahující fotky z databáze do složky určené k recognitionu
"""
db_conn()
def downloading_images_from_db():
    db_conn()
    for row in relevant_rows_initial.loc[:,"upload_path"]:
        time.sleep(3)
        urllib.request.urlretrieve(row,(f"C:/Users/Administrator/Desktop/Automatizace/Pics_to_check/{row[47:]}"))
        db_conn()
        dir_with_pics = "C:/Users/Administrator/Desktop/Automatizace/Pics_to_check/"
        cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.processed = 1 WHERE nfturtle_io_wp2.form_responses.upload_path='{row}'")
        time.sleep(3)
        cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.os_path_of_all_pics = '{dir_with_pics}\{row[47:]}' WHERE nfturtle_io_wp2.form_responses.upload_path='{row}'")
        mydb.commit()
        cursor.close()
        mydb.close()
        with open("C:/Users/Administrator/Desktop/Automatizace/event_log.txt", "a") as log:
            log.write(str(f"-------New image {row[47:]} was downloaded to C:/Users/Administrator/Desktop/Automatizace/Pics_to_check/\n"))

"""
funkce obsahující identifikaci želvy
"""
#funkce pro transformaci obrázku nutné pro model identifikující želvu
def prepare():
    db_conn()
    loaded_model_b = load_model('C:/Users/Administrator/Desktop/Automatizace/Turtle/ModelA/DSR_results/final_retrained_b.h5')
    for row in relevant_rows_is_turtle_part.loc[:,"os_path_of_all_pics"]:
        img = image.load_img(row, target_size=(150,150))
        x = image.img_to_array(img)
        x = x/255
        if loaded_model_b.predict(np.expand_dims(x, axis=0)) > 0.7:
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.is_turtle = 1 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
        else:
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.is_turtle = 0 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
    mydb.commit()
    cursor.close()
    mydb.close()


"""
funkce zaručující unikátnost jednotlivé fotky
"""

def unique():
    db_conn()
    for row in relevant_rows_unique_part.loc[:,"os_path_of_all_pics"]:
        db_conn()
        shutil.copy(row, f"C:/Users/Administrator/Desktop/Automatizace/tmp_folder/{row[37:]}")
        if pd.DataFrame(data = difPy.dif('C:/Users/Administrator/Desktop/Automatizace/only_unique_turtles','C:/Users/Administrator/Desktop/Automatizace/tmp_folder').result).empty:
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.is_unique = 1 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
            time.sleep(3)
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.os_path_of_turtles = 'C:/Users/Administrator/Desktop/Automatizace/only_unique_turtles/{row[37:]}' WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
            shutil.copy(row, f"C:/Users/Administrator/Desktop/Automatizace/only_unique_turtles/{row[36:]}")     
        else:
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.is_unique = 0 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
        mydb.commit()
        cursor.close()
        mydb.close()
        for root,dirs, files in os.walk("C:/Users/Administrator/Desktop/Automatizace/tmp_folder"):
            for file in files:
                os.remove(os.path.join(root, file))

"""
funkce přiřazující unikátní id_of_nft pro unikátní želvy
"""

def assign_nft_id_for_unique():
    db_conn()
    for row in relevant_rows_path_of_turtles.loc[:,"os_path_of_turtles"]:
        db_conn()
        df = pd.read_sql_query("SELECT max(nft_id)+1 FROM nfturtle_io_wp2.form_responses",mydb)
        id_to_assign = int(df.iloc[0,0]) 
        cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.nft_id = {id_to_assign} WHERE nfturtle_io_wp2.form_responses.os_path_of_turtles='{row}'")
        mydb.commit()
        cursor.close()
        mydb.close()

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
  msg["From"] = sender_email
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

#unikátní želvy
def sending_unique_turtles():
    db_conn_for_emails()
    for email in relevant_rows_for_unique_turtles.loc[:,"email"]:
        time.sleep(2)
        sending_email(html_unique,email)
        for row in relevant_rows_for_unique_turtles.loc[:,"os_path_of_all_pics"]:
            time.sleep(2)
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.email_sent = 1 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
    mydb.commit()
    cursor.close()
    mydb.close()

#neunikátní želvy
def sending_non_unique_turtles():
    db_conn_for_emails()
    for email in relevant_rows_for_non_unique_turtles.loc[:,"email"]:
        time.sleep(2)
        sending_email(html_non_unique,email)
        for row in relevant_rows_for_non_unique_turtles.loc[:,"os_path_of_all_pics"]:
            time.sleep(2)
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.email_sent = 1 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
    mydb.commit()
    cursor.close()
    mydb.close()

#non želvy
def sending_non_turtles():
    db_conn_for_emails()
    for email in relevant_rows_for_non_turtles.loc[:,"email"]:
        time.sleep(2)
        sending_email(html_no_turtle,email)
        for row in relevant_rows_for_non_turtles.loc[:,"os_path_of_all_pics"]:
            time.sleep(2)
            cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.email_sent = 1 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
    mydb.commit()
    cursor.close()
    mydb.close()


"""
test
"""
def db_conn_for_emails():
    global mydb
    mydb = mysql.connector.connect(
    host="localhost",
    user="69278_wp2",
    password="x0psu5j5ofr",
    database="nfturtle_io_wp2")
    global cursor 
    cursor = mydb.cursor()
    global df_inner
    df_inner = pd.read_sql_query("select * from nfturtle_io_wp2.nft_table a1 inner join nfturtle_io_wp2.form_responses a2 on a1.nft_id = a2.nft_id",mydb)

def sending_unique_turtles():
    db_conn_for_emails()
    for email in relevant_rows_for_unique_turtles.loc[:,"email"]:
        for upload_path in relevant_rows_for_unique_turtles.loc[:,"upload_path"]:
            for nft_pic in df_inner.loc[:,"url_pic"]:
                for nft_site in df_inner.loc[:,"url_site"]:
                    sending_email(html_unique.replace(replace_main_photo,upload_path).replace(),email)
                    time.sleep(2)
                    for row in relevant_rows_for_unique_turtles.loc[:,"os_path_of_all_pics"]:
                        time.sleep(2)
                        cursor.execute(f"UPDATE nfturtle_io_wp2.form_responses SET nfturtle_io_wp2.form_responses.email_sent = 1 WHERE nfturtle_io_wp2.form_responses.os_path_of_all_pics='{row}'")
    time.sleep(2)
    cursor.execute("UPDATE nfturtle_io_wp2.nft_table a1, nfturtle_io_wp2.form_responses a2 SET a1.is_paired = 1 WHERE a1.nft_id = a2.nft_id")
    mydb.commit()
    cursor.close()
    mydb.close()



"""
automatizovaný select relevantních záznamů z db 
jestliže není objeven nový záznam, je do logu zaznamenána informace a čas s informací o nulovém záznamu
jestliže je objeven nový záznam, tak pro každý řádek df 
"""
   
while True:
    db_conn()
    if relevant_rows_initial.empty:
        time.sleep(10)
        current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        with open("C:/Users/Administrator/Desktop/Automatizace/event_log.txt", "a") as log:
            log.write(str(f"{current_time} - No new records where found in target table during regular itteration.\n"))
    else:
        time.sleep(10)
        number_of_rows_in_df = relevant_rows_initial.shape[0]
        current_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        downloading_images_from_db()
        prepare()
        with open("C:/Users/Administrator/Desktop/Automatizace/event_log.txt", "a") as log:
            log.write(str(f"{current_time} - {number_of_rows_in_df} records were found during regular itteration.\n"))
        continue