#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from PIL import Image
import math
import time

import facenet
import detect_align


# ### Database
# - local 사용을 위해 sqlite 사용 
# - preinfo.db

# #### 함수설명
# - local_dataset<br> local data를 network에 넣어 나온 feature vector를 database에 stroe하기 위해 processing
# - store<br> local data를 database에 담음

# In[3]:


def local_dataset(local_path):
    facial_img_paths = []
    ids=[]
    instances = []

    for root, directory, files in os.walk(local_path):
        for file in files:
            if '.jpg' in file:
                facial_img_paths.append(root+"/"+file)
                ids.append(file[:(len(file)-4)])
    for i in tqdm(range(0, len(facial_img_paths))):
        facial_img_path = facial_img_paths[i]
        id= ids[i]
        #detect and align
        facial_img = detect_align.preprocess_face(facial_img_path, target_size = (160, 160))

        #represent
        embedding = model.predict(facial_img)[0]

        #store
        instance = []
        instance.append(id)
        instance.append(embedding)
        instances.append(instance)

    df = pd.DataFrame(instances, columns = ["id", "embedding"])
    return df

def store(df):
    for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
        ID = instance["id"]
        embeddings = instance["embedding"]

        insert_statement = "INSERT INTO face_meta (ID, EMBEDDING) VALUES (?, ?)"
        insert_args = (ID, embeddings.tobytes())
        cursor.execute(insert_statement, insert_args)

        for i, embedding in enumerate(embeddings):
            insert_statement = "INSERT INTO face_embeddings (FACE_ID, DIMENSION, VALUE) VALUES (?, ?, ?)"
            insert_args = (index,i, str(embedding))
            cursor.execute(insert_statement, insert_args)
    conn.commit()


# ### main

# In[5]:


if __name__ == "__main__":
    local_path="/project/img/"
    model=facenet.loadModel()
    df=local_dataset(local_path)
    conn = sqlite3.connect('pre_info.db')
    cursor = conn.cursor()
    """
    cursor.execute('''drop table if exists face_meta ''')
    cursor.execute('''drop table if exists face_embeddings''')
    cursor.execute('''create table face_meta (ID VARCHAR(10) primary key, EMBEDDING BLOB)''')
    cursor.execute('''create table face_embeddings (FACE_ID INT, DIMENSION INT, VALUE DECIMAL(5, 30))''')
    """

    store(df)
    

"""
# #### * Clinent sever에서 가지고 있는 embedding form

# In[6]:


    tic = time.time()

    select_statement = "select ID, embedding from face_meta"
    results = cursor.execute(select_statement)

    instances = []
    for result in results:
        img_name = result[0]
        embedding_bytes = result[1]
        embedding = np.frombuffer(embedding_bytes, dtype = 'float32')

        instance = []
        instance.append(img_name)
        instance.append(embedding)
        instances.append(instance)

    toc = time.time()
    print(toc-tic,"seconds")

    db_form = pd.DataFrame(instances, columns = ["ID", "embedding"])
    print(db_form)
"""

