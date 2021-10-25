
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
import ocr



def local_dataset(local_path, id_):
    instances = []
    facial_img_path = local_path + id_ + '.jpg'

    # name and birth
    name, birth = ocr.ocr(facial_img_path)

    # detect and align
    facial_img = detect_align.preprocess_face(facial_img_path, info=True, target_size=(160, 160))

    # represent
    embedding = model.predict(facial_img)[0]

    # store
    instance = []
    instance.append(id_)
    instance.append(name)
    instance.append(birth)
    instance.append(embedding)
    instances.append(instance)

    df = pd.DataFrame(instances, columns=["id", "name", "birth", "embedding"])
    return df


def store(df):
    for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
        ID = instance["id"]
        name = instance["name"]
        birth = instance["birth"]
        embeddings = instance["embedding"]

        insert_statement = "INSERT INTO face_meta (ID,name,birth EMBEDDING) VALUES (?, ?, ?, ?)"
        insert_args = (ID,name,birth, embeddings.tobytes())
        cursor.execute(insert_statement, insert_args)

        for i, embedding in enumerate(embeddings):
            insert_statement = "INSERT INTO face_embeddings (FACE_ID, DIMENSION, VALUE) VALUES (?, ?, ?)"
            insert_args = (index, i, str(embedding))
            cursor.execute(insert_statement, insert_args)
    conn.commit()


# ### main

# In[5]:


if __name__ == "__main__":
    local_path = "/project/img/"
    model = facenet.loadModel()
    df = local_dataset(local_path,'BBB')
    conn = sqlite3.connect('pre_info.db')
    cursor = conn.cursor()

    cursor.execute('''drop table if exists face_meta ''')
    cursor.execute('''drop table if exists face_embeddings''')
    cursor.execute('''create table face_meta (ID VARCHAR(10) primary key, name VARCHAR(10) , birth INT ,EMBEDDING BLOB)''')
    cursor.execute('''create table face_embeddings (FACE_ID INT, DIMENSION INT, VALUE DECIMAL(5, 30))''')

    store(df)

