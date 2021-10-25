#!/usr/bin/env python
# coding: utf-8


from pymongo import MongoClient
import warnings

import pandas as pd
from tqdm import tqdm


import detect_align
import facenet
import ocr


# 함수설명
# - local_dataset<br> local data를 network에 넣어 나온 feature vector를 database에 stroe하기 위해 processing
# - store<br> local data를 database에 담음


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


def insertInfo(df):

    infodb = client.Infodb
    userInfo = infodb.userInfo

    for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
        ID = instance["id"]
        name = instance["name"]
        birth = instance["birth"]
        embeddings = instance["embedding"].tobytes()
        user = {'_id': ID, 'name': name, 'birth': birth, 'embeddings': embeddings}
        try:
            userInfo.insert_one(user)
        except:
            print('ID already exists.')


# main

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # now = dt.datetime.now()
    local_path = "/project/img/"
    model = facenet.loadModel()
    df = local_dataset(local_path, "BBB")

    client = MongoClient('mongodb://localhost:27017/')

    insertInfo(df)

