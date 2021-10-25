#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sqlite3

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import time

import facenet
import detect_align
import cam
import distance



# ### 함수 설명
# - get_fv<br>
# wabcam에서 받은 영상의 feature vecotor를 구함
# - preinfo <br>
# 데이터베이스에서 입력한 ID에 대한 fearture vector를 return<br>
# - verification <br>
# get_fv에서 feature vector(target)와 preinfo로 받은 feature vector(source) 사이의 거리를 구하여 verification

# In[2]:


def get_fv(cam_path):
    cam.webcam(cam_path)
    img=detect_align.preprocess_face(cam_path, target_size = (160, 160))
    target = model.predict(img)[0] 
    return target


def load_preinfo(ID):
    tic = time.time()

    select_statement = "select ID,name, embedding from face_meta where ID = " + "'" + ID + "'"
    results = cursor.execute(select_statement)
    embedding = []
    for result in results:
        name = result[1]
        embedding_bytes = result[2]
        embedding = np.frombuffer(embedding_bytes, dtype='float32')

    return name,embedding

def verify(ID): 
    img_path="/project/img/"+ID+ ".jpg"
    cam_path="/project/cam/"+ID+ ".jpg"
    name, source = load_preinfo(ID)
    target = get_fv(cam_path) 
    
    #------------------------------
    #display
    
    fig = plt.figure(figsize = (10, 10))

    img = Image.open(img_path)
    w, h = img.size
    h1 = int(h * 0.1)
    h2 = int(h * 0.7)
    w1 = int(w * 0.62)
    w2 = int(w * 0.96)
    img = img.crop([w1, h1, w2, h2])
    
    ax1 = fig.add_subplot(1,2,1)
    plt.axis('off')
    plt.imshow(img)
    
    ax2 = fig.add_subplot(1,2,2)
    plt.axis('off')
    plt.imshow(Image.open(cam_path))
    
    plt.show()
    
    #------------------------------
    #verification

    print("ID = " + ID)
    print("=====================")
    print(" VERIFICATION RESULT")
    print("---------------------")
    
    distance.verify(source, target)
    print("===================== \n")


# ### main

# In[3]:


if __name__ == "__main__":
    conn = sqlite3.connect('pre_info.db')
    cursor = conn.cursor()
    model=facenet.loadModel()
    verify("BBB")


