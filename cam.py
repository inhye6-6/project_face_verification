#!/usr/bin/env python
# coding: utf-8

# In[54]:


import datetime
import cv2

 
def webcam(img):
    capture = cv2.VideoCapture(0)


    while (capture.isOpened):

        ret, frame = capture.read()

        if ret == False:

            break

        cv2.imshow("VideoFrame", frame)



        now = datetime.datetime.now().strftime("%d_%H-%M-%S")

        key = cv2.waitKey(33) 



        if key == 27:  # esc

            break

        elif key == 26:  # ctrl + z

            cv2.IMREAD_UNCHANGED
            cv2.imwrite(img, frame)

    capture.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam_path = "/project/cam/liveness_ih.jpg"
    webcam(cam_path)
