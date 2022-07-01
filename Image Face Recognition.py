# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:10:32 2022

@author: Shivam
"""

#Image Detection
#______________________________________________________________________________________________________________________

import cv2
import numpy as np
import face_recognition

imgReal = face_recognition.load_image_file('C:/Users/Shivam/Desktop/My Files/My Files/Shivam2.jpg')
imgReal = cv2.cvtColor(imgReal,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('C:/Users/Shivam/Desktop/My Files/My Files/1363056845618.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgReal)[0]
encodeReal = face_recognition.face_encodings(imgReal)[0]
#print(faceLoc)
cv2.rectangle(imgReal,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]

results = face_recognition.compare_faces([encodeReal],encodeTest)
faceDist = face_recognition.face_distance([encodeReal],encodeTest)

print(results,faceDist)

cv2.putText(imgTest,f'{results}{round(faceDist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Real',imgReal)
cv2.imshow('Test',imgTest)
cv2.waitKey(0)