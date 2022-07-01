# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:50:59 2022

@author: Shivam
"""

import numpy as np
import face_recognition
import os
import cv2 

path = 'C:/Users/shubh/OneDrive/Documents/Face Recognition Project/My Files/Dataset'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Done')

video_capture = cv2.VideoCapture(0)

while True:
    success,img = video_capture.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS)
    
    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDist) 
        accu = str(int(max(faceDist)*100))
        
        matchIndex = np.argmin(faceDist)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            name_accu = name + ',' + accu + '%'
            
        y1,x2,y2,x1 = faceLoc
        y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2+60,y2),(0,0,0),cv2.FILLED)
        cv2.putText(img,name_accu,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
    cv2.imshow('Video',img) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()