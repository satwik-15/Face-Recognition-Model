import cv2
import numpy as np
import face_recognition
import os

path = '#Location on your PC#'
images = []
Names = []
myList = os.listdir(path)

for i in myList:
    curImg = cv2.imread(f'{path}/{i}')
    images.append(curImg)
    Names.append(os.path.splitext(i)[0])
print("List of Names whose Faces are there in my Database :- ")
print(Names)

def findEncodings(images):
    eList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        eList.append(encode)
    return eList
eListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    eCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(eCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(eListKnown, encodeFace)
        faceDist = face_recognition.face_distance(eListKnown, encodeFace)
        #print(faceDist)
        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = Names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 12 , y2 - 8), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 255, 255), 2)

        else:
            name ="Unknown"
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 12, y2 - 8), cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)



