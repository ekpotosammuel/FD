import numpy as np
import face_recognition as fr
import cv2
import os

path = 'db'
images = []
className = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    className.append(os.path.splitext(cl)[0])
print(className)

def findEcondngs(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)
        encodeList.append(encode)
    return encodeList
encodeListknown = findEcondngs(images)
print('loading complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = fr.face_locations(imgS)
    encodeCurFrame = fr.face_encodings(imgS,facesCurFrame)

    for encodeFace, faceloc in zip(encodeCurFrame,facesCurFrame):
        matches = fr.compare_faces(encodeListknown,encodeFace)
        faceDis =fr.face_distance(encodeListknown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = className[matchIndex.upper()]

    cv2.imshow('Webcam_facerecognition', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



