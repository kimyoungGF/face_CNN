import numpy as np
import cv2



xml = 'haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(xml)

face_num=1



cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)

#---------------얼굴 박스---------------------
    if len(faces)==1:
        for (x, y, w, h) in faces:


            cut_frame = cv2.resize(frame[y:y+h, x:x+w],(250,250))


            cv2.imwrite('face_mask/face{0}.jpg'.format(face_num), cut_frame)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_num = face_num + 1

            print(str(face_num)+'장')
            print('{0},{1},{2},{3}'.format(x,x+w,y,y+h))

    cv2.imshow('result', frame)





cap.release()
cv2.destroyAllWindows()