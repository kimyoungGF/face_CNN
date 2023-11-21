import numpy as np
import cv2

face_num=1

capmod=0
#nothing=0, glass=1, mask=2

cap = cv2.VideoCapture(cv2.CAP_DSHOW+0)
cap.set(3, 640)
cap.set(4, 480)

w=200
h=240


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (220,120), (220+w,120+h), (255, 0, 0), 2)
    cv2.imshow('result', frame)
    cut_frame = cv2.resize(frame[120:120 + h, 220:220 + w], (200, 240))

    if cv2.waitKey(1) == ord('c'):
        if capmod==0:
            cv2.imwrite('./face/nothing/face_nothing{0}.jpg'.format(face_num), cut_frame)
        elif capmod==1:
            cv2.imwrite('./face/glasses/glasses{0}.jpg'.format(face_num), cut_frame)
        else:
            cv2.imwrite('./face/mask/face_mask{0}.jpg'.format(face_num), cut_frame)
        print(str(face_num) + '장')
        face_num = face_num + 1
    if cv2.waitKey(1) == ord('q'):
        break





cap.release()
cv2.destroyAllWindows()