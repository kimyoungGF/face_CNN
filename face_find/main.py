import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

img_height = 240
img_width = 200

class_names=['glasses', 'mask', 'nothing']

model = keras.models.load_model('./my_model')


def modelpass(img_face):

    img_array = tf.expand_dims(img_face, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return (int(100 * np.max(score)),class_names[np.argmax(score)])


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(frame, (220,120), (220+img_width,120+img_height), (255, 0, 0), 2)
    cv2.imshow('result', frame)

    if cv2.waitKey(1) == ord('c'):
        cut_frame = cv2.resize(frame[120:120 + img_height, 220:220 + img_width], (200, 240))
        gray = cv2.cvtColor(cut_frame, cv2.COLOR_BGR2GRAY)
        score, faceclass = modelpass(gray)
        str = ("{:.2f} % is {}".format(score, faceclass))
        print(str)


    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()