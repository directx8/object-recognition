import cv2
import mediapipe as mp
from face_detection import FaceDetection
from text_detection import TextDetection
import numpy as np

# Load face landmarks
fl = FaceDetection()
td = TextDetection()

# LOad the webcam stream
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Webcam is not accessable")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA) #resize video by 0.75 the original size

    # call blurring func
    result_face = fl.blur_facial_features(frame)
    result = td.text_blur(result_face)
    key = cv2.waitKey(1)

    cv2.imshow('result', result)

    # If you press the esc key you exit
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()


