import cv2
import mediapipe as mp
from facial_landmarks import FaceLandmarks
import numpy as np
# Load face landmarks
fl = FaceLandmarks()

# LOad the video
cap = cv2.VideoCapture("forth_Trim.mp4")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5) #resize video
    frame_copy = frame.copy()
    height, width, _ = frame.shape

    # 1. Face landmarks detection
    landmarks = fl.get_facial_landmarks(frame) # get the landmarks of the face
    convexhull = cv2.convexHull(landmarks) # get only the outer landmarks

    # 2. Face blurring
    mask = np.zeros((height, width), np.uint8)
    # cv2.polylines(mask, [convexhull], True, 255, 3) # graw a polygone from the outer landmarks
    cv2.fillConvexPoly(mask, convexhull, 255) # White is allowed and black is thrown away

    # Extract the face
    frame_copy = cv2.blur(frame_copy, (37,37))
    face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask) # extract the face using bitwise operations

    # Extract background
    background_mask = cv2.bitwise_not(mask) # oposite of face extract
    background = cv2.bitwise_and(frame, frame, mask=background_mask)

    # Final result
    result = cv2.add(background, face_extracted)

    cv2.imshow("Result", result)
    key = cv2.waitKey(1)

    # If you press the esc key you exit
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
