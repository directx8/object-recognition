import numpy as np
import argparse
import imutils
import time
import cv2


net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
cap = cv2.VideoCapture(0)

while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    _,frame = cap.read()

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.7:
            continue
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        y = startY - 10 if startY - 10 > 10 else startY + 10
        try:
            frame[startY:endY, startX:endX] = cv2.blur(frame[startY:endY, startX:endX], (27,27))
        except Exception as e:
            print(e)
            pass
        # cv2.rectangle(frame, (startX, startY), (endX, endY),
        #     (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
 
    # if the `ESC` key was pressed, break from the loop
    if key == 27:
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()