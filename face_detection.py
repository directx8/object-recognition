import argparse
import imutils
import time
import dlib
import cv2

class FaceDetection:
    def __init__(self, upsampling=2):
        self.upsampling = upsampling

    def convert_and_trim_bb(self, image, rect):
        # extract the starting and ending (x, y)-coordinates of the
        # bounding box
        startX = rect.left()
        startY = rect.top()
        endX = rect.right()
        endY = rect.bottom()
        # ensure the bounding box coordinates fall within the spatial
        # dimensions of the image
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(endX, image.shape[1])
        endY = min(endY, image.shape[0])
        # compute the width and height of the bounding box
        w = endX - startX
        h = endY - startY
        # return our bounding box coordinates
        return (startX, startY, w, h)

    def face_blur(self, frame):
        detector = dlib.get_frontal_face_detector()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rects = detector(rgb, self.upsampling)
        # convert the resulting dlib rectangle objects to bounding boxes,
        # then ensure the bounding boxes are all within the bounds of the
        # input image
        boxes = [self.convert_and_trim_bb(frame, r) for r in rects]
        # loop over the bounding boxes
        for (x, y, w, h) in boxes:
            # draw the bounding box on our image
            try:
                frame[y:(y+h), x:(x+w)] = cv2.blur(frame[y:(y+h), x:(x+w)], (27,27))
            except Exception as e:
                #print(e)
                pass
        return frame