from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2

class TextDetection:
	def __init__(self, net="models/frozen_east_text_detection.pb", min_confidence=0.4):
		self.layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
		self.net = net
		self.min_confidence = min_confidence

	def decode_predictions(self, scores, geometry):
		# grab the number of rows and columns from the scores volume, then
		# initialize our set of bounding box rectangles and corresponding
		# confidence scores
		(numRows, numCols) = scores.shape[2:4]
		rects = []
		confidences = []
		# loop over the number of rows
		for y in range(0, numRows):
			# extract the scores (probabilities), followed by the
			# geometrical data used to derive potential bounding box
			# coordinates that surround text
			scoresData = scores[0, 0, y]
			xData0 = geometry[0, 0, y]
			xData1 = geometry[0, 1, y]
			xData2 = geometry[0, 2, y]
			xData3 = geometry[0, 3, y]
			anglesData = geometry[0, 4, y]
			# loop over the number of columns
			for x in range(0, numCols):
				# if our score does not have sufficient probability,
				# ignore it
				if scoresData[x] < self.min_confidence:
					continue
				# compute the offset factor as our resulting feature
				# maps will be 4x smaller than the input image
				(offsetX, offsetY) = (x * 4.0, y * 4.0)
				# extract the rotation angle for the prediction and
				# then compute the sin and cosine
				angle = anglesData[x]
				cos = np.cos(angle)
				sin = np.sin(angle)
				# use the geometry volume to derive the width and height
				# of the bounding box
				h = xData0[x] + xData2[x]
				w = xData1[x] + xData3[x]
				# compute both the starting and ending (x, y)-coordinates
				# for the text prediction bounding box
				endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
				endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
				startX = int(endX - w)
				startY = int(endY - h)
				# add the bounding box coordinates and probability score
				# to our respective lists
				rects.append((startX, startY, endX, endY))
				confidences.append(scoresData[x])
		# return a tuple of the bounding boxes and associated confidences
		return (rects, confidences)

	def text_blur(self, frame):
		# initialize the original frame dimensions, new frame dimensions,
		# and ratio between the dimensions
		(W, H) = (None, None)
		(newW, newH) = (320,320)
		(rW, rH) = (None, None)

		# load the pre-trained EAST text detector
		network = cv2.dnn.readNet(self.net)
		# resize the frame, maintaining the aspect ratio
		frame = imutils.resize(frame, width=1000)
		orig = frame.copy()
		# if our frame dimensions are None, we still need to compute the
		# ratio of old frame dimensions to new frame dimensions
		if W is None or H is None:
			(H, W) = frame.shape[:2]
			rW = W / float(newW)
			rH = H / float(newH)
		# resize the frame, this time ignoring aspect ratio
		frame = cv2.resize(frame, (newW, newH))
			# construct a blob from the frame and then perform a forward pass
		# of the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		network.setInput(blob)
		(scores, geometry) = network.forward(self.layerNames)
		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = self.decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		# loop over the bounding boxes
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
			try:
				# draw the bounding box on the frame
				orig[startY:endY, startX:endX] = cv2.blur(orig[startY:endY, startX:endX], (27 , 27))
			except Exception as e:
				#print(e)
				pass
		return orig