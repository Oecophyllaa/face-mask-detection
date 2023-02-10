# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from threading import Timer
from pymongo import MongoClient, collection
import mysql.connector
import numpy as np
import pandas as pd
import paho.mqtt.publish as publish
import paho
import imutils
import sched
import time
import cv2
import os
import pymongo

max_inflight_messages = 20

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), #224, 224 default
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# Background subtraction algortihm
backSub = cv2.createBackgroundSubtractorKNN()

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# declare for live fps count
fps_start_time = 0
fps = 0

# func to insert data
def insertData(buzzer):
    if buzzer == "On":
    	post = {"buzz_status": "On"}
    if buzzer == "Off":
        post = {"buzz_status": "Off"}
        
    res = collection.insert_one(post)
    print("data {buzzer} berhasil dimasukkan")
    
def publishTopic(buzzer):
    if buzzer == "On":
        publish.single("CoreElectronics/test", buzzer, hostname="test.mosquitto.org")

 
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Live FPS
	fps_end_time = time.time()
	time_diff = fps_end_time - fps_start_time
	fps = 1/(time_diff)
	fps_start_time = fps_end_time

	fps_text = "FPS: {:.2f}".format(fps)

	cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Signal to Buzzer
		# global buzzer
		buzzer = "On" if label == "No Mask" else "Off"
		buzzer_text = "Buzz: {:s}".format(buzzer)
		colorBuzz = (0, 255, 0) if buzzer == "On" else (0, 0, 255)
		publishTopic(buzzer)
		# if buzzer == "On":
		# 	publish.single("CoreElectronics/test", buzzer, hostname="test.mosquitto.org")
   
		# if buzzer == "Off":
		# 	publish.single("CoreElectronics/topic", buzzer, hostname="test.mosquitto.org")
		# cv2.putText(frame, buzzer_text, (5, 60), cv2.FONT_HERSHEY_COMPLEX, 1, colorBuzz, 1)

		# call function to insert
		# insertData(buzzer)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# data input to db
	

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
