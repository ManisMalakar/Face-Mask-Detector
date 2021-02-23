# Importing the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


def detect_and_predict_mask(frame, faceNet, maskNet):
	# Grabbing the dimensions of the frame and then construct a square
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# Passing the blob through the network and obtaining the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# Initializing the list of faces, their corresponding locations,
	# and the list of predictions from the face mask network
	faces = []
	locs = []
	preds = []

	# Looping over the detections
	for i in range(0, detections.shape[2]):
		# Extracting the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# Filtering out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# Computing the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Ensuring the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extracting the face ROI, convert it from BGR to RGB channel
			# Ordering, and Resizing it to 224x224, and PreProcessing it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Adding the face and bounding boxes to their respective
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Making a predictions if at least one face was detected
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	# Returning a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# Loading the serialized face detector model from the premade model
prototxtPath = "face_detector/deploy.prototxt"
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Loading the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# Initializing the video stream
print("Starting video stream...")
vs = VideoStream(src=0).start()

# Looping over the frames from the video stream
while True:
	# Grabbing the frame from the threaded video stream and resize it
	# to have a maximum width of 500 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=1024)

	# Detecting faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Looping over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# Unpacking the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Determining the class label and color drawing
		# the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Including the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Displaying the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Showing the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# Cleaning up
cv2.destroyAllWindows()
vs.stop()