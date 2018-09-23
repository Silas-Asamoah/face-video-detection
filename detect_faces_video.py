
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading model...")
#load serialized model from local disk
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
	# grab the frame from the threaded video stream and resize to maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob (Binary Large Object)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# obtain predictions
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		# extract the confidence
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence < args["confidence"]:
			continue

		# computing the (x, y)-coordinates of the bounding box for the object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# draw the bounding box of the face along with the associated confidence
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# output frame
	cv2.imshow("Video Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `m` key is pressed, break loop
	if key == ord("m"):
		break

cv2.destroyAllWindows()
vs.stop()

#Run in terminal 
#python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel


