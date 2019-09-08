# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
from scipy.spatial import distance as dist

 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] LOADING YOUR BEAUTY...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] LOADING AHHHHHH ...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

counter = 0
isOpen = False

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		##grab mouth index
		
		
		upperMouth = shape[49:54]	
		upperMouth2 = shape[61:65]

		lowerMouth = shape[55:60]
		lowerMouth2 = shape[65:68]
		
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in upperMouth:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		for (x, y) in upperMouth2:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		for (x, y) in lowerMouth:
			cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
		for (x, y) in lowerMouth2:
			cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
		
		distA = dist.euclidean(upperMouth[3], lowerMouth[4]) #52 and 58
		distB = dist.euclidean(upperMouth[2], lowerMouth[4]) #51 and 57
		distC = dist.euclidean(upperMouth[4], lowerMouth[2]) #53 and 56
		avgD = (distA+distB+distC)/3.
		
		#detecting mouth open and closed
		#counter to track time
		if(avgD>15):
			counter += 1

		
		if (counter >=1) and (counter <= 25) and (avgD<15):
			cv2.putText(frame, "Pause", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			counter = 0
	
		if (counter >= 26) and (counter <= 49) and (avgD<15):
			cv2.putText(frame, "Next song", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			counter = 0
		if (counter >= 50) and (avgD<15):
			cv2.putText(frame, "Previous song", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
			counter = 0
		cv2.putText(frame, str(counter), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()