# import the necessary packages
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2# OpenCV library
import pyttsx3
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from firebase_admin import firestore
import urllib.request
# Initialize the camera
camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (640, 480)
 
# Set the number of frames per second
camera.framerate = 32
 
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(640, 480))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)


classNames = []
classFile = 'coco.names'
talk = pyttsx3.init()
with open(classFile, 'rt') as f:
     classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

weightPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320 , 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
     
    
    # Grab the raw NumPy array representing the image
    image = frame.array
    classIds, confs, bbox = net.detect(image, confThreshold=0.5)
    
     
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
    
    if len(classIds) !=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
           cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
           if(classNames[classId-1] == "person") :
               name = "Unknown"
               face_locations = face_recognition.face_locations(image)
               face_encodings = face_recognition.face_encodings(image, face_locations)
               pil_image = Image.fromarray(image)
               draw = ImageDraw.Draw(pil_image)

               for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                  matches = face_recognition.compare_faces(knownPeople.values(), face_encoding)
                  face_distances = face_recognition.face_distance(knownPeople.values(), face_encoding)
                  best_match_index = np.min(face_distances)
                  
                  if matches[best_match_index] :
                     name = knownPeople[best_match_index]
                  draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                  text_width, text_height = draw.textsize(name)
               
               cv2.putText(image, name, (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
           else :
               cv2.putText(image, classNames[classId-1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
         
    # Display the frame using OpenCV
    cv2.imshow("Frame", image)
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF  
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break