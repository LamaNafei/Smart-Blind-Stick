import os
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
#import RPi.GPIO as GPIO
import sys, webbrowser
import speech_recognition as sr



GPIO.setwarnings(False) # Ignore warning for now
#GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
#GPIO.setup(3, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 12 to be an input pin and set initial value to be pulled low (off)


def save_database(known_face_names) :
    try:
       testCon = False
       urllib.request.urlopen('http://google.com')
       testCon = True
       if testCon :
           db = firestore.client()
           doc_ref = db.collection(u'users').document(u'00001').collection(u'Images')
           doc = doc_ref.stream()
           for docc in doc :
               known_face_names.append(docc.to_dict()["name"])
               urllib.request.urlretrieve(docc.to_dict()["image"], ("/home/smart-blinds-stick/python-object-detection-opencv-main/images/"+docc.to_dict()["name"]+".jpg"))
    except :
        for filename in os.listdir("/home/smart-blinds-stick/python-object-detection-opencv-main/images/") :
            known_face_names.append(filename.split('.')[0])


def GPScode():
    print("hi")
    r = sr.Recognizer()
    talk = pyttsx3.init()
    os.system("arecord -d 5 -f S16_LE -r 48000 -c 2 test.wav")
    try :
        with sr.AudioFile("test.wav") as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            if len(text) > 1 :
                webbrowser.open("https://www.google.com/maps/place/" + text  )
    
    except sr.RequestError as e:
      talk.say("I can't get what you said, please try again")
    
    except sr.UnknownValueError:
      talk.say("I can't hear what you said, please try again")
      print("lAMA")

    talk.runAndWait()


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

cred = credentials.Certificate("/home/smart-blinds-stick/python-object-detection-opencv-main/smart blinds stick.json")
firebase_admin.initialize_app(cred)

classNames = []
classFile = '/home/smart-blinds-stick/python-object-detection-opencv-main/coco.names'
talk = pyttsx3.init()

known_face_names = []
known_face_encodings = []

save_database(known_face_names)

for name in known_face_names:
    img = face_recognition.load_image_file("/home/smart-blinds-stick/python-object-detection-opencv-main/images/" + name + ".jpg") #Edit path
    encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(encoding)


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
     
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(18, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 12 to be an input pin and set initial value to be pulled low (off)
    if GPIO.input(18) == GPIO.HIGH:
       GPScode()
    GPIO.cleanup()
    # Grab the raw NumPy array representing the image
    image = frame.array
    classIds, confs, bbox = net.detect(image, confThreshold=0.5)
    
     
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
    
    if len(classIds) !=0:
            
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
           cv2.rectangle(image, box, color=(0, 255, 0), thickness=2)
           playback = ""
           name = "Unknown person"
           if(classNames[classId-1] == "person") :
               face_locations = face_recognition.face_locations(image)
               face_encodings = face_recognition.face_encodings(image, face_locations)
               pil_image = Image.fromarray(image)
               draw = ImageDraw.Draw(pil_image)

               for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                  matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                  face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                  best_match_index = np.argmin(face_distances)
                  
                  if matches[best_match_index] :
                     name = known_face_names[best_match_index]
                  draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                  text_width, text_height = draw.textsize(name)
               
               cv2.putText(image, name, (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
               if not (name == playback) :
                  talk.say(name)
                  playback = name
           else :
               cv2.putText(image, classNames[classId-1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
               if not (name == playback) :
                  talk.say(classNames[classId-1])
                  playback = classNames[classId-1]
    talk.runAndWait()

    # Display the frame using OpenCV
    cv2.imshow("Frame", image)
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF  
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
