from picamera import PiCamera
import picamera.array
from picamera.array import PiRGBArray
import cv2
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


def Camera():

   cred = credentials.Certificate('smart blinds stick.json')
# Initialize the app with a service account, granting admin privileges
 #  database = firebase_admin.initialize_app(cred)
   
   app = firebase_admin.initialize_app(cred, {
    'storageBucket' : "gs://smart-blinds-stick-ff409.appspot.com/"
   })
   # urllib.request.urlretrieve(ref, '../images/{name}.jpg')
   
   # known_face_names = ref.document(name)
   # known_face_encoding = []
   bucket = storage.bucket("images")

   # urllib.request.urlretrieve(ref, '../images/{name}.jpg')
   
   # known_face_names = ref.document(name)
   known_face_encoding = []

   camera = PiCamera()
   camera.resolution = (640, 480)
   camera.framerate = 32
   raw_capture = picamera.array.PiRGBArray(camera, size=(640, 480))

   
   classNames = []
   classFile = 'coco.names'
  
   talk = pyttsx3.init()
   
   #knownPeople = ref.get().to_dict()
   # for name in known_face_names :
   #    image = face_recognition.load_image_file("images/" + name + ".jpg") #Edit path
   #    encoding = face_recognition.face_encodings(image)[0]
   #    known_face_encoding.append(encoding)

  # for name in knownPeople.keys():
   #        image = face_recognition.load_image_file(knownPeople[name]) #Edit path  
    #       encoding = face_recognition.face_encodings(image)[0]
     #      knownPeople[name] = encoding

   with open(classFile, 'rt') as f:
      classNames = f.read().rstrip('\n').split('\n')

   configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
   weightPath = 'frozen_inference_graph.pb'

   net = cv2.dnn_DetectionModel(weightPath, configPath)
   net.setInputSize(320 , 230)
   net.setInputScale(1.0 / 127.5)
   net.setInputMean((127.5, 127.5, 127.5))
   net.setInputSwapRB(True)

   #while True :
   for frame in camera.capture_continuous(raw_capture, format="jpeg", use_video_port=True):
      img = frame.array
      cv2.imshow("Frame", image)
      key = cv2.waitKey(1) & 0xFF
      classIds, confs, bbox = net.detect(img, confThreshold=0.5)
      raw_capture.truncate(0)
      print(classIds, bbox)

      if len(classIds) !=0:
         for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            
            if(classNames[classId-1] == "person") :
               face_locations = face_recognition.face_locations(img)
               face_encodings = face_recognition.face_encodings(img, face_locations)
               pil_image = Image.fromarray(img)
               draw = ImageDraw.Draw(pil_image)
               
               for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                  matches = face_recognition.compare_faces(knownPeople.values(), face_encoding)
                  name = "Unknown"
                  face_distances = face_recognition.face_distance(knownPeople.values(), face_encoding)
                  best_match_index = np.min(face_distances)
                  
                  if matches[best_match_index] :
                     name = knownPeople[best_match_index]
                  draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
                  text_width, text_height = draw.textsize(name)
               
               cv2.putText(img, name, (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
               talk.say(classNames[classId-1])

            else :
               cv2.putText(img, classNames[classId-1], (box[0] + 10, box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)
               talk.say(classNames[classId-1])
            
            talk.runAndWait()

      cv2.imshow('Output', img)
      cv2.waitKey(1)

Camera()

########## no needed ############

## known_face_names = ["abed","ayman","basel","basem","qusai"]
## known_face_encodings=[]

####### no needed #########3

# knownPeople = {}

#### get data from database ####

# for name in knownPeople.keys():
#         image = face_recognition.load_image_file("images/" + name + ".jpg") #Edit path
#         encoding = face_recognition.face_encodings(image)[0]
#         knownPeople[name] = encoding

#### get data from database ####

########### in if statement ############### 

# unknown_image = face_recognition.load_image_file("images/unknown.jpg") موجودة في الكود الي فوق بجيب صورة من الكاميرا
# face_locations = face_recognition.face_locations(img)
# face_encodings = face_recognition.face_encodings(img, face_locations)
# pil_image = Image.fromarray(img)
# draw = ImageDraw.Draw(pil_image)
# for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#     matches = face_recognition.compare_faces(knownPeople.values(), face_encoding)
#     name = "Unknown"
#     face_distances = face_recognition.face_distance(knownPeople.values(), face_encoding)
#     best_match_index = np.min(face_distances)
#     if matches[best_match_index] :
#         name = knownPeople[best_match_index]
#     draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
#     text_width, text_height = draw.textsize(name)

########### in if statement ############### 


########### already in code ################

##     draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
##     draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

########### already in code ################

