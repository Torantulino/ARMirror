from guizero import App, Text
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import sys

#Initialise camera and get ref to raw feed
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

#Background Subtractor
#fgbg = cv2.createBackgroundSubtractorMOG2()

# Create face cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Wait for camera to initialise
time.sleep(1)

app = App(title="Hello world")
welcomeMessage = Text(app, text="Goodbye!", size=50, color="blue")

#Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    #fgmask = fgbg.apply(frame.array) #interesting background selector

    #Greyscale Image
    grey = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY) 

    # Detect Face
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor = 2,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame.array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    
    image = frame.array

    
    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("q"):
        cv2.destroyAllWindows()
        break



app.display()

