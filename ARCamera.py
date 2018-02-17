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

# Create face cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Wait for camera to initialise
time.sleep(1)

#------------------GLASSES---------------------------

#Load Glasses TestImage - all channels
imgGlasses = cv2.imread('pic/glasses.png', -1)

#Create glasses mask using only alpha
#glassMask = imgGlasses[:,:,3]

#invert mask
#invGlassMask = cv2.bitwise_not(glassMask)

#Convert glasses to BGR
#imgGlasses = imgGlasses[:,:,0:3]
#origGlassesHeight, origMaskWidth = glassMask.shape[:2]

#-----------------------------------------------

# Image overlay Method
def RenderOnto(base, overlay, position, alphaMask):
    #Alpha mask must be the same size as overlay

    x, y = position

    # Base Ranges
    y1B, y2B = max(0, y), min(base.shape[0], y + overlay.shape[0])
    x1B, x2B = max(0, x), min (base.shape[1], x + overlay.shape[1])
    
    # Overlay Ranges
    y1O, y2O = max(0, -y), min(overlay.shape[0], base.shape[0] - y)
    x1O, x2O = max(0, -x), min(overlay.shape[1], base.shape[1] - x)

    alpha = alphaMask[y1O:y2O, x1O:x2O]
    # Invert Alpha
    alphaInv = 1.0 - alpha

    base.setflags(write=1)
    
    for c in range(0, 3):
        base[y1B:y2B, x1B:x2B, c] = (alpha * overlay[y1O:y2O, x1O:x2O, c] +
                                 alphaInv * base[y1B:y2B, x1B:x2B, c])

    cv2.imshow("Last Augmented", base)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.destroyAllWindows()


#Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    #Greyscale Image
    grey = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)


    # Detect Face
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor = 2.2,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    baseImg = frame.array
        
    # Draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame.array, (x, y), (x+w, y+h), (0, 255, 0), 2)

        RenderOnto(baseImg, imgGlasses[:, :, 0:3], (x, y), imgGlasses[:,:,3] / 255.0)
        
        break
    

    
    cv2.imshow("Current Frame", baseImg)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord("q"):
        cv2.destroyAllWindows()
        break



    
