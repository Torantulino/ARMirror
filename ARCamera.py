from guizero import App, Text
from picamera.array import PiRGBArray
from picamera import PiCamera
import numpy as np
import time
import cv2
import sys
import os
import subprocess

#Initialise camera and get ref to raw feed
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

#Initialise Bluetooth Script
blueARM = subprocess.Popen(['python', 'ARMBluetooth.py'])

# Initialise variables
selectedItem = 0

# Create face cascade
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Wait for camera to initialise
time.sleep(1)

# Create Clothing Array
clothing = []

#Load Test Images - all channels
for file in os.listdir('pic'):
    clothing.append(cv2.imread('pic/' + file, -1))
    
imgGlasses = cv2.imread('pic/glasses.png', -1)


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

    # Enable writing on base image
    base.setflags(write=1)
    
    for c in range(0, 3):
        base[y1B:y2B, x1B:x2B, c] = (alpha * overlay[y1O:y2O, x1O:x2O, c] +
                                 alphaInv * base[y1B:y2B, x1B:x2B, c])

"""
    cv2.imshow("Last Augmented", base)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.destroyAllWindows()
"""

#Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    #Greyscale Image
    grey = cv2.cvtColor(frame.array, cv2.COLOR_BGR2GRAY)


    # Detect Face
    faces = faceCascade.detectMultiScale(
        grey,
        scaleFactor = 1.5,
        minNeighbors = 5,
        minSize = (30,30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    baseImg = frame.array
        
    # Overlay chosen accessory
    for (x, y, w, h) in faces:
        '''
        # Debug Rectangle
        v2.rectangle(frame.array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        '''
        # Resize selected overlay
        overlayR = cv2.resize(clothing[selectedItem], (w, int(h/2)))

        # Render overlay onto camera frame
        RenderOnto(baseImg, overlayR[:, :, 0:3], (x, int(y+h/5)), overlayR[:,:,3] / 255.0)
        
        break
    
    # Render final frame
    cv2.imshow("Current Frame", baseImg)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    # Switch item
    print ("Bluetooth Poll:", blueARM.poll())
    # If Subprocess has finished running, run again
    if blueARM.poll() != None:
        print("BlueARM Launched")
        # Execute Subprocess to wait for incomming data from phone
        blueARM = subprocess.Popen(['python', 'ARMBluetooth.py'])
    
    
    """
    if mobileData == "next":
        if len(clothing) > selectedItem + 1:
            selectedItem += 1
        else:
            # Loop back to first Item
            selectedItem = 0

    if mobileData == "s":
        if selectedItem - 1 >= 0:
            selectedItem += -1
        else:
            # Loop to last item
            selectedItem = len(clothing) - 1
    """
    #------------    

    # Exit Program
    if key == ord("q"):
        cv2.destroyAllWindows()
        break



    
