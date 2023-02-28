import cv2 as cv
import numpy as np
import os, sys

cv.startWindowThread()
cap = cv.VideoCapture("D:/Videos/Video1 for Vision CW.avi")

#Params for inner blob
params = cv.SimpleBlobDetector_Params()
#Threshold
params.minThreshold = 10;
params.maxThreshold = 200;
#Filter by area
params.filterByArea = True
params.minArea = 500
#Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.01
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.3

#Params for outer blob
params2 = cv.SimpleBlobDetector_Params()

params2.minThreshold = 10
params2.maxThreshold = 200

params2.filterByArea =  True
params2.minArea = 550

params2.filterByCircularity = True
params2.minCircularity = 0.000000001

params2.filterByConvexity = True
params2.minConvexity = 0.2

params2.filterByInertia = True
params2.minInertiaRatio = 0.1

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #Blob Detection
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))

    gray = cv.dilate(gray, kernel, iterations = 1)
    gray = cv.erode(gray, kernel, iterations = 1)

    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    #Hue
    lowH = 0
    highH = 180

    #Saturation
    lowS = 0
    highS = 255

    #Value
    lowV = 0
    highV = 70

    thresholdIMG = cv.inRange(hsv, (lowH, lowS, lowV), (highH, highS, highV))

    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    cannyThreshold = 55
    thresholdIMG = cv.dilate(thresholdIMG, kernel2, iterations = 2)
    thresholdIMG = cv.erode(thresholdIMG, kernel2, iterations = 1)
    thresholdIMG = cv.Canny(thresholdIMG, cannyThreshold, cannyThreshold*2)

    detector = cv.SimpleBlobDetector_create(params)
    detector2 = cv.SimpleBlobDetector_create(params2)
    blobs = detector.detect(gray)
    blobs2 = detector2.detect(thresholdIMG)

    inputImgBlobs = cv.drawKeypoints(rgb, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    inputImgBlobs = cv.drawKeypoints(inputImgBlobs, blobs2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the resulting frame
    cv.imshow('RGB', rgb)
    cv.imshow('Gray', gray)
    cv.imshow('HSV', hsv)
    cv.imshow('ThresholdIMG', thresholdIMG)
    cv.imshow('Inner & Outer Blobs', inputImgBlobs)
 
    # Press Q on keyboard to  exit
    if cv.waitKey(50) & 0xFF == ord('q'):
      break
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()