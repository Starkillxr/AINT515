import cv2 as cv
import numpy as np
import os, sys

cv.startWindowThread()
cap = cv.VideoCapture("E:/Videos/Video1 for Vision CW.avi")

params = cv.SimpleBlobDetector_Params()
#Threshold
params.minThreshold = 0
params.maxThreshold = 2000
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
params.minInertiaRatio = 0.1

#Params for outer blob
params2 = cv.SimpleBlobDetector_Params()

params2.minThreshold = 0
params2.maxThreshold = 2000

params2.filterByArea = True
params2.minArea = 1000

params2.filterByCircularity = False
params.minCircularity = 0.000000001

params2.filterByConvexity = True
params2.minConvexity = 0.2

params2.filterByInertia = True
params2.minInertiaRatio = 0.01

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
    #Blob Detection
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
  
    cannyThreshold = 55
  
    gray = cv.dilate(gray, kernel, iterations = 1)
    gray = cv.erode(gray, kernel, iterations = 1)
    #gray = cv.Canny(gray, cannyThreshold, cannyThreshold*2)


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #Hue
    lowH = 0
    highH = 180

    #Saturation
    lowS = 63.3
    highS = 255

    #Value
    lowV = 11.8
    highV = 70

    thresholdIMG = cv.inRange(hsv, (lowH, lowS, lowV), (highH, highS, highV))

    
    
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    thresholdIMG = cv.GaussianBlur(thresholdIMG, (3,3),1)
    thresholdIMG = cv.Canny(thresholdIMG, 700, 800)
    cv.imshow("Canny Before Dilate", thresholdIMG)
    thresholdIMG = cv.dilate(thresholdIMG, kernel2, iterations = 1)
    #cv.imshow("Canny After Dilate", thresholdIMG)
    #thresholdIMG = cv.erode(thresholdIMG, kernel2, iterations = 1)
    #contours, hierarchy = cv.findContours(thresholdIMG, cv.RETR_EXTERNAL,
    #                                      cv.CHAIN_APPROX_NONE)
    #cv.drawContours(thresholdIMG, contours, -1, (0,255,0),4)

    detector = cv.SimpleBlobDetector_create(params)
    detector2 = cv.SimpleBlobDetector_create(params2)
    blobs = detector.detect(gray)
    blobs2 = detector2.detect(thresholdIMG)

    inputImgBlobs = cv.drawKeypoints(rgb, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    inputImgBlobs2 = cv.drawKeypoints(inputImgBlobs, blobs2, np.array([]), (0,255,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the resulting frame
    #cv.imshow('Gray', gray)
    #cv.imshow('HSV', hsv)
    cv.imshow('ThresholdIMG', thresholdIMG)
    cv.imshow('Blobs', inputImgBlobs)
    text = 'Number of Blobs:'
    org = (50,150)
    cv.putText(inputImgBlobs2, text, org, fontFace= cv.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0,255,255))
    cv.imshow('Blobs 2', inputImgBlobs2)
 
    # Press Q on keyboard to  exit
    if cv.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()