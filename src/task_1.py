import cv2 as cv
import numpy as np
import os, sys

cv.startWindowThread()
cap = cv.VideoCapture("C:/Users/tjago/Documents/AINT515/src/Videos/Video1 for Vision CW.avi")

params = cv.SimpleBlobDetector_Params()

#Threshold
params.minThreshold = 10;
params.maxThreshold = 200;

#Filter by area
params.filterByArea = True
params.minArea = 300

#Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.01

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.2


if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #Blob Detection
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))

    gray = cv.dilate(gray, kernel, iterations = 1)
    gray = cv.erode(gray, kernel, iterations=1)

    detector = cv.SimpleBlobDetector_create(params)
    blobs = detector.detect(gray)

    inputImgBlobs = cv.drawKeypoints(rgb, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #Contours
    ret, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, method = cv.CHAIN_APPROX_NONE)

    frameCopy = rgb.copy()
    cv.drawContours(frameCopy, contours, contourIdx= -1, color = (0,255,0), thickness = 2, lineType = cv.LINE_AA)
    
    grayContours = cv.cvtColor(frameCopy, cv.COLOR_BGR2GRAY)

    grayContours = cv.dilate(grayContours, kernel, iterations = 1)
    grayContours = cv.dilate(grayContours, kernel, iterations = 1)
    contourBlobs = detector.detect(grayContours)

    inputImgContours = cv.drawKeypoints(frameCopy, contourBlobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the resulting frame
    cv.imshow('Frame',frame)
    cv.imshow('RGB', rgb)
    cv.imshow('Gray', gray)
    cv.imshow('Blobs', inputImgBlobs)
    cv.imshow("Thresh", thresh)
    cv.imshow("Contours", frameCopy)
    #cv.imshow("Blob detect with Contours", inputImgContours)
 
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