import cv2 as cv
import numpy as np
import os, sys
#Variables to run the program
#
#Which video? 1 2 or 3
x = 1

#Selecting the video
if(x == 1):
  cap = cv.VideoCapture("D:/Videos/Video1 for Vision CW.avi")
elif(x == 2):
  cap = cv.VideoCapture("D:/Videos/Video2 for Vision CW.avi")
elif(x == 3):
    cap = cv.VideoCapture("D:/Videos/Video3 for Vision CW.avi")
else:
  print("Select Value 1, 2 or 3 for a video")

cv.startWindowThread()

#Params for inner blob
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

params2.filterByArea =  True
params2.minArea = 800

params2.filterByCircularity = True
params2.minCircularity = 0.000000001

params2.filterByConvexity = True
params2.minConvexity = 0.2

params2.filterByInertia = True
params2.minInertiaRatio = 0.7
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #Blob Detection
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
    font = cv.FONT_HERSHEY_SIMPLEX

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (13,13))

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
    thresholdIMG = cv.GaussianBlur(thresholdIMG, (11,11), 1)
    #thresholdIMG = cv.dilate(thresholdIMG, kernel2, iterations = 1)
    #thresholdIMG = cv.erode(thresholdIMG, kernel2, iterations = 1)
    thresholdIMG = cv.Canny(thresholdIMG, 100, 600,3)
    thresholdIMG = cv.dilate(thresholdIMG, (5,5), iterations=1)

    detector = cv.SimpleBlobDetector_create(params)
    detector2 = cv.SimpleBlobDetector_create(params2)
    blobs = detector.detect(gray)
    blobs2 = detector2.detect(thresholdIMG)

    inputImgBlobs = cv.drawKeypoints(rgb, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    inputImgBlobs = cv.drawKeypoints(inputImgBlobs, blobs2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    nblobs = len(blobs2)
    text = 'Number of blobs: '
    text2 = text + str(nblobs)
    cv.putText(inputImgBlobs, text2, (50,150), font, 1, (255,255,255), 2, cv.LINE_4)
    
    # Display the resulting frame
    cv.imshow('RGB', rgb)
    cv.imshow('Gray', gray)
    cv.imshow('HSV', hsv)
    cv.imshow('ThresholdIMG', thresholdIMG)
    cv.imshow('Inner & Outer Blobs', inputImgBlobs)
    
    #print(len(blobs2))
    # Press Q on keyboard to  exit
    if cv.waitKey(50) & 0xFF == ord('q'):
      break
  # Break the loop
  else: 
    break 
# When everything done, release the video capture object
cap.release()
 