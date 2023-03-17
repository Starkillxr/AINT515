import cv2 as cv
import numpy as np
import os, sys

<<<<<<< HEAD
cv.startWindowThread()
<<<<<<< HEAD
cap = cv.VideoCapture("E:/Videos/Video1 for Vision CW.avi")
=======
>>>>>>> f610af92287b1de216fdd086d5006d18129973ab

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
=======
cap = cv.VideoCapture("D:/Videos/Video1 for Vision CW.avi")
>>>>>>> parent of b92bcd4... Update task_1.py

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

<<<<<<< HEAD
params2.filterByArea = True
params2.minArea = 1000

params2.filterByCircularity = False
params.minCircularity = 0.000000001
=======
params2.filterByArea =  True
params2.minArea = 550

params2.filterByCircularity = True
params2.minCircularity = 0.000000001
>>>>>>> f610af92287b1de216fdd086d5006d18129973ab

params2.filterByConvexity = True
params2.minConvexity = 0.2

params2.filterByInertia = True
params2.minInertiaRatio = 0.1

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
<<<<<<< HEAD
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    #Blob Detection
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
    cannyThreshold = 55
  
    gray = cv.dilate(gray, kernel, iterations = 1)
    gray = cv.erode(gray, kernel, iterations = 1)
    #gray = cv.Canny(gray, cannyThreshold, cannyThreshold*2)


    hsv = cv.cvtColor(rgb, cv.COLOR_RGB2HSV)
    #Hue
    lowH = 0
    highH = 180

    #Saturation
    lowS = 63.3
    highS = 255

    #Value
    lowV = 11.8
    highV = 70
=======
for i in range(1,3):
  # Read until video is completed
  x=i
  while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      #Blob Detection
      rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
      gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
      gray = cv.GaussianBlur(gray, (11,11), 0)
      gray = cv.Canny(gray, 30, 150, 3)
      gray = cv.dilate(gray, (1,1), iterations = 1)

      kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))

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
>>>>>>> f610af92287b1de216fdd086d5006d18129973ab

      kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
      cannyThreshold = 55
      thresholdIMG = cv.GaussianBlur(thresholdIMG, (11,11), 0)
      thresholdIMG = cv.Canny(thresholdIMG, cannyThreshold, cannyThreshold*2)
      thresholdIMG = cv.dilate(thresholdIMG, kernel2, iterations = 1)

<<<<<<< HEAD
    
    
    kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
    cannyThreshold = 60
    thresholdIMG = cv.GaussianBlur(thresholdIMG, (3,3),1)
    thresholdIMG = cv.Canny(thresholdIMG, 60, 300)
    thresholdIMG = cv.dilate(thresholdIMG, kernel2, iterations = 1)
    #thresholdIMG = cv.erode(thresholdIMG, kernel2, iterations = 1)
<<<<<<< HEAD
    #contours, hierarchy = cv.findContours(thresholdIMG, cv.RETR_EXTERNAL,
    #                                      cv.CHAIN_APPROX_NONE)
    #cv.drawContours(thresholdIMG, contours, -1, (0,255,0),4)
=======
      detector = cv.SimpleBlobDetector_create(params)
      detector2 = cv.SimpleBlobDetector_create(params2)
      blobs = detector.detect(gray)
      blobs2 = detector2.detect(thresholdIMG)
>>>>>>> f610af92287b1de216fdd086d5006d18129973ab

      inputImgBlobs = cv.drawKeypoints(rgb, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
      inputImgBlobs = cv.drawKeypoints(inputImgBlobs, blobs2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
<<<<<<< HEAD
      inputImgBlobs2 = cv.drawKeypoints(thresholdIMG, blobs2, np.array([]), (0,255,0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
=======
>>>>>>> parent of 85419e2... Update task_1.py
      # Display the resulting frame
      cv.imshow('RGB', rgb)
      cv.imshow('Gray', gray)
      cv.imshow('HSV', hsv)
      cv.imshow('ThresholdIMG', thresholdIMG)
      cv.imshow('Inner & Outer Blobs', inputImgBlobs)

<<<<<<< HEAD
=======
    contours, hierarchy = cv.findContours(thresholdIMG, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_NONE)
    cv.drawContours(thresholdIMG, contours, -1, (0,255,0),4)

    detector = cv.SimpleBlobDetector_create(params)
    detector2 = cv.SimpleBlobDetector_create(params2)
    blobs = detector.detect(gray)
    blobs2 = detector2.detect(thresholdIMG)

>>>>>>> parent of b92bcd4... Update task_1.py
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
=======
      # Press Q on keyboard to  exit
      if cv.waitKey(50) & 0xFF == ord('q'):
        break
    # Break the loop
    else: 
>>>>>>> f610af92287b1de216fdd086d5006d18129973ab
      break
  if cv.waitKey(5) & 0xFF == ord('q'):
    break
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv.destroyAllWindows()