# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:57:53 2019

@author: AVyas
"""

#importingh Copy library to copy data
import copy
#Importing Some Global Functions
from global_functions import findCheckerboardCoordinates,rescale,thresholding,find_red,remove_value,iterate,find_concurrent,apply_cascade
#Importing RealSense SDK
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Importing pandas for data storage
import pandas as pd

import pickle

"""
mouse_callback can be used on opencv window to print the cordinate or to start some action.
For e.g. : 
    When event==1 -> left click is made and this will turn on the ix Flag. and opposite for event==2.
    if ix is set True the event will happen and vice versa.
    This can be used as a button to perform some function when working with videos in real time
It was really helpful for me when I turned on the camera carrying the mouse and then I went in front of it and 
I clicked on the mouse to capture images
"""
def mouse_callback(event, x, y, flags, params):
    global ix
    if event ==1:
        print("LEFT_CLICK_MADE") 
        ix=True
    if event == 2:
        ix=False
        print([x,y])
        
#IF YOU WANT TO SAVE IT TO VIDEO CREATE A VIDEO WRITER        
out = cv2.VideoWriter('data/read.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,780))

#IMPORT THE RECORDED FILES
pkl_file = open('data/record_15.pkl', 'rb')
record_15 = pickle.load(pkl_file)
pkl_file.close()


#Some Variables
ix=False
depth_scale=0.0010000000474974513 #YOU CAN GET IT FROM CAMERA

first_depth,first_color = record_15[0] #THE FIRST FRAME IS THE BLANK FRAME



# Streaming loop
#cv2.namedWindow("W")
#cv2.setMouseCallback("W",mouse_callback)
#FACE ITERATION

answer=[]
try:

    for i in range(1,len(record_15)):
        key = cv2.waitKey(1000)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        depth,color = record_15[i]
        
        #APPLYING DIFFERENT FACE RECOGNTION CASCADE. CHECK FOR THE OUTPUT OF APPLY CASCADE.ACTIVATE ONE AT A TIME.
#        color_image1,check,values=apply_cascade(color.copy(),'models/cascades/haarcascade_frontalface_default.xml')
        color_image1,check,values=apply_cascade(color.copy(),'models/cascades/haarcascade_lefteye_2splits.xml')
#        color_image1,check,values=apply_cascade(color.copy(),'models/cascades/haarcascade_righteye_2splits.xml')
#        color_image1,check,values=apply_cascade(color.copy(),'models/cascades/haarcascade_eye.xml')

        #USE THE FIRST DETECTED AREA
        if check and len(values)>0:
            (x,y,w,h)=values[0]
#            x*=1.02
#            w*=0.5
#            y*=1.02
#            h*=0.5
            x,y,w,h = int(x),int(y),int(w),int(h)
            cv2.rectangle(color_image1,(x,y),(x+w,y+h),(255,0,0),2)
            Area=depth[y:y+h,x:x+w]
            d=iterate(Area)
            d=d*depth_scale*100 #ACTUAL DISTANCE
            cv2.putText(color_image1,'distance {} cm'.format(d), (100,250),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            cv2.imshow('W', rescale(color_image1,90))
            out.write(color_image1)
            print(d)
            answer.append(d)

        else:
            cv2.putText(color,"NOT FOUND", (100,250),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            cv2.imshow('W', rescale(color,90))

finally:
    cv2.destroyAllWindows() #CLOSE EVERY WINDOW

#%%
    
#FULL BODY ITERATION    
#LOCAL VARIABLE
answer=[]

try:

    for i in range(1,len(record_15)):
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        depth,color = record_15[i]
        

        image3 = cv2.subtract(first_color,color)
        ret,a= cv2.threshold(image3,100,255,cv2.THRESH_BINARY)
        grayImage = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        
        grayImage[grayImage>100]=255
        grayImage[grayImage<=100]=0


        depth_copy = depth.copy()
        for i in range(720):
            for j in range(1280):
                if int(grayImage[i][j])==255:
                    depth_copy[i][j] = depth[i][j]
                else :
                    depth_copy[i][j]= 0

        print(iterate(depth_copy)*depth_scale*100)
        
        cvuint8 = cv2.convertScaleAbs(depth_copy)
        cv2.imshow("image",rescale(cvuint8,75))


finally:
    cv2.destroyAllWindows()
    
    
    
#%%    
#FACE DLIB ITERATION
from imutils import face_utils
import dlib
import imutils
from collections import OrderedDict

pkl_file = open('data/record_15.pkl', 'rb')
record_15 = pickle.load(pkl_file)
pkl_file.close()

#Some Variables
ix=False
depth_scale=0.0010000000474974513

first_depth,first_color = record_15[0]

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

"""
FIND_HIGHEST AND FIND_LOWEST WILL TAKE A THRESHOLDED IMAGE AND FIND THE HIGHEST/LOWEST WHITE POINT IN THE IMAGE
I USED THIS TO CREATE A YELLOW LINE THAT YOU SEE IN THE FINAL IMAGE i.e. TO MEASURE PIXEL HEIGHT
"""
def find_highest(image): #IMAGE SHOULD BE unit8 single layer
    lowest=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j]>lowest:
                return [j,i]
    return [0,0]            

def find_lowest(image): #IMAGE SHOULD BE  unit8 single layer
    lowest=0
    for i in range(image.shape[0]-1,-1,-1):
        for j in range(image.shape[1]-1,-1,-1):
            if image[i][j]>lowest:
                return [j,i]
    return [0,0]        

"""
FIND FACE POINTS TAKES AN IMAGE APPLY FACE DETECTION OF DLIB LIBRARY.
IT GIVES 68 FACIAL LANDMARK POINTS INTO A DICTIONARY FORMAT.
I IMPLEMENTED THIS FUNCTION WHILE WORKING WITH LOGITECH CAMERAS
THEREFORE, CHECK GLOBAL_FUNCTIONS OF LOGITECH C290 PROJECT. IT WILL HAVE A CLASS CALLED FACE. 
THIS FUNCTION IS TAKEN FROM THAT CLASS.
THE FACE CLASS ALSO HAS A FUNCTION TO COLORIZE THE REGION DETECTED.
"""    
def find_face_points(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/cascades/shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    dic={}
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    	# show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1)
        dic[i]=shape
    return dic,image
    
"""
IT TAKES THE IMAGE,LIST OF CORDINATES, THE TUPLE LIST i.e. THE REGION WE WANT TO CONSIDER, AND DEPTH IMAGE.
BASED ON TUPLE LIST AND LIST1, THE AREA IS CREATED AND ITS DEPTH INFORMATION IS GIVEN.
THE TUPLE LIST IS GIVEN FROM FACIAL_LANDMARKS_IDXS VARIABLE. USE IT TO DETECT NOSE OR ANY OTHER REGION YOU WANT
"""    
def get_area(image,list1,tuple_list,depthimage):
    (j,k) = tuple_list
    pts = list1[j:k]
    hull = cv2.convexHull(pts)
#    cv2.drawContours(image, [hull], -1, (19,199,109), -1)
    rect = cv2.boundingRect(hull)
    cv2.rectangle(image,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),3)
    area = depthimage[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return image,area,rect[0]

    
    

answer=[]

df = pd.DataFrame()
temp = {}
try:

    for i in range(1,len(record_15)):
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        depth,color = record_15[i]
        dic1,image1 = find_face_points(color.copy())
        if len(dic1)<1:
            continue

        #DETECT THE RIGHT EYE AND ITS DISTANCE        
        image,area,loc1 = get_area(color.copy(),dic1[0],FACIAL_LANDMARKS_IDXS["right_eye"],depth)        
        size = area.shape[0]*area.shape[1]
        re = iterate(area)*depth_scale*100

        #DETECT THE LEFT EYE AND ITS DISTANCE        
        image,area,loc2 = get_area(color.copy(),dic1[0],FACIAL_LANDMARKS_IDXS["left_eye"],depth)        
        size2 = area.shape[0]*area.shape[1]
        le = iterate(area)*depth_scale*100

        #THE AREA CONSTRAINT        
        if min([size,size2])/max([size,size2])<0.8:
            continue
        print([size,size2])

        #NULL CONSTRAINT
        if np.isnan(le) or np.isnan(re):
            continue

        loc = min(loc1,loc2)
        
        #ADD SOME VALUE TO DISTANCE MEASUREMNT i.e. REDUCE THE ERROR
        error = -1  #FOUND BY EXPERIMENTAION. IT CAN VARY SO REMOVE THIS THREE LINES OF CODE FOR FRESH ENVIRONMENT
        le+=error
        re+=error

        #FINDING THE SUBJECT HEIGHT IN PIXELS
        #IF YOU SEE YELLOW LINES WAY TOO FAR FROM SUBJECT INCREASE THRESHOLD AND VICE VERSA
        threshold = 70
        #BACKGROUND SUBTRACTION
        image3 = cv2.subtract(first_color,color)
        #THRESHOLDING
        ret,a= cv2.threshold(image3,threshold,255,cv2.THRESH_BINARY)
        grayImage = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        grayImage[grayImage>threshold]=255
        grayImage[grayImage<=threshold]=0
        
        #FINDING TOP AND BOTTOM POINTS OF SUBJECT
        [x1,y1]=find_highest(grayImage)
        [x2,y2]=find_lowest(grayImage)
        height = abs(y2-y1) #PIXEL HEIGHT

        #DRAW LINES TO SEE WHERE IS THE TOP POINT
        cv2.line(color,(0,y1),(1280,y1),(0,255,255),2)    
        cv2.line(color,(0,y2),(1280,y2),(0,255,255),2)  

        #CONVERSTION OF PIXEL HEIGHT INTO REAL HEIGHT
        real_height1 = le*height/916.364501953125 #916.36 IS FOCAL LENGTH OBTAINED FROM GET_CAMERA_FY
        real_height2 = re*height/916.364501953125
        expected = 177.8*916.364501953125/height

        #DISPLAY THE IMAGE
        cv2.imshow("image",rescale(color,75))
        
        #STORE THE DATA INTO DATAFRAME
        temp = {"Eye_l":le,"Eye_R":re,"pixel_height":height,'expected':expected,'location':loc,'E_R':real_height2,'E_L':real_height1}
        df=df.append(temp,ignore_index=True)
#        print(temp)

finally:
    cv2.destroyAllWindows() #CLOSE ALL WINDOWS
    
    
    

    
    
    
    
    
    
    