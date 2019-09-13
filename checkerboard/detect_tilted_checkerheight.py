# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:26:45 2019

@author: AVyas
"""


#importingh Copy library to copy data
import copy
#Importing Some Global Functions
from global_functions import findCheckerboardCoordinates,rescale,thresholding,find_red,remove_value,iterate
#Importing RealSense SDK
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Importing pandas for data storage
import pandas as pd
#import the math class for mathematical operations
import math



"""
Find_c_height will take the image and divide it into two parts.
Then it will detect 2 checkerboards i.e. upper and lower
it will detect highest and lowest point and will return 
difference of those points
"""
def find_c_height(img):
    frame1 = copy.copy(img)
    frame2=copy.copy(frame1)
    frame2[0:450,:]=(0,0,0)
    frame1[230:719,:]=(0,0,0)
    cv2.imshow("img2",rescale(frame1,50))
    cv2.imshow("img3",rescale(frame2,50))
    a1,check1= findCheckerboardCoordinates(frame1)
    a2,check2=findCheckerboardCoordinates(frame2)
#    print([check1,check2])
    if check1 and check2:
        y1 = min([a1[i][1] for i in range(len(a1)) ])
        y2 = max([a2[i][1] for i in range(len(a1)) ])
        cv2.line(img,(0,y1),(1920,y1),(0,255,255),2)    
        cv2.line(img,(0,y2),(1920,y2),(0,255,255),2)
#        cv2.imshow("img",rescale(img,20))
        height = y2-y1
#        print(height)
        return True,height
    return False,0


# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()

#Enabling High Accuracy Preset
depth_sensor.set_option(rs.option.visual_preset, 3.0)
preset_name = depth_sensor.get_option_value_description(rs.option.visual_preset, 3.0)

#Enabling Emitter To increase accuracy
enable_ir_emitter = True
depth_sensor.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)

#GET the depth scale from the SDK
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 10 #10 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

colorizer = rs.colorizer() # TO colorize the depth image

#FILTERS
dec_filter = rs.decimation_filter ()   # Decimation - reduces depth frame density
dec_filter.set_option(rs.option.filter_magnitude, 4)

spat_filter = rs.spatial_filter()          # Spatial    - edge-preserving spatial smoothing
spat_filter.set_option(rs.option.filter_magnitude, 5)
spat_filter.set_option(rs.option.filter_smooth_alpha, 1)
spat_filter.set_option(rs.option.filter_smooth_delta, 50)
spat_filter.set_option(rs.option.holes_fill, 3)

temp_filter = rs.temporal_filter()    # Temporal   - reduces temporal noise

hole_filling = rs.hole_filling_filter() #FIll ALL THE HOLES
#FILTERS END

#Some Variables
count = 1000
record = []
answer=[]
height_data=[]
global t1,t2,t3
t1,t2,t3,t4=[],[],[],[]
height1 = []

# Streaming loop
try:
    while True:
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 1280x720 depth image
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image

#        aligned_depth_frame = dec_filter.process(aligned_depth_frame)
        aligned_depth_frame = spat_filter.process(aligned_depth_frame) #Applying Filter
        aligned_depth_frame = temp_filter.process(aligned_depth_frame) #Applying Filter       
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)#Applying Filter        

        color_frame = aligned_frames.get_color_frame() #Getting RGB frame
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) #Getting final depth image

        #Colorize the depth image
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data()) 
        
#        #See the depth image
#        cv2.imshow("colorized depth",rescale(colorized_depth,50))


        color_image = np.asanyarray(color_frame.get_data()) #Getting final RGB frame

        #UPPER CHECKER BOARD
        img1=color_image.copy() #Create Copy
        img1[230:719,:]=(0,0,0) #Threshold remaining area
        a1,check1= findCheckerboardCoordinates(img1) #FInd the cordinates 
        if not check1: #if we don't find the checkerboard
            print("Check 1") 
            cv2.imshow('Found Red', rescale(color_image,50))
            continue
#        #Visualise the color image with a drawn rectangle. NOTE: Turn it off when processing the RGB Image. Only for debugging.
        cv2.rectangle(color_image,tuple(a1[53]),tuple(a1[43]),(0,255,0),5) 
        cv2.rectangle(color_image,tuple(a1[46]),tuple(a1[36]),(0,255,0),5) 

#        Repeating the procedure for other checkerboard
        
        #BOTTOM Checkerboard
        img2=color_image.copy()
        img2[0:450,:]=(0,0,0)
        a2,check2= findCheckerboardCoordinates(img2)
        if not check2:
            print("Check 2")
            cv2.imshow('Found Red', rescale(color_image,50))
            continue
        cv2.rectangle(color_image,tuple(a2[17]),tuple(a2[7]),(0,255,0),5)
        cv2.rectangle(color_image,tuple(a2[11]),tuple(a2[1]),(0,255,0),5)
        

        #Extracting the depth map for first checkerboard
        x1= a1[53][0]
        y1= a1[53][1]
        w1= abs(a1[43][0] - a1[53][0] )
        h1= abs(a1[43][1] - a1[53][1] )
        Area1 = depth_image[int(y1):int(a1[43][1]),int(x1):int(a1[43][0])]


        x1= a1[46][0]
        y1= a1[46][1]
        w1= abs(a1[36][0] - a1[46][0] )
        h1= abs(a1[36][1] - a1[46][1] )
        Area2 = depth_image[int(y1):int(a1[36][1]),int(x1):int(a1[36][0])]

        #calculating the final distance by applying the iterate method to eliminate local minima/maxima
        distance1=iterate(Area1)*depth_scale*100 #inches
        distance2=iterate(Area2)*depth_scale*100 #inches 
#        print([distance1,distance2]) 
        
        height1 = a2[8][1] - a1[53][1]
        height2 = a2[0][1] - a1[45][1]
        
        height_left= distance1*height1/916.364501953125
        height_right= distance2*height2/916.364501953125
        print([height_left,height_right])
        if not math.isnan(height_left) and not math.isnan(height_right):
            t1.append(height_left)
            t2.append(height_right)
            t3.append(distance1)
            t4.append(distance2)

#        #Recording of data
#        if height!=0 :
#            height1.append(real_height)
#            t1.append(distance1)
#            t2.append(distance3)
            #Turn this off when you don't want to record some value. Only for debugging or for checking the data
#            if distance1<count:
#                count = distance1
#                record = Area1
            
        #Put the results            
#        cv2.putText(color_image,'middle {} '.format(distance3), (100,250),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
#        cv2.putText(color_image,'Height {} '.format(real_height), (100,350),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        

        cv2.imshow('Found Red', rescale(color_image,50))

finally:
    pipeline.stop() #Turn off the camera
    cv2.destroyAllWindows() #Remove all the windows

#Create a dataframe 
df = pd.DataFrame()
#Store the data gathered in a dataframe for better view and handling
for i in range(len(t1)):
    temp = {"heightleft":t1[i],"heightright":t2[i],"d1":t3[i],"d2":t4[i]}
    df=df.append(temp,ignore_index=True)
    
    

