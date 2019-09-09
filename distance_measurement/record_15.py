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


def mouse_callback(event, x, y, flags, params):
    global ix
    if event ==1:
        print("LEFT_CLICK_MADE") 
        ix=True
    if event == 2:
        ix=False
        print([x,y])
        

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
clipping_distance_in_meters = 6 #10 meter
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

out = cv2.VideoWriter('data/read.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,780))


#Some Variables
count = 0
answer = {}
cv2.namedWindow("W")
cv2.setMouseCallback("W",mouse_callback)
ix=False


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

        depth_image[depth_image>5000]=0
        
        color_image = np.asanyarray(color_frame.get_data()) #Getting final RGB frame
        
        if count == 0:
            answer[count]=[depth_image,color_image] #0 is the first image. can be used for background subtraction
            count+=1
            continue

        if ix:
            color_image1,check,values=apply_cascade(color_image.copy(),'models/cascades/haarcascade_lefteye_2splits.xml')
            if check:
                answer[count]=[depth_image,color_image]
                count+=1
        cv2.imshow('W', rescale(color_image,90))
        if len(answer)==15:
            break

finally:
    pipeline.stop() #Turn off the camera
    cv2.destroyAllWindows() #Remove all the windows
    out.release()

import pickle    
output = open('data/record_15.pkl', 'wb')
pickle.dump(answer, output)
output.close()

