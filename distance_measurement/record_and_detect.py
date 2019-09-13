
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:57:53 2019

@author: AVyas

THIS CODE IS COMBINATION OF HUMAN_DETECTION.PY AND RECORD_15 FOR EASE OF USE.
GO THROUGHT README FOR HOW TO CAPTURE/OBSERVE DATA.
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

#FACE DLIB ITERATION
from imutils import face_utils
import dlib
import imutils
from collections import OrderedDict

#pickle for data storage
import pickle 

def mouse_callback(event, x, y, flags, params):
    global ix
    if event ==1:
        print("LEFT_CLICK_MADE") 
        ix=True
    if event == 2:
        ix=False
        print([x,y])
        
def record_15():
    global ix
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

def find_highest(image): #unit8 single layer
    lowest=0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j]>lowest:
                return [j,i]
    return [0,0]            

def find_lowest(image): #unit8 single layer
    lowest=0
    for i in range(image.shape[0]-1,-1,-1):
        for j in range(image.shape[1]-1,-1,-1):
            if image[i][j]>lowest:
                return [j,i]
    return [0,0]        
    
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
    
    
def get_area(image,list1,tuple_list,depthimage):
    (j,k) = tuple_list
    pts = list1[j:k]
    hull = cv2.convexHull(pts)
#    cv2.drawContours(image, [hull], -1, (19,199,109), -1)
    rect = cv2.boundingRect(hull)
    cv2.rectangle(image,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),3)

    area = depthimage[rect[1]-3:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    return image,area,rect[0]

    
    

#divider    
#%%
 
def detect():
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
            
            image,area,loc1 = get_area(color.copy(),dic1[0],FACIAL_LANDMARKS_IDXS["right_eye"],depth)        
            size = area.shape[0]*area.shape[1]
            re = iterate(area)*depth_scale*100
    
            image,area,loc2 = get_area(color.copy(),dic1[0],FACIAL_LANDMARKS_IDXS["left_eye"],depth)        
            size2 = area.shape[0]*area.shape[1]
            le = iterate(area)*depth_scale*100
            
            if min([size,size2])/max([size,size2])<0.8:
                continue
#            print([size,size2])
    
    
            if np.isnan(le) or np.isnan(re):
                continue
    
            loc = min(loc1,loc2)
            error = -1
            le+=error
            re+=error
    
            threshold = 70
            image3 = cv2.subtract(first_color,color)
            ret,a= cv2.threshold(image3,threshold,255,cv2.THRESH_BINARY)
            grayImage = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            grayImage[grayImage>threshold]=255
            grayImage[grayImage<=threshold]=0
            [x1,y1]=find_highest(grayImage)
            [x2,y2]=find_lowest(grayImage)
            height = abs(y2-y1)
            cv2.line(color,(0,y1),(1280,y1),(0,255,255),2)    
            cv2.line(color,(0,y2),(1280,y2),(0,255,255),2)  
            real_height1 = le*height/916.364501953125
            real_height2 = re*height/916.364501953125
            expected = 177.8*916.364501953125/height
            cv2.imshow("image",rescale(color,75))
            temp = {"Eye_l":le,"Eye_R":re,"pixel_height":height,'expected':expected,'location':loc,'E_R':real_height2,'E_L':real_height1}
            df=df.append(temp,ignore_index=True)
            print(temp)
    
    finally:
        cv2.destroyAllWindows()
        return df
#DIVIDER
#%%
record_15()

#DIVIDER
#%%
df = detect()