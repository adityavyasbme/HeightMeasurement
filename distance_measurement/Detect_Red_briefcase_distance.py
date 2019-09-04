# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 13:01:58 2019

@author: AVyas
"""

import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2


def rescale(img,amount):
    scale_percent = amount # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def thresholding(img):
    lower_red = np.array([0,200,0]) #example value
    upper_red = np.array([255,255,255]) #example value
    mask = cv2.inRange(img, lower_red, upper_red)
    img_result = cv2.bitwise_and(img, img, mask=mask)
    return  img_result


def find_red(image):
#    image = cv2.imread(path)
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s=thresholding(hsv_img)
    s= cv2.cvtColor(s, cv2.COLOR_HSV2BGR)
    s= cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
    ret, s= cv2.threshold(s, 0, 255, 0)
    _,contours, hierarchy = cv2.findContours(s,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(s, contours, -1, (0, 0, 0), 3)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    x,y,w,h = int(x*1.03),int(y*1.03),int(w*0.7),int(0.7*h)
    cv2.rectangle(s,(x,y),(x+w,y+h),(255,255,255),5)
    coordinatex,coordinatey = int(x+(w/2)),int(y+(h/2))
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
    cv2.circle(image,(coordinatex,coordinatey), 5, (0,255,255), -1)
    return image,[x,y,w,h]

def find_mean(stack):
    (rows,columns)=np.shape(stack)
    count,tsum = 0,0
    for i in range(rows):
        for j in range(columns):
            if stack[i][j] != 0 :
                tsum+=stack[i][j]
                count+=1
    if count==0 or tsum==0:
        return 0
    return float(tsum/count)


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
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 5 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

answer=[]

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        image1,[x,y,w,h]=find_red(color_image.copy())
        cv2.imshow('Found Red', rescale(image1,50))

        cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,0),5)

        Area = depth_image[y:y+h,x:x+w,] 
        distance=find_mean(Area)*depth_scale*39.3701 #inches
#        distance -= (4.2*39.3701/1000)  #depth measurement starts from 4.2 mm behind the camera
        answer.append(distance)

        print("distance is : {} inches".format(distance))    


        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', rescale(images,50))

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

answer.remove(answer[0])



import statistics as stat

print(stat.mean(answer))

print(stat.stdev(answer))


