# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:35:38 2019

@author: AVyas
"""

#importing Copy library to copy data
import copy
#Importing Some Global Functions
from global_functions import findCheckerboardCoordinates,rescale,thresholding,find_red,remove_value,iterate,two_point_distance
#Importing RealSense SDK
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2


"""
Find_c_height will take the image and divide it into two parts.
Then it will detect 2 checkerboards i.e. upper and lower
it will detect highest and lowest point and will return 
difference of those points
"""
def find_c_height(img):
    frame1 = copy.copy(img)
    frame2=copy.copy(frame1)
    frame2[0:359,:]=(0,0,0)
    frame1[360:720,:]=(0,0,0)
    a1,check1= findCheckerboardCoordinates(frame1)
    a2,check2=findCheckerboardCoordinates(frame2)
#    print([check1,check2])
    if check1 and check2:
        y1 = min([a1[i][1] for i in range(len(a1)) ])
        y2 = max([a2[i][1] for i in range(len(a1)) ])
#        cv2.line(img,(0,y1),(1920,y1),(0,255,255),2)    
#        cv2.line(img,(0,y2),(1920,y2),(0,255,255),2)
#        cv2.imshow("img",rescale(img,20))
        height = y2-y1
        print(height)
        return True,height
    return False,0

"""
find_mean will take the list and remove all the zeros and give the mean of the list
"""
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
clipping_distance_in_meters = 10 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

#Variables
answer=[]
height_data=[]

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

        image1,[x,y,w,h]=find_red(color_image.copy()) #get the rectangle coordinates



        
        
        cv2.imshow('Found Red', rescale(color_img,50))

        cv2.rectangle(color_image,(x,y),(x+w,y+h),(0,255,0),5)

        Area = depth_image[y:y+h,x:x+w,] 
        distance=find_mean(Area)*depth_scale*39.3701 #inches
        
#        distance -= (4.2*39.3701/1000)  #depth measurement starts from 4.2 mm behind the camera
        answer.append(distance)

        
        check,height=find_c_height(color_image.copy())
        
        real_height= distance*height/916.364501953125
#        print("Height is : {} inches".format(real_height)) 
        print([distance,height,real_height])
        if height!=0:
            height_data.append(real_height) 


#        # Remove background - Set pixels further than clipping_distance to grey
#        grey_color = 153
#        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
#        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop() #TURN OFF CAMERA
    cv2.destroyAllWindows() #TURN off all the windows

##Remove all the zero
#answer.remove(answer[0])

"""
The remaining code is for measuring the height of one checker. 
findcheckerheight and findcheckerheight2 are created to measure the height of checker of two different checkerboards.
"""

import statistics as stat
print(stat.mean(answer))
print(stat.stdev(answer))


def find_checker_height(img):
    frame1 = copy.copy(img)
    frame1[0:360,:]=(0,0,0)
    a1,check1= findCheckerboardCoordinates(frame1)
    a=[]
    if check1:
        for i in range(0,44,9):
            a.append(two_point_distance(a1[i],a1[i+9]))
        return True, a
    return False, a

def find_checker_height2(img):
    frame1 = copy.copy(img)
    frame1[360:719,:]=(0,0,0)
    a1,check1= findCheckerboardCoordinates(frame1)
    a=[]
    if check1:
        for i in range(0,44,9):
            a.append(two_point_distance(a1[i],a1[i+9]))
        return True, a
    return False, a


a1,check1= findCheckerboardCoordinates(color_image)


_,p2 = find_checker_height2(color_image)        
_,p = find_checker_height(color_image)    

print("-----------")

for i in p2:
    print(i)
#    print(i*s.mean(answer)/916.364501953125)
    
print("-----------")
for i in p:
    print(i)
#    print(i*s.mean(answer)/916.364501953125)
        


