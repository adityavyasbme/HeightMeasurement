
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 11:35:38 2019

@author: AVyas
"""
import copy

def findCheckerboardCoordinates(image):
    try:
        grayR= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(grayR, (9,6))
        if found:
            return corners[:,0,:],True
        else:
            return [],False
    except:
        print("Error In finding Co-Ordinates")
        return [],False


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
    x,y,w,h = int(x*1.03),int(y*1.03),int(w*0.5),int(0.5*h)
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



def get_device_fy(config):
    from realsense_device_manager import DeviceManager    
    device_manager = DeviceManager(rs.context(), config)
    device_manager.enable_all_devices()
    # Allow some frames for the auto-exposure controller to stablise
    for frame in range(30):
        frames = device_manager.poll_frames()
    assert( len(device_manager._available_devices) > 0 )
    intrinsics_devices = device_manager.get_device_intrinsics(frames)
    temp = intrinsics_devices['908212070032'] 
    temp2 = temp[rs.stream.color].__getattribute__('fy')
    device_manager.disable_streams()
    return temp2



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

answer=[]
height_data=[]

global t1,t2,t3

t1,t2,t3=[],[],[]

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

        cv2.line(color_image,(0,175),(1280,175),(255,0,0),5)

        cv2.line(color_image,(0,450),(1280,450),(255,0,0),5)
        cv2.imshow('Found Red', rescale(color_image,50))

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

#answer.remove(answer[0])


