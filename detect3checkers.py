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
    
    frame2[0:450,:]=(0,0,0)
    
    frame1[250:719,:]=(0,0,0)
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

def find_concurrent(a):
    global dic
    dic={}
    for i in a:
        if i not in dic:
            dic[i]=1
        else:
            dic[i]+=1
    dic[0]=0
    return int(max(dic,key=dic.get))    


def remove_value(a,val,sign):
    temp=[]
    for i in a:
        if sign == "==":
            if i!=val:
                temp.append(i)
        if sign == ">":
            if i>val:
                temp.append(i)
        if sign == "<":
            if i<val:
                temp.append(i)
    return temp

def find_dist(x):
    tot = len(x)

    a = remove_value(x,0,'==')

    m = np.mean(a)
    sd = np.std(a)

    lower = m-(sd*2)
    upper= m+(sd*2)

    removed_lower = remove_value(a,lower,'>')
    removed_upper = remove_value(removed_lower,upper,'<')
#    print(np.mean(removed_upper))
#    print(len(removed_upper))
#    print("--------")
    return removed_upper

def iterate(a):
    a = a.flatten()
    temp_check=len(a)
    while True:
        a=find_dist(a)
        if temp_check==len(a):
            break
        else:
            temp_check=len(a)
    return np.mean(a)
        
        

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
height1 = []
# Streaming loop

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
colorizer = rs.colorizer()



count = 1000
record = []

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

#        aligned_depth_frame = dec_filter.process(aligned_depth_frame)
        aligned_depth_frame = spat_filter.process(aligned_depth_frame)
        aligned_depth_frame = temp_filter.process(aligned_depth_frame)        
        aligned_depth_frame = hole_filling.process(aligned_depth_frame)        

        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        cv2.imshow("colorized depth",rescale(colorized_depth,50))


        color_image = np.asanyarray(color_frame.get_data())


        
        img1=color_image.copy()
        img1[250:719,:]=(0,0,0)
        a1,check1= findCheckerboardCoordinates(img1)
        if not check1:
            print("Check 1")
            cv2.imshow('Found Red', rescale(color_image,50))

            continue
#        cv2.rectangle(color_image,tuple(a1[10]),tuple(a1[0]),(0,255,0),5)

        img2=color_image.copy()
        img2[0:450,:]=(0,0,0)
        a2,check2= findCheckerboardCoordinates(img2)
        if not check2:
            print("Check 2")
            cv2.imshow('Found Red', rescale(color_image,50))


            continue
#        cv2.rectangle(color_image,tuple(a2[10]),tuple(a2[0]),(0,255,0),5)


        img3=color_image.copy()
        img3[0:250,:]=(0,0,0)
        img3[450:719,:]=(0,0,0)
        
        a3,check3= findCheckerboardCoordinates(img3)
        if not check3:
            print("Check 3")
            cv2.imshow('Found Red', rescale(color_image,50))

            continue
#        cv2.rectangle(color_image,tuple(a3[10]),tuple(a3[0]),(0,255,0),5)

        
        x1= a1[10][0]
        y1= a1[10][1]
        w1= abs(a1[10][0] - a1[0][0] )
        h1= abs(a1[10][1] - a1[0][1] )
        
        Area1 = depth_image[int(y1):int(a1[0][1]+100),int(x1)-50:int(a1[0][0])+50]
        

        x2= a2[10][0]
        y2= a2[10][1]
        w2= abs(a2[10][0] - a2[0][0] )
        h2= abs(a2[10][1] - a2[0][1] )

        Area2 = depth_image[int(y2)-100:int(a2[0][1]),int(x2)-50:int(a2[0][0])+50]


        x3= a3[10][0]
        y3= a3[10][1]
        w3= abs(a3[10][0] - a3[0][0] )
        h3= abs(a3[10][1] - a3[0][1] )

        Area3 = depth_image[int(y3)-50:int(a3[0][1]+50),int(x3)-50:int(a3[0][0])+50]



#        Area = depth_image[y:y+h,x:x+w,] 
        distance1=iterate(Area1)*depth_scale*39.3701 #inches
        distance2=iterate(Area2)*depth_scale*39.3701 #inches
        distance3=iterate(Area3)*depth_scale*39.3701 #inches
        
        

        print([distance1,distance3,distance2])
        
        if distance1==0.0  or distance3==0.0:
            cv2.imshow('Found Red', rescale(color_image,50))

            continue

        
        


        check,height=find_c_height(color_image)
        real_height= distance3*height/916.364501953125

        if height!=0 :
            height1.append(real_height)
            t1.append(distance1)
            t2.append(distance3)
            t3.append(distance2)
            
            if distance1<count:
                count = distance1
                record = Area1
            
            
        cv2.putText(color_image,'middle {} '.format(distance3), (100,250),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(color_image,'Lower {} '.format(distance2), (100,300),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        cv2.putText(color_image,'Height {} '.format(real_height), (100,350),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        

        cv2.imshow('Found Red', rescale(color_image,50))



finally:
    pipeline.stop()
    cv2.destroyAllWindows()

#answer.remove(answer[0])



import statistics as stat

#print(stat.mean(answer))
#
#print(stat.stdev(answer))


def two_point_distance(a,b):
    return ((b[1]-a[1])**2 + (b[0]-a[0])**2)**0.5

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



import pandas as pd

df = pd.DataFrame()

for i in range(len(t1)):
    temp = {"up":t1[i],"middle":t2[i],"down":t3[i],"height1":height1[i]}
    df=df.append(temp,ignore_index=True)
    
    

