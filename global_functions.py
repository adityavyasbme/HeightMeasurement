# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:55:29 2019

@author: AVyas
"""
#CV library
import cv2
#mathmeatical functions library
import numpy as np
#To copy data
import copy
#Importing RealSense SDK
import pyrealsense2 as rs


"""
findCheckerboardCoordinates will take an image and will return 54(9x6) cordinates of the checkerboard
Args:
    Img : Image
Returns :
    list of 54 coordinates
"""
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
"""
find_checker_height can be updated and we can use it to find the height of one checker
Useful when we have multiple checkerboard
"""
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

"""
Rescale Function is used to resize the image so that it can be visibile in small resolution
Args : 
    Img : Image to be reduced
    amount : by how much amount 
return :
    image of reduced size
"""
def rescale(img,amount):
    scale_percent = amount # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized


"""
Thresholding function is set to red color right now 
Args:
    Img : Image
Returns :
    Thresolded Image
"""
def thresholding(img):
    lower_red = np.array([0,200,0]) #example value
    upper_red = np.array([255,255,255]) #example value
    mask = cv2.inRange(img, lower_red, upper_red)
    img_result = cv2.bitwise_and(img, img, mask=mask)
    return  img_result

"""
Find_red will detect the largest red area
Args:
    Img : Image
Returns :
    Image : thresholded 
    [x,y,w,h] : [x,y] coordinates of rectangle and [w,h] represents width and height 
"""

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


"""
get_device_fy will take the configuration and take some images and then it will return the focal length in y direction
Args:
    config : Realsense Configuration
Returns :
    temp2 : focal_length in y direction
"""
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

"""
find_concurrent will take a list and create a dictionary then 
it will return the key with maximum value
Args:
    a : list 
Returns :
    number which occurs most in the list
"""
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

"""
remove value can be used as a thresholder for a list
Args:
    a : list 
    val : (int) value to be removed/threshold value
    sign: (str) depends on the user what he wants to do i.e. :
        '==' will remove the value
        '<' '>' can be used as thresholder
Returns :
    updated list
"""
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

"""
find_dist will take the data and cut off values outside of region mean+-2*standard deviation
Args : 
    a: (list)
Returns : 
    updated_list

"""
def find_dist(x):
#    tot = len(x)
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
"""
iterate will use find_dist function and continue to update the list untill there are no changes
Args : 
    a: (list) not in 1D
Returns : 
    updated_list
"""
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
        

"""
takes two coordinates and returns 
Args:
    a : cordinates 1
    b : coridnates 2
Returns:
    two point distance
"""
def two_point_distance(a,b):
    return ((b[1]-a[1])**2 + (b[0]-a[0])**2)**0.5



"""
APPLY_CASCADE can be used to apply different haar cascasdes
Arg: 
    original : image
    cascade: location of the model
Returns : 
    Image if found else None
    True if found else False
"""
def apply_cascade(original,cascade):
#    height = np.size(original, 0)
#    width = np.size(original, 1)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    apply_cascade1 = cv2.CascadeClassifier(cascade)
    try:
        faces = apply_cascade1.detectMultiScale(gray, 1.3, 5)
#        p,q=0,0
        for i in faces:
            (x,y,w,h)=i
#            cv2.rectangle(original,(x,y),(x+w,y+h),(255,0,0),2)
#            cv2.line(original,(int(x+w/2),0),(int(x+w/2),height),(255,255,255),1)
#            p = x+(w/2)
#            q = y+(h/2)
        return original,True,faces
    except:
        return None,False,None
