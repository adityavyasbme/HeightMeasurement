# Distance Measurement

Imports :
 - global_functions
 - data/record_15.pkl file

Exports :
 - data/record_15.pkl file

Files : 

* Detect_Red_briefcase_distance.py --> It can be used to find the distance of the red object. It's an old program, therefore need to be updated.
* human_detection.py --> This contain 3 different codes to interpret the human distance measurement.
* record_15.py --> This can be used to record 15 frames and store the data into data/record_15.pkl file. 
* record_and_detect.py --> I have combined human_detection and record_15 in a single python file for ease of use.

Note: 

To reproduce result:
1. open record_and_detect.
2. Run all functions
3. When running the record function, make sure no one is standing in front of camera and when you see a window pop up you can record the other 14 images
	3.1 Just stand in front of the camera, stare in the front and press the left click
	3.2 After it captures 14 frames it will store the file in data folder.
4. Now run the detect function. This function reads the pickle file we stored earlier and then interpret that data. 
5. Make sure the yellow lines you see on the windows detect the person propoerly. As they are used to measure the pixel height.
6. If the yellow lines are off, change the threshold value and see if it detect the top and lowest point of person propoerly or not.
7. The results are stored in dataframe which can be used for observations. 
8. For different poistion repeat the above steps.  
