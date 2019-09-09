# Height Measurement

Imports :
 - global_functions

Files : 

* detect3checkers --> Used To detect 3 checkerboards in the image and find the distance at each one of them
* detect_3checkers_support_file --> Used to find thresholding values
* checkerboardheightmeasurement --> This can be used to find the dimension of one checkerboard. It is kind of incomplete in sense that we have to modify the code as per the requirement.

Note: 

For a new environment you have to find the area where the checkerboards are and then threshold accordingly. Therefore, to help you with this, I created a support file that can be used to draw lines in a live image.
Find the cv2.line functions and get the values of area accordingly.

Also, detect3checkers can be updated to n number of checkerboards.


