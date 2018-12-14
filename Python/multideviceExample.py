import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import leapuvc            #  Ensure leapuvc.py is in this folder

numLeaps = 2

# Start the Leap Capture Threads
leaps = []
for i in range(numLeaps):
    leaps.append(leapuvc.leapImageThread(source = i))
    leaps[i].start()

# Capture images until 'q' is pressed
while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leapuvc.allLeapsRunning(leaps)):
    for i, leap in enumerate(leaps):
        newFrame, leftRightImage = leap.read()
        if(newFrame):
            # Display the raw frame
            cv2.imshow('Leap '+str(i)+', Frame L', leftRightImage[0])
            cv2.imshow('Leap '+str(i)+', Frame R', leftRightImage[1])

cv2.destroyAllWindows()