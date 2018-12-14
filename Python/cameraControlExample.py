import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import leapuvc            #  Ensure leapuvc.py is in this folder

# Start the Leap Capture Thread
leap = leapuvc.leapImageThread()
leap.start()

# Define Various Camera Control Settings
cv2.namedWindow   ('Settings')
cv2.createTrackbar('Rectify',   'Settings', 0, 1,  lambda a:0)                # Applies image rectification
cv2.createTrackbar('Exposure',  'Settings', 1000,  32222, leap.setExposure)   # Sets the exposure time in microseconds
cv2.createTrackbar('LEDs',      'Settings', 1, 1,  lambda a: (leap.setLeftLED(a), leap.setCenterLED(a), leap.setRightLED(a))) # Turns on the IR LEDs
cv2.createTrackbar('Gamma',     'Settings', 1, 1,  leap.setGammaEnabled)      # Applies a sqrt(x) contrast-reducing curve in 10-bit space
cv2.createTrackbar('Anlg Gain', 'Settings', 0, 63, leap.setGain)              # Amplifies the signal in analog space, 16-63
cv2.createTrackbar('Dgtl Gain', 'Settings', 0, 16, leap.setDigitalGain)       # Digitally amplifies the signal in 10-bit space
cv2.createTrackbar('HDR',       'Settings', 0, 1,  leap.setHDR)               # Selectively reduces the exposure of bright areas at the cost of fixed-pattern noise
#cv2.createTrackbar('Rotate',    'Settings', 0, 1,  leap.set180Rotation)       # Rotates each camera image in-place 180 degrees (need to unflip when using calibrations!)
#cv2.createTrackbar('V Offset',  'Settings', 0, 240, leap.setVerticalCenter)   # Control the Vertical Offset of the camera Image (need to offset when using calibrations!)

# Capture images until 'q' is pressed
while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    newFrame, leftRightImage = leap.read()
    if(newFrame):
        for i, cam in enumerate(leap.cameras):
            if(cv2.getTrackbarPos('Rectify','Settings')):
                # Rectify each image
                maps = leap.calibration[cam]["undistortMaps"]
                leftRightImage[i] = cv2.remap(leftRightImage[i], maps[0], maps[1], cv2.INTER_LINEAR)

            # Display the raw frame
            cv2.imshow(cam + ' Frame', leftRightImage[i])

cv2.destroyAllWindows()