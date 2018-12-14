import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import leapuvc            #  Ensure leapuvc.py is in this folder

# Start the Leap Capture Thread
leap = leapuvc.leapImageThread(resolution=(640, 480))
leap.start()
leap.setExposure(16666)       # Sets the exposure time in microseconds
leap.setGain(63)              # Amplifies the signal in the image

# Define some trackbars to edit stereo matching parameters
def nothing(x):
    pass
cv2.namedWindow('Settings')
cv2.createTrackbar('Use SGBM','Settings',0, 1, nothing)
cv2.createTrackbar('BlockSize','Settings',5, 32, nothing)
cv2.createTrackbar('UQRatio','Settings', 70, 255, nothing)
cv2.createTrackbar('TexThresh','Settings', 248, 255, nothing)
cv2.createTrackbar('P1','Settings', 0, 320*3*3**2, nothing)
cv2.createTrackbar('P2','Settings', 100, 320*3*3**2, nothing)
cv2.createTrackbar('Exposure', 'Settings', 16666, 65535, leap.setExposure)

# Capture images until 'q' is pressed
while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    newFrame, leftRightImage = leap.read()
    if(newFrame):
        rectifiedImages = []
        for i, cam in enumerate(leap.cameras):
            # Rectify each image
            maps = leap.calibration[cam]["undistortMaps"]
            rectifiedImages.append(cv2.remap(leftRightImage[i], maps[0], maps[1], cv2.INTER_LANCZOS4))

        # Show combined frames with epilines
        #combinedFrames = np.hstack((rectifiedImages[0], rectifiedImages[1]))
        #for y in range(int(leap.resolution[0]*0.025)):
        #    cv2.line(combinedFrames, (0, y*40), (int(leap.resolution[0]*2), y*40), 255, 1)
        #cv2.imshow('Rectified Frames', combinedFrames)

        if(cv2.getTrackbarPos('Use SGBM','Settings') == 0):
            # Use a Stereo Block Matcher to solve for the stereo disparity (Fast!)

            # Fix the Block Size setting to be greater than 5 and odd
            blockSize = cv2.getTrackbarPos('BlockSize','Settings')
            blockSize = max(5, blockSize)
            if blockSize % 2 == 0:
                blockSize += 1

            stereo = cv2.StereoBM_create(numDisparities=32, 
                                         blockSize=blockSize)
            stereo.setUniquenessRatio(cv2.getTrackbarPos('UQRatio','Settings'))
            stereo.setTextureThreshold(cv2.getTrackbarPos('TexThresh','Settings'))
        else:
            # Use a Semi-Global Block Matcher to solve for the stereo disparity (Slow!)
            stereo = cv2.StereoSGBM_create(numDisparities=64, 
                                           blockSize = cv2.getTrackbarPos('BlockSize','Settings'),
                                           P1 = cv2.getTrackbarPos('P1','Settings'),
                                           P2 = cv2.getTrackbarPos('P2','Settings'),
                                           disp12MaxDiff = 255,
                                           uniquenessRatio = cv2.getTrackbarPos('UQRatio','Settings'),
                                           speckleWindowSize = 40,
                                           speckleRange = 255-cv2.getTrackbarPos('TexThresh','Settings'))

        disparity = (stereo.compute(rectifiedImages[0], rectifiedImages[1]).astype(np.float32) / 16.0) * 0.05
        

        # Display the stereo disparity image
        cv2.imshow('Disparity', disparity)

cv2.destroyAllWindows()