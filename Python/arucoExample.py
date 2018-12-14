import numpy as np        # `pip install numpy`
import cv2                # `pip install opencv-python`
import cv2.aruco as aruco # `pip install opencv-contrib-python`
import leapuvc            #  Ensure leapuvc.py is in this folder

# This examples tracks the pose of an ArUco Cube; the template for the cube can be found at the bottom of the file

# Initialize ArUco Tracking
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100 )
parameters =  aruco.DetectorParameters_create()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
markerWidth = 0.045 # This value is in meters

# Define Aruco board, which can be any 3D shape. See helper CAD file @ https://cad.onshape.com/documents/d51fdec31f121f572b802b11/w/83fac6aaee78bdc978fd804d/e/8ae3ae505e4af3c7402b131a
board_ids = np.array([[94], [95], [96], [97], [98], [99]], dtype=np.int32)
board_corners = [
     np.array([[-0.022, 0.023, 0.03],  [ 0.023, 0.022, 0.03],  [ 0.023,-0.023, 0.03],  [-0.022,-0.023, 0.03]],  dtype=np.float32),
     np.array([[-0.022,-0.03,  0.022], [ 0.023,-0.03,  0.022], [ 0.022,-0.03, -0.022], [-0.022,-0.03, -0.022]], dtype=np.float32),
     np.array([[-0.03, -0.023, 0.022], [-0.03, -0.022,-0.023], [-0.03,  0.023,-0.022], [-0.03,  0.023, 0.023]], dtype=np.float32),
     np.array([[-0.022,-0.022,-0.03],  [ 0.023,-0.023,-0.03],  [ 0.023, 0.023,-0.03],  [-0.022, 0.023,-0.03]],  dtype=np.float32),
     np.array([[ 0.03, -0.023,-0.022], [ 0.03, -0.023, 0.023], [ 0.03,  0.023, 0.022], [ 0.03,  0.022,-0.023]], dtype=np.float32),
     np.array([[-0.022, 0.03, -0.023], [ 0.023, 0.03, -0.022], [ 0.023, 0.03,  0.023], [-0.022, 0.03,  0.022]], dtype=np.float32)
]
board = aruco.Board_create( board_corners, aruco_dict, board_ids )

# Start the Leap Capture Thread
leap = leapuvc.leapImageThread(resolution=(640, 480))
leap.start()
leap.setExposure(2000)
leap.setGain(20)

# Capture images until 'q' is pressed
while((not (cv2.waitKey(1) & 0xFF == ord('q'))) and leap.running):
    newFrame, leftRightImage = leap.read()
    if(newFrame):
        for i, cam in enumerate(leap.cameras):
            # Only track in left camera for speed
            if(cam == 'right'):
                break

            corners, ids, rejectedImgPoints = aruco.detectMarkers(leftRightImage[i], aruco_dict, parameters=parameters)

            if(corners is not None):
                camera_matrix = leap.calibration[cam]["extrinsics"]["cameraMatrix"]
                dist_coeffs   = leap.calibration[cam]["intrinsics"]["distCoeffs"]
                retval, rvec, tvec = aruco.estimatePoseBoard( corners, ids, board, camera_matrix, dist_coeffs )

                # Convert to color for drawing
                colorFrame = None
                colorFrame = cv2.cvtColor(leftRightImage[i], cv2.COLOR_GRAY2BGR)
                if(retval):
                    colorFrame = aruco.drawAxis( colorFrame, camera_matrix, dist_coeffs, rvec, tvec, markerWidth/2 )
                colorFrame = aruco.drawDetectedMarkers(colorFrame, corners, ids)

            # Display the resulting Frames
            cv2.imshow(cam + ' Frame', colorFrame)

cv2.destroyAllWindows()




# Draw Paper Template for the board
def drawPaperTemplate():
    paperPxWidth = 300
    paperWidth = 0.2159 # This value is in meters, for 8.5x11" paper
    faceWidth  = 0.06   # This value is in meters
    cubeTemplate = np.ones((int(paperPxWidth*11/8.5),paperPxWidth))*255
    markerPx = int((markerWidth/paperWidth)*cubeTemplate.shape[1]/2)
    facePx   = int((faceWidth/paperWidth)*cubeTemplate.shape[1]/2)
    def drawMarkerAt(id, markerCenter):
        marker = aruco.drawMarker(aruco_dict, id, markerPx*2, borderBits=1)	
        paddedMarker = np.ones((facePx*2, facePx*2))*255
        padding = facePx-markerPx
        paddedMarker[padding:(facePx*2)-padding, 
                     padding:(facePx*2)-padding] = marker
        cv2.rectangle(paddedMarker, (0,0), (paddedMarker.shape[0]-1, paddedMarker.shape[1]-1), 0, 1)
        cubeTemplate[markerCenter[1] - facePx:markerCenter[1] + facePx, 
                     markerCenter[0] - facePx:markerCenter[0] + facePx] = paddedMarker

    drawMarkerAt(board_ids[0], (int(cubeTemplate.shape[1]/2), facePx))
    drawMarkerAt(board_ids[1], (int(cubeTemplate.shape[1]/2), facePx + 1*facePx*2))
    drawMarkerAt(board_ids[2], (int(cubeTemplate.shape[1]/2) - facePx*2, facePx + 2*facePx*2))
    drawMarkerAt(board_ids[3], (int(cubeTemplate.shape[1]/2), facePx + 2*facePx*2))
    drawMarkerAt(board_ids[4], (int(cubeTemplate.shape[1]/2) + facePx*2, facePx + 2*facePx*2))
    drawMarkerAt(board_ids[5], (int(cubeTemplate.shape[1]/2), facePx + 3*facePx*2))
    return cubeTemplate

# Draw the paper template for the board
#cv2.imshow('Template', drawPaperTemplate())
#while(not (cv2.waitKey(1) & 0xFF == ord('q'))):
#    pass
#cv2.destroyAllWindows()