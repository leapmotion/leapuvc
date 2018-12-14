import threading
import time
import struct

try:
    import numpy as np
    import cv2
    from scipy.optimize import curve_fit
except:
    print("ERROR LOADING MODULES: Ensure 'numpy', 'opencv-python', and 'scipy' are installed.")

class leapImageThread(threading.Thread):
    '''A dedicated thread that handles retrieving imagery from an unlocked Leap Motion Peripheral'''
    def __init__(self, source = 0, resolution = (640, 480), timeout=3.0):
        '''Initialize Leap Image Capture'''
        threading.Thread.__init__(self)
        self.source = source
        self.cam = cv2.VideoCapture(self.source)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, False) # Does not work reliably in DirectShow :(
        self.resolution = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                           int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame = None
        self.newFrame = False
        self.timeoutTimer = time.time()
        self.timeout = timeout
        self.calibration = retrieveLeapCalibration(self.cam, self.resolution)
        self.cameras = ['left', 'right']
        self.running = False
        self.doYUYConversion = source == cv2.CAP_DSHOW # Not implemented!
        self.embeddedLine = None
    def run(self):
        self.running = True
        while(time.time() - self.timeoutTimer < self.timeout):
            if self.cam.isOpened():
                rval, frame = self.cam.read()
                if(rval):
                    # Reshape our one-dimensional image into a proper side-by-side view of the Peripheral's feed
                    frame = np.reshape(frame, (self.resolution[1], self.resolution[0]*2))
                    self.embeddedLine = self.getEmbeddedLine(frame)
                    leftRightImage = np.empty((2, self.resolution[1], self.resolution[0]), dtype=np.uint8)
                    leftRightImage[0,:,:]  = frame[:,  ::2]
                    leftRightImage[1,:,:] = frame[:, 1::2]
                    self.frame = leftRightImage
                    self.newFrame = True
        print("Exiting Leap Image Thread!")
        self.running = False
    def read(self):
        '''Attempts to retrieve the latest leap image; also resets the timeout on the image thread'''
        if(self.running):
            newFrame = self.newFrame
            if(newFrame):
                self.timeoutTimer = time.time()
            self.newFrame = False
            return newFrame, self.frame
        else:
            return False, None
    def get(self, param, value):
        '''Gets a UVC parameter on the internal cv2.VideoCapture object. \n(param, value) -> (int)'''
        return self.cam.get(param, value)
    def set(self, param, value):
        '''Sets a UVC parameter on the internal cv2.VideoCapture object. \n(param, value) -> (ret)'''
        return self.cam.set(param, value)
    def setExposure(self, exposureUS):
        '''Sets the sensor's exposure in microseconds (up to 65535). \n(param, exposureUS) -> (ret)'''
        return self.cam.set(cv2.CAP_PROP_ZOOM, max(10, exposureUS))
    def setGammaEnabled(self, gammaEnabled):
        '''Sets whether the image will be in a non-linear color space approximating sqrt(x) (or a linear color space if gammaEnabled is False). \n(param, gammaEnabled) -> (ret)'''
        return self.cam.set(cv2.CAP_PROP_GAMMA, 1 if gammaEnabled else 0)
    def getEmbeddedLine(self, interleavedImage):
        '''Parse the embedded line data coming in from the peripheral image into a tuple of integers'''
        embeddedArray = interleavedImage[-1,(self.resolution[0]*2)-12:]
        label1 = int(embeddedArray[6] >> 4 & 0x1)
        label2 = int(((embeddedArray[2] & 0xF) << 4) + (embeddedArray[4] & 0xF))
        darkFrameInterval = max(label1, label2)
        darkFrameInterval &= 0x7F
        exposure1 = int(embeddedArray[6] & 0xF)
        exposure2 = int(embeddedArray[8] & 0x1F)
        exposure  = int((exposure1 << 5) + exposure2) # Loops in 512 increments
        gain = int(embeddedArray[10] & 0x1F) # Loops in 32 increments
        return (darkFrameInterval, exposure, gain)
    def openSettings(self):
        '''Opens a settings adjustment window *when using DirectShow on Windows*. \n(param) -> (ret)'''
        return self.set(cv2.CAP_PROP_SETTINGS, 0)
    def setLEDsHDRorRotate(self, selector, value):
        '''Sets HDR (0), 180 degree Rotation (1), the indicator LEDs (2, 3, 4), and vertical center/zoom (5, 6).  VZoom crashes the device at the moment. \n(selector, value) -> (ret)'''
        return self.set(cv2.CAP_PROP_CONTRAST, ((selector) | ((value) << 6)))
    def setHDR(self, enabled):
        '''Sets the HDR Parameter \n(enabled) -> (ret)'''
        return self.setLEDsHDRorRotate(0, 1 if enabled else 0)
    def set180Rotation(self, enabled):
        '''Flips the image 180 degrees (calibrations do NOT flip!) \n(enabled) -> (ret)'''
        return self.setLEDsHDRorRotate(1, 1 if enabled else 0)
    def setLeftLED(self, enabled):
        '''Controls the Left LED. \n(enabled) -> (ret)'''
        return self.setLEDsHDRorRotate(2, 1 if enabled else 0)
    def setCenterLED(self, enabled):
        '''Controls the Center LED. \n(enabled) -> (ret)'''
        return self.setLEDsHDRorRotate(3, 1 if enabled else 0)
    def setRightLED(self, enabled):
        '''Controls the Right LED. \n(enabled) -> (ret)'''
        return self.setLEDsHDRorRotate(4, 1 if enabled else 0)
    def setVerticalCenter(self, value):
        '''Changes the Vertical Center. \n(value) -> (ret)'''
        return self.setLEDsHDRorRotate(5, value)
    def crashDevice(self, value):
        '''Crashes the device by trying to set the Vertical Zoom.  Replugging the device in will reset it. \nDon't use this\n(value) -> (ret)'''
        return self.setLEDsHDRorRotate(6, value)
    def setDigitalGain(self, value):
        '''Changes the Digital Gain, between 0 and 16. \n(value) -> (ret)'''
        return self.set(cv2.CAP_PROP_BRIGHTNESS, value)
    def setGain(self, gain):
        '''Specifies the analog gain as a scalar, between 16 and 63. \n(enabled) -> (ret)'''
        return self.set(cv2.CAP_PROP_GAIN, gain)
        #return self.setGainFPSRatioOrDFrameInterval(0x4000, gain)

    # These parameters seem to be out-of-date
    #def setGainFPSRatioOrDFrameInterval(self, selector, value):
    #    '''Sets Analog Gain (0x4000), FPS Ratio (0x8000), or the Dark Frame Interval (0xc000). \n(selector, value) -> (ret)'''
    #    return self.set(cv2.CAP_PROP_GAIN,  ((selector) | ((value) & 0x3fff)))
    #def setFPSRatio(self, ratio):
    #    '''Adjusts framerate as a proportion of the maximum for the configured resolution, in 1/1000 increments. \n(ratio) -> (ret)'''
    #    return self.setGainFPSRatioOrDFrameInterval(0x8000, ratio)
    #def setLEDDarkFrameInterval(self, interval):
    #    '''Sets the ratio of frames where the LEDs are on vs off.  0 is always on, 1 is always off, 2 is every other frame, etc. \n(param, interval) -> (ret)'''
    #    return self.cam.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, interval) # Only works in DirectShow
    #    #return self.setGainFPSRatioOrDFrameInterval(0xc000, interval)


def retrieveLeapCalibration(cap, resolution):
    '''Retrieves the device calibration and calculates OpenCV distortion parameters.  (resolution) -> (calibrationDict)'''
    def normalWarp(r, k1, k2, k3, k4, k5, k6):
        '''The standard OpenCV Radial Distortion Function'''
        return ((1.0 + r*(k1 + r*(k2 + r*k3))) /
    	        (1.0 + r*(k4 + r*(k5 + r*k6))))

    def monotonicWarp1(r, k1, k2, k3, k4, k5, k6):
        '''A monotonic version of OpenCV's radial distortion function; necessary for inversion!'''
        kr = normalWarp(r, k1, k2, k3, k4, k5, k6)
        return r * (kr**3)

    def monotonicWarp2(r, k1, k2, k3, k4, k5, k6):
        '''In case the first fit fails, try this one!'''
        kr = normalWarp(r, k1, k2, k3, k4, k5, k6)
        return r * (kr**2)

    # Begin the property knocking routine
    calibrationBytes = []
    for i in range(100, 256):
        # 1) Set the Sharpness Value to the address in memory
        cap.set(cv2.CAP_PROP_SHARPNESS, i)
        # 2) Wait 4ms
        cv2.waitKey(4)
        # 3) Read the Saturation Value
        calibrationBytes.append(int(cap.get(cv2.CAP_PROP_SATURATION)))

    # 5) Convert the Raw Bytes into Raw Values
    # Sig, Sig, Ver, Score, Timestamp, Baseline, Q2Init
    # Left:  FocalLength, Offset^2, Tangential^2, Radial^6
    # Left:  FocalLength, x0, y0, CayleyRotation^3
    # Right: FocalLength, Offset^2, Tangential^2, Radial^6
    # Right: FocalLength, x0, y0, CayleyRotation^3
    # Checksum
    # 'B B B B I ff fffffffffff ffffff fffffffffff ffffff I'
    calibrationArray = struct.unpack('BBBBIffffffffffffffffffffffffffffffffffffI', bytes(calibrationBytes))

    # 6) Store raw values in a named dict
    cameras = ['left', 'right']
    calibration = {}
    byteOffset = 7
    for cam in cameras:
        calibration[cam] = {}
        calibration[cam]["intrinsics"] = {}
        calibration[cam]["intrinsics"]["focalLength"] = calibrationArray[byteOffset]
        calibration[cam]["intrinsics"]["offset"] = (calibrationArray[byteOffset+1], calibrationArray[byteOffset+2])
        calibration[cam]["intrinsics"]["tangential"] = (calibrationArray[byteOffset+3], calibrationArray[byteOffset+4])
        calibration[cam]["intrinsics"]["radial"] = [calibrationArray[byteOffset+5], calibrationArray[byteOffset+6], calibrationArray[byteOffset+7], calibrationArray[byteOffset+8], calibrationArray[byteOffset+9], calibrationArray[byteOffset+10]]
        calibration[cam]["extrinsics"] = {}
        calibration[cam]["extrinsics"]["focalLength"] = calibrationArray[byteOffset+11] # This focalLength is deprecated
        calibration[cam]["extrinsics"]["center"] = (calibrationArray[byteOffset+12], calibrationArray[byteOffset+13])
        calibration[cam]["extrinsics"]["rotation"] = [calibrationArray[byteOffset+14], calibrationArray[byteOffset+15], calibrationArray[byteOffset+16]]
        byteOffset += 17

        aspect = resolution[1]/480
        cameraMatrix = np.asarray([[calibration[cam]["intrinsics"]["focalLength"], 0.0,           320+calibration[cam]["intrinsics"]["offset"][0]],
                                   [0.0, calibration[cam]["intrinsics"]["focalLength"] * aspect, (240+calibration[cam]["intrinsics"]["offset"][1])*aspect],
                                   [0.0, 0.0, 1.0]], dtype=np.float32)
        calibration[cam]["extrinsics"]["cameraMatrix"] = cameraMatrix

        # The stored radial distortion values are inverted from their OpenCV counterparts; invert the radial distortion coefficients
        xdata = np.linspace(-0.99, -0.35, 33)
        xdata = (1.0 / (xdata** 2)) - 1.0
        try:
            ydata = monotonicWarp(xdata, *calibration[cam]["intrinsics"]["radial"])
            k, pcov = curve_fit(monotonicWarp, ydata, xdata)
        except:
            # The first fit failed, try another one...
            ydata = monotonicWarp2(xdata, *calibration[cam]["intrinsics"]["radial"])
            k, pcov = curve_fit(monotonicWarp2, ydata, xdata)
        calibration[cam]["intrinsics"]["inverseRadial"] = np.asarray(k)

        # Construct the OpenCV "distCoeffs" array, used for rectification!
        p = calibration["left"]["intrinsics"]["tangential"]
        distCoeffs = np.asarray([k[0], k[1], p[0], p[1], k[2], k[3], k[4], k[5]], dtype=np.float32)
        calibration[cam]["intrinsics"]["distCoeffs"] = distCoeffs

        # Construct Undistortion Maps (for convenience)
        calibration[cam]["extrinsics"]["r"] = CayleyTransform(-np.asarray(calibration[cam]["extrinsics"]["rotation"]))
        calibration[cam]["undistortMaps"] = [None, None]
        calibration[cam]["undistortMaps"][0], calibration[cam]["undistortMaps"][1] = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, calibration[cam]["extrinsics"]["r"], None, resolution, cv2.CV_8UC1 )

    return calibration

def CayleyTransform(p):
    '''Vector3 input, 3x3 Rotation Matrix output
http://sf-github.leap.corp/leapmotion/platform/blob/develop/source/AlgorithmUtility/MathUtil.cpp#L436
For a mathematical definition, see:
http://en.wikipedia.org/wiki/Rotation_matrix#Skew_parameters_via_Cayley.27s_formula'''
 
    # Compute terms
    x = p[0] * 2
    y = p[1] * 2
    z = p[2] * 2
    xx = p[0] * p[0]
    xy = p[0] * p[1] * 2
    yy = p[1] * p[1]
    yz = p[1] * p[2] * 2
    zz = p[2] * p[2]
    zx = p[2] * p[0] * 2
    # Save the matrix
    retval = np.matrix([[1 + xx - yy - zz, xy - z, zx + y],
                        [xy + z, 1 - xx + yy - zz, yz - x],
                        [zx - y, yz + x, 1 - xx - yy + zz]], dtype=np.float32)
    # Compute the denominator
    return retval / (1 + xx + yy + zz)

def allLeapsRunning(leaps):
    for leap in leaps:
        if leap.running is False:
            return False
    return True