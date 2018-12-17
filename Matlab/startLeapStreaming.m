% If the camera is already streaming then restart it

if exist('vid','var')   
    stop(vid);
else
    
    % If the camera isn't streaming then start everything up
    
    % Grab a video camera object
    vid = videoinput('winvideo', deviceNum, imageMode);
    
    % Change some of the default camera aquisition settings
    vid.ReturnedColorSpace = 'YCbCr';   % Set the returned colorspace to 'YCbCr'
    triggerconfig(vid, 'manual');       % Set the trigger mode to manual
end

% Start camera streaming
start(vid);