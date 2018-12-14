% Set variables required for device aquisition
imageMode = 'YUY2_752x480'; % This is the image mode name
deviceNum = 2;              % This is the device number on the system

% Grab a video camera object 
vid = videoinput('winvideo', deviceNum, imageMode);

% Change some of the default camera aquisition settings
vid.ReturnedColorSpace = 'YCbCr';   % Set the returned colorspace to 'YCbCr'
triggerconfig(vid, 'manual');       % Set the trigger mode to manual

% Start camera streaming
start(vid);

while(1)
    
    % Get an image from the camera
    ycbcr = getsnapshot(vid);
    
    % Convert the ycbcr output to stereo grayscale
    left = ycbcr(:,:,1);
    right = uint8(zeros(size(left)));
    right(:,1:2:end) = ycbcr(:,1:2:end,2);
    right(:,2:2:end) = ycbcr(:,2:2:end,3);
    
    % Display the images
    subplot_tight(1,2,1);    imagesc(left);
    subplot_tight(1,2,2);    imagesc(right);
    
    caxis([0 255]);    colormap(gray(255));    
    drawnow;
    
end

% Stop the camera
stop(vid);