% ======== Image streaming example =======
% === Requires Image Aquisition Toolbox ==

% You can specify any resolution you want as long as the first (width)
% value is 752 or 640 pixels wide and the second value is 480, 240, or 120
% pixels tall

imageMode = 'YUY2_752x480'; % This is the image mode name 

% This is the device number on the system 
% (you will probably need to change this to 1 or 2)
deviceNum = 2;              
                            
% Start camera streaming
startLeapStreaming;

% Display images until user presses ctrl+c

while(1)
    
    % Get an image from the camera
    [left, right] = getLeapFrame(vid);
    
    % Display the images
    displayStereo(left,right);
    
end

