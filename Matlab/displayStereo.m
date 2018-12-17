function displayStereo(left,right)

imagesc([left, right]);     % Display the two images side by side
caxis([0 255]);             % 0 is black and 255 is white (disable this line to auto-scale brightness)
colormap(gray(255));        % Display images in grayscale
drawnow;                    % Update the image viewer

end
