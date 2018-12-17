function [left, right] = getLeapFrame(vid)

ycbcr = getsnapshot(vid);

% Convert the ycbcr output to stereo grayscale
left = ycbcr(:,:,1); right = left;
right(:,1:2:end) = ycbcr(:,1:2:end,2);
right(:,2:2:end) = ycbcr(:,2:2:end,3);

