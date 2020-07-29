function output = downSampleGPU(im, scale);
%% Downsample through box filter.
% Scale should be 1 / integer.
reduceScale = round(1 / scale);
originalHeight = size(im, 1); 
originalWidth = size(im, 2); 
assert(rem(originalHeight, reduceScale) == 0 && ...
       rem(originalWidth, reduceScale) == 0)

newHeight = originalHeight / reduceScale;
newWidth = originalWidth / reduceScale;

% Row scaling
imTemp = reshape(im, reduceScale, []);
outputRowSquz = reshape(sum(imTemp, 1), newHeight, []);

% Column scaling
outputColSquz = reshape(outputRowSquz.', reduceScale, []);
outputColSquz = sum(outputColSquz, 1);
output = reshape(outputColSquz, [newWidth, newHeight]).' / (reduceScale^2);

end