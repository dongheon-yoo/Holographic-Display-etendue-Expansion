function output = upSampleGPU(im, scale);
%% Upsampling through box filter
% Scale should be integer.

originalHeight = size(im, 1); newHeight = round(scale) * originalHeight;
originalWidth = size(im, 2); newWidth = round(scale) * originalWidth;
imVec = im(:).';
% Row scaling
output = repmat(imVec, [round(scale), 1]);
output = reshape(output, [newHeight, originalWidth]);

% Column scaling
output = output.'; output = output(:).';
output = repmat(output, [round(scale), 1]);
output = reshape(output, [newWidth, newHeight]).';
end