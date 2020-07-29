function output = bwLowPassFilter(im, params)
%% Low pass filtering using Butterworth filter
FT = @(x) fftshift(fft2(ifftshift(x)));
IFT = @(x) fftshift(ifft2(ifftshift(x)));

cutOffFreq = params.cutOffFreq;
scatMaskPitch = params.scatMaskPitch;
filterDimension = params.filterDimension;
outCplx = params.outCplx;

% Fourier domain for the image plane
% Magnification is ignored.
Nh = size(im, 1); Nw = size(im, 2);
Lh = Nh * scatMaskPitch; Lw = Nw * scatMaskPitch;
minX = -Lw / 2; maxX = Lw / 2;
minY = -Lh / 2; maxY = Lh / 2;
xVec = linspace(minX, maxX, Nw + 1); xVec(end) = [];
yVec = linspace(minY, maxY, Nh + 1); yVec(end) = [];

[X, Y] = meshgrid(xVec, yVec);
bwFilter = (X.^2 + Y.^2) ./ (cutOffFreq^2);
bwFilter = 1 ./ (1 + bwFilter.^filterDimension);

imFourier = FT(im);
imFilter = bwFilter .* imFourier;

if outCplx
    output = IFT(imFilter);
else
    output = abs(IFT(imFilter));
end
end