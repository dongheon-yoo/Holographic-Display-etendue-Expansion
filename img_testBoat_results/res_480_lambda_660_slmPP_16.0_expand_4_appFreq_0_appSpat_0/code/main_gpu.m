close all; clear all;
addpath ./fminlbfgs_version2c ./util

FT = @(x) fftshift(fft2(ifftshift(x)));
IFT = @(x) fftshift(ifft2(ifftshift(x)));
targetImName = 'testBoat.jpg';

%% Basic system specifications
applyFreqConst = false;
applySpatConst = false;
mm = 1e-3; um = 1e-6; nm = 1e-9;
lambda = 660 * nm;
k = 2 * pi / lambda;
slmPitch = 16 * um;
scatMaskPitch = 4 * um;
expandRatio = round(slmPitch / scatMaskPitch);

% Define planes
slmNh = 1080 / 4; slmNw = 1920 / 4;
targetH = slmNh * expandRatio; targetW = slmNw * expandRatio;

% Results directory
[~, targetImBodyName, ~] = fileparts(targetImName);
resultDirName = sprintf('./img_%s_results', targetImBodyName);
resultDirName2 = sprintf('/res_%d_lambda_%03d_slmPP_%0.1f_expand_%d_appFreq_%d_appSpat_%d', ...
                            slmNw, ...
                            floor(lambda * 1e9), ...
                            slmPitch * 1e6, ...
                            round(expandRatio), ...
                            double(applyFreqConst), ...
                            double(applySpatConst));
mkdir([resultDirName resultDirName2]);

%% Target image
targetIm = imresize(imread(targetImName), [targetH, targetW]);
targetIm = im2double(rgb2gray(targetIm));
params.localIm = targetIm;
targetIm = convertGPU(targetIm);

% Low pass filtering for target image
cutOffFreq = scatMaskPitch .* sqrt(slmNh * slmNw / pi);
filterDimension = 5;
filteringParams.filterDimension = filterDimension;
filteringParams.cutOffFreq = cutOffFreq;
filteringParams.scatMaskPitch = scatMaskPitch;
filteringParams.outCplx = false;
targetImLPF = bwLowPassFilter(targetIm, filteringParams);
targetImLPF = targetImLPF ./ max(targetImLPF(:));
params.targetImLPF = convertGPU(targetImLPF);

figure;
imshow(targetIm);
title 'Original target image'

figure;
imshow(targetImLPF);
title 'Filtered target image'

%% Define scattering mask
scatMaskH = round(expandRatio * slmNh); 
scatMaskW = round(expandRatio * slmNw);
scatMaskPhase = rand(scatMaskH, scatMaskW);
scatMaskPhase = round(scatMaskPhase) * pi;
scatMaskField = exp(1.j .* scatMaskPhase);

%% Initialization
% Initialization is critical for optimization
% According to the article, it is initialized with One-step
% backpropagation.
% If it is initialized with just random phase, stucked in local minimum.
targetField = sqrt(targetIm) .* exp(1.j .* 2 * pi * rand(size(targetIm)));
slmPhase = IFT(targetField) ./ (scatMaskPitch^2) .* conj(scatMaskField);
slmPhase = downSampleGPU(angle(slmPhase), 1 / expandRatio);

%% Define spatial filtering for foveation
% Magnification = lambda * focal length is ignored.
xVec = 1 : targetW; yVec = 1 : targetH;
xVec = xVec - mean(xVec); yVec = yVec - mean(yVec);
[X, Y] = meshgrid(xVec, yVec);

% Depends on total FOV
subFOVRatioToX = 0.3;
cutOffLength = subFOVRatioToX * max(xVec);
radius = sqrt(X.^2 + Y.^2);
spatFilter = double(radius < cutOffLength);

% standard deviation should be carefully designed
spatFilter = imgaussfilt(spatFilter, 10);
% According to the article.
filtThreshold = 0.1;
spatFilter = max(spatFilter, filtThreshold);

%% Define optimization parameters
params.slmNh = slmNh;
params.slmNw = slmNw;
params.lambda = lambda;
params.slmPitch = slmPitch;
params.steps_per_plot = 50;
params.dirname = [resultDirName resultDirName2];
params.scatMaskPhase = scatMaskPhase;
params.expandRatio = expandRatio;
params.scatMaskPitch = scatMaskPitch;
params.applyFreqConst = applyFreqConst;
params.applySpatConst = applySpatConst;
params.spatFilter = spatFilter;

options = optimset('fminunc');
options.Display = 'iter';
options.HessUpdate = 'bfgs';        
options.GoalsExactAchieve = 0;
options.GradObj = 'on';
options.TolFun = 1e-20;
options.FunValCheck = 'on';
options.DerivativeCheck = 'off';
options.MaxIter = 2000;
options.MaxFunEvals = 1e7;
options.TolX = 1e-20;

%% Save code as text files
codeDir = fullfile(resultDirName, resultDirName2, 'code');
mkdir(codeDir);
mainCodeDir = './main_gpu.m';
saveCodeAsText(mainCodeDir, codeDir);

mFileLists = dir('util');
for i = 1 : length(mFileLists)
    filename = mFileLists(i).name;
    [~, ~, ext] = fileparts(filename);
    if strcmp(ext, '.m')
        folder = mFileLists(i).folder;
        origCodeDir = fullfile(folder, filename);
        saveCodeAsText(origCodeDir, codeDir);
    end
end

%% Optimization starts
[optimPhase, history] = runopt_wgrad_lbfgs_gpu(slmPhase, targetIm, params, options);
