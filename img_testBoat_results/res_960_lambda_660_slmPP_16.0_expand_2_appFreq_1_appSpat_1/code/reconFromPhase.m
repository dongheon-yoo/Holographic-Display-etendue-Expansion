function IProp = reconFromPhase(slmPhase, params)

FT = @(x) fftshift(fft2(ifftshift(x)));

expandRatio = params.expandRatio;
scatMaskPhase = params.scatMaskPhase;
scatMaskPitch = params.scatMaskPitch;
targetImLPF = params.targetImLPF;

%% Reconstruction from phase
slmNh = params.slmNh; slmNw = params.slmNw;

% Upsampling
slmPhase = reshape(slmPhase, [slmNh, slmNw]);
slmPhaseUp = upSampleGPU(slmPhase, expandRatio);

% Reconstruction
slmField = exp(1.j .* slmPhaseUp);
scatMaskField = exp(1.j .* scatMaskPhase);
fieldComplx = slmField .* scatMaskField;
reconField = FT(fieldComplx) * (scatMaskPitch^2);
IProp = abs(reconField) .^ 2;

% Low pass filtering
cutOffFreq = scatMaskPitch .* sqrt(numel(slmPhase) / pi);
filterDimension = 5;
filteringParams.filterDimension = filterDimension;
filteringParams.cutOffFreq = cutOffFreq;
filteringParams.scatMaskPitch = scatMaskPitch;
filteringParams.outCplx = false;
IProp = bwLowPassFilter(IProp, filteringParams);
IProp = IProp ./ sum(IProp, 'all') .* sum(targetImLPF, 'all');
IProp = max(min(IProp, 1), 0);

end
