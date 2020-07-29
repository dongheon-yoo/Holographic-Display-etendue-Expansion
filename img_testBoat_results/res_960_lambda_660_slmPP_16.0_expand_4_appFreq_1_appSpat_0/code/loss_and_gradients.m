function [loss, gradients] = loss_and_gradients(slmPhase, targetIm, params)

%% Compute loss (L2)                                            
% Basic parameters
FT = @(x) fftshift(fft2(ifftshift(x)));
IFT = @(x) fftshift(ifft2(ifftshift(x)));
applyFreqConst = params.applyFreqConst;
applySpatConst = params.applySpatConst;
spatFilter = params.spatFilter;

slmNh = params.slmNh; slmNw = params.slmNw;
expandRatio = params.expandRatio;
scatMaskPhase = params.scatMaskPhase;
scatMaskPitch = params.scatMaskPitch;

% Upsampling
slmPhase = reshape(slmPhase, [slmNh, slmNw]);
slmPhaseUp = upSampleGPU(slmPhase, expandRatio);

% Reconstruction
slmField = exp(1.j .* slmPhaseUp);
scatMaskField = exp(1.j .* scatMaskPhase);
fieldComplx = slmField .* scatMaskField;
reconField = FT(fieldComplx) * (scatMaskPitch^2);
reconIm = abs(reconField) .^ 2;
reconMeanShift = reconIm ./ sum(reconIm, 'all') .* sum(targetIm, 'all');

% No constraint
if applyFreqConst
    % Apply frequency constraint
    intDiff = (reconMeanShift - targetIm);
    cutOffFreq = scatMaskPitch .* sqrt(numel(slmPhase) / pi);
    filterDimension = 5;
    filteringParams.filterDimension = filterDimension;
    filteringParams.cutOffFreq = cutOffFreq;
    filteringParams.scatMaskPitch = scatMaskPitch;
    filteringParams.outCplx = true;
    intDiffLPF = bwLowPassFilter(intDiff, filteringParams);
    if applySpatConst
        intDiffLPF = intDiffLPF .* spatFilter;
    end
    loss = 0.5 * mean(abs(intDiffLPF).^2, 'all');
else
    % No constraint
    loss = 0.5 * mean((targetIm - reconMeanShift) .^ 2, 'all');
end

%% Compute gradients
if nargout > 1 % gradient required
    if applyFreqConst
        if applySpatConst
            intDiffLPF = intDiffLPF .* spatFilter;
        end
        filteringParams.outCplx = true;
        dL_dI = bwLowPassFilter(intDiffLPF, filteringParams) / numel(targetIm);
    else
        dL_dI = intDiff / numel(targetIm);
    end
    
    gradients = reconField .* dL_dI;
	gradients = gradients .* sum(targetIm, 'all') .* ...
                (sum(reconIm, 'all') - reconIm) ./ (sum(reconIm, 'all') ^ 2);
    gradients = IFT(gradients) / (scatMaskPitch^2);
    gradients = real(gradients .* conj(fieldComplx .* 1.j));
    gradients = downSampleGPU(gradients, 1 / expandRatio);
end
