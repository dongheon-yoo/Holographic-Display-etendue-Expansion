%% Run image based optimization via L-BFGS method 
function [optimPhase, history] = runopt_wgrad_lbfgs_gpu(slmPhase, targetIm, params, options)
% Basic parameters
slmNh = params.slmNh; slmNw = params.slmNw;
targetImLPF = params.targetImLPF;
targetImLPF = extractGPU(targetImLPF);
history.mseVal = []; history.psnrVal = []; history.iter = [];
options.OutputFcn = @lbfgs_outfun;

% Start optimization
optimFunc = @(x)loss_and_gradients(x, targetIm, params);

% For recording
f1 = figure; f2 = figure;
[optimPhase, ~] = fminlbfgs_gpu(optimFunc, slmPhase, options);
    
    function stop = lbfgs_outfun(x, optimValues, state)
        stop = false;
        switch state
            case 'init'
            case 'iter'
                % Record & Plot
                if rem(optimValues.iteration, params.steps_per_plot) == 0
                    phi = reshape(x, [slmNh, slmNw]);
                    phi = angle(exp(1.j .* phi));
                    phi = (phi + pi) / (2 * pi);
                    local_phi = extractGPU(phi);
                    phase_name = [params.dirname, sprintf('/phase_%dIter.png', optimValues.iteration)];
                    imwrite(uint8(local_phi * 255), phase_name);
                    
                    IProp = extractGPU(reconFromPhase(x, params));
                    I_name = sprintf('/recon_%dIter.png', optimValues.iteration);
                    I_name = [params.dirname, I_name];
                    imwrite(uint8(IProp * 255), I_name);
                    pval = psnr(IProp, targetImLPF);
                    history.psnrVal = [history.psnrVal; pval];
                    history.mseVal = [history.mseVal; optimValues.fval];
                    
                    history.iter = [history.iter; optimValues.iteration];
                    figure(f1);
                    plot(history.iter, history.psnrVal);
                    xlabel('Iterations'); ylabel('PSNR (dB)');
                    title('PSNR (dB)');
                    saveas(f1,[params.dirname, './psnr.png']);
                    
                    figure(f2);
                    plot(history.iter, history.mseVal);
                    xlabel('Iterations'); ylabel('MSE');
                    title('L2 Loss');
                    saveas(f2,[params.dirname, './L2loss.png']);
   
                end
               
            case 'done'
            otherwise
        end
    end
end

