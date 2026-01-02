%% Application in Gorodnichenko and Lee (2019)
% Section 5, LM and Monetary Policy shocks on RGDP and Price

% # 0. Adding library
addpath([pwd '/lib'])       % 'pwd' = current folder

% #1. Parameters
lagVARSet = 1:12;           % lag length used for bias correction.
lagVAR = 3;                 % For VAR based estimates
horizon = 12;
NBootstrap = 2000;
TBurnIn = 100;
ylevely = 1;                % Delta y on the RHS if ylevel == 0, y on the RHS if ylevel == 1.
trend = 0;                  % Because we estimate models with Delta y, we include only intercepts as trends.
dfIndicator = 0;            % degrees of freedom adjustment when we estimate the second moments asymptotically
ICIndicator = 1;
quantiles = [0.05, 0.95];
seIndicator = 0;
cbIndicator = 0;
boundIndicator = 1;
jointIndicator = 0;


% #2. Data 
T1 = 1;
T2 = 444;

data = readtable('data_for_lp.xls');

timeseries = data{T1:T2,["lm","lsp","ffr","lemp","lgdp"]};

[T,N] = size(timeseries);

y = timeseries(:,4);
XLM = timeseries;

% #3.Estimators correcting for biases based on a VAR model

% VAR
[LMpsiVAR, LMpsiCbVAR, LMpsiSeVAR, LMvdVAR, LMvdCbVAR, LMvdSeVAR] ...
    = var_chol(timeseries, lagVAR, horizon, NBootstrap, 1, 0.9, [], 0, 1);

% Save VAR results
writematrix(LMpsiVAR, ['output/lemp/LMpsiVAR_h' num2str(horizon) '.csv']);
writematrix(LMpsiCbVAR, ['output/lemp/LMpsiCbVAR_h' num2str(horizon) '.csv']);
writematrix(LMpsiSeVAR, ['output/lemp/LMpsiSeVAR_h' num2str(horizon) '.csv']);
writematrix(LMvdVAR, ['output/lemp/LMvdVAR_h' num2str(horizon) '.csv']);
writematrix(LMvdCbVAR, ['output/lemp/LMvdCbVAR_h' num2str(horizon) '.csv']);
writematrix(LMvdSeVAR, ['output/lemp/LMvdSeVAR_h' num2str(horizon) '.csv']);

% LM - y
% R2
lagy = 0;
lagX = [3,3,3,3,3];
XVAROrder = [1,2,3,5];
yPlace = 4;
XtIndicatorLM = [1, 0, 0, 0, 0];

[ LMyvdR2, LMyvdBcR2, LMyvdSeR2, LMyvdBcCbR2, LMydetailsR2] ...
    = BootstrapVARR2( y, XLM, horizon, ylevely, trend, lagy, lagX, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );

% LP A
 [ LMyvdLPA, LMyvdBcLPA, LMyvdSeLPA, LMyvdBcCbLPA, LMydetailsLPA] ...
    = BootstrapVARLPA( y, XLM, horizon, ylevely, trend, lagy, lagX, XtIndicatorLM, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );

% LP B
 [ LMyvdLPB, LMyvdBcLPB, LMyvdSeLPB, LMyvdBcCbLPB, LMydetailsLPB] ...
    = BootstrapVARLPB( y, XLM, horizon, ylevely, trend, lagy, lagX, XtIndicatorLM, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );

% Save R2-based results
writematrix(LMyvdBcR2, ['output/lemp/LMyvdBcR2_h' num2str(horizon) '.csv']);
writematrix(LMyvdSeR2, ['output/lemp/LMyvdSeR2_h' num2str(horizon) '.csv']);
writematrix(LMyvdBcCbR2, ['output/lemp/LMyvdBcCbR2_h' num2str(horizon) '.csv']);


% #4. Estimators correcting for biases based on simulating asymptotic
% distributions


% LM - y
% R2
XtIndicatorLM = [1, 0, 0, 0, 0];
lagyLP = 1;
lagXLP = [1,0,0,0,0];

[ LMyvdAVarR2, LMyvdBcAVarR2, LMyvdSeAVarR2, LMyvdBcAVarCbR2, LMydetailsAVarR2] ...
    = SimulationAVarR2( y, XLM, horizon, ylevely, trend, lagy, lagX, ...
    NBootstrap, quantiles, seIndicator, cbIndicator, boundIndicator );

% LP A
[ LMyvdAVarLPA, LMyvdBcAVarLPA, LMyvdSeAVarLPA, LMyvdBcAVarCbLPA, LMydetailsAVarLPA] ...
    = SimulationAVarLPA( y, XLM, horizon, ylevely, trend, lagy, lagX, XtIndicatorLM, ...
    NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
    jointIndicator, XLM, lagyLP, lagXLP, XtIndicatorLM );


% LP B
[ LMyvdAVarLPB, LMyvdBcAVarLPB, LMyvdSeAVarLPB, LMyvdBcAVarCbLPB, LMydetailsAVarLPB] ...
    = SimulationAVarLPB( y, XLM, horizon, ylevely, trend, lagy, lagX, XtIndicatorLM, ...
    NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
    jointIndicator, XLM, lagyLP, lagXLP, XtIndicatorLM );



% VAR - VAR

LMpsiVARb = zeros(N, horizon+1, NBootstrap);
LMvdVARb  = zeros(N, horizon+1, NBootstrap);

[~, ~, ~, ~, ~, ~, ~, ...
    cVAR, phiVAR, ~, ~, ~, ~, residualVAR] ...
    = var_chol(timeseries, lagVAR, horizon, NBootstrap, 1, 0.9, [], 0, 0);


for nb = 1:NBootstrap
    [timeseriesb] = var_bootstrap(cVAR, phiVAR, timeseries, residualVAR, T, TBurnIn);
    
    [LMpsiVARb(:,:,nb), ~, ~, LMvdVARb(:,:,nb)] ...
        = var_chol(timeseries, lagVAR, horizon, NBootstrap, 1, 0.9, [], 0, 0);

end

% bias-corrected VAR based vd
LMvdVARBc  = 2 * LMvdVAR - mean(LMvdVARb,3);

% bootstrapped standard error
LMpsiSeVAR = std(LMpsiVARb, 0, 3);
LMvdSeVAR  = std(LMvdVARb, 0, 3);

% bootstrapped confidence interval in deviation. by adding the estimate, we
% can construct an interval centered around the estimate.
dLMpsiVARCb = quantile(LMpsiVARb - repmat(mean(LMpsiVARb,3), [1,1,NBootstrap]), quantiles, 3); % (y variable, horizon, quantile), deviation from the mean
dLMvdVARCb  = quantile(LMvdVARb - repmat(mean(LMvdVARb,3), [1,1,NBootstrap]), quantiles, 3); % (y variable, horizon, quantile), deviation from the mean


% #5. Reshaping

% ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls), LP-(lagyLP, lagXLP only x)] ), ir
irLMy                 = [ LMpsiVAR(2,:)', LMydetailsLPB.stdx  * LMydetailsLPB.ir,   LMydetailsAVarLPB.stdx   * LMydetailsAVarLPB.ir  ];

% ( quantile, h, [VAR, LP-(VAR equivalent set-up with controls)] ), ir - bootstrapped confidence band 
irLMyCb               = zeros(length(quantiles), horizon+1, 2);

for idxq = 1:length(quantiles)
    irLMyCb(idxq,:,:)  = [ LMpsiVAR(2,:)' + dLMpsiVARCb(2,:,idxq)', LMydetailsLPB.irStrCb(idxq,:)' ];

end

% ( idxn, h, [VAR-Bootstrap, LP-(VAR equivalent set-up with controls, VAR Bootstrap), LP-(VAR equivalent set-up with controls, Asymptotic variance),
%                            LP-(lagyLP, lagXLP only x, Asymptotic variance)] ), s.e.
irSeLMy               = [ LMpsiSeVAR(2,:)', LMydetailsLPB.stdx  * LMydetailsLPB.irSeBootstrap,  LMydetailsLPB.stdx  * LMydetailsLPB.irSeAVar,  LMydetailsAVarLPB.stdx  * LMydetailsAVarLPB.irSeAVar  ];



% ( idxn, h, [VAR, R2, LP A/B-(VAR equivalent set-up with controls),
% LP A/B-(lagyLP, lagXLP only x)] ), vd w/o bias correction
vdLMy                  = [ LMvdVAR(2,:)', LMyvdR2,  LMyvdLPA,  LMyvdLPB,  LMyvdAVarLPA,  LMyvdAVarLPB ];

% ( quantile, h, [VAR, R2, LPA, LPB] ), vd w/ bias correction - bootstrapped confidence band 
vdLMyCb               = zeros(length(quantiles), horizon+1, 4);

for idxq = 1:length(quantiles)
    vdLMyCb(idxq,:,:)  = [ LMvdVAR(2,:)' + dLMvdVARCb(2,:,idxq)', LMyvdR2  + (LMyvdBcCbR2(idxq,:)'  - LMyvdBcR2) , LMyvdLPA  + (LMyvdBcCbLPA(idxq,:)'  - LMyvdBcLPA) , LMyvdLPB  + (LMyvdBcCbLPB(idxq,:)'  - LMyvdBcLPB)  ];
end


% ( idxn, h, [VAR, R2, LP A/B] ), vd w/ bias correction by simulating the estimated VAR model
vdVARBcLMy             = [ LMvdVARBc(2,:)', LMyvdBcR2,  LMyvdBcLPA,  LMyvdBcLPB  ];

% ( quantile, h, [VAR, R2, LPA, LPB] ), vd w/ bias correction - bootstrapped confidence band 
vdVARBcLMyCb               = zeros(length(quantiles), horizon+1, 4);

for idxq = 1:length(quantiles)
    vdVARBcLMyCb(idxq,:,:)  = [ LMvdVARBc(2,:)' + dLMvdVARCb(2,:,idxq)', LMyvdBcCbR2(idxq,:)' , LMyvdBcCbLPA(idxq,:)' , LMyvdBcCbLPB(idxq,:)'  ];
end


% ( idxn, h, [VAR, R2, LP A/B] ), s.e. by simulating the estimated VAR model
vdVARSeBootstrapLMy    = [ LMvdSeVAR(2,:)', LMyvdSeR2,  LMyvdSeLPA,  LMyvdSeLPB  ];

% ( idxn, h, [R2, LP A/B] ), vd w/ bias correction by simulating the asymptotic distribution
vdAVarBcLMy            = [ LMyvdBcAVarR2,   LMyvdBcAVarLPA,  LMyvdBcAVarLPB  ];

% ( idxn, h, [R2, LP A/B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.
vdAVarSeLMy            = [ LMyvdSeAVarR2,  LMyvdSeAVarLPA,  LMyvdSeAVarLPB  ];

% ( idxn, h, [R2, LP A/B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.
vdAVarSeSimulationLMy            = [ LMydetailsAVarR2.vdSeBootstrap, LMydetailsAVarLPA.vdSeBootstrap, LMydetailsAVarLPB.vdSeBootstrap ];


% Save
%eval(['save App_T_' num2str(T1) '_' num2str(T2) ' ir* vd* horizon'])

writematrix(LMpsiVAR, ['output/lemp/LMpsiVAR_h' num2str(horizon) '.csv']);
writematrix(LMpsiSeVAR, ['output/lemp/LMpsiSeVAR_h' num2str(horizon) '.csv']);
writematrix(LMvdVAR, ['output/lemp/LMvdVAR_h' num2str(horizon) '.csv']);
writematrix(LMvdSeVAR, ['output/lemp/LMvdSeVAR_h' num2str(horizon) '.csv']);

writematrix(LMyvdBcR2, ['output/lemp/LMyvdBcR2_h' num2str(horizon) '.csv']);
writematrix(LMyvdBcAVarCbR2, ['output/lemp/LMyvdBcAVarCbR2_h' num2str(horizon) '.csv']);

