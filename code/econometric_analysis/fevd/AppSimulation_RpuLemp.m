%% Application in Gorodnichenko and Lee (2019)
% Section 5, RPU and Monetary Policy shocks on RGDP and Price

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

timeseries = data{T1:T2,["rpu","lsp","ffr","lemp","lgdp"]};

[T,N] = size(timeseries);

y = timeseries(:,4);
XRPU = timeseries;

% #3.Estimators correcting for biases based on a VAR model

% VAR
[RPUpsiVAR, RPUpsiCbVAR, RPUpsiSeVAR, RPUvdVAR, RPUvdCbVAR, RPUvdSeVAR] ...
    = var_chol(timeseries, lagVAR, horizon, NBootstrap, 1, 0.9, [], 0, 1);

% Save VAR results
writematrix(RPUpsiVAR, ['output/lemp/RPUpsiVAR_h' num2str(horizon) '.csv']);
writematrix(RPUpsiCbVAR, ['output/lemp/RPUpsiCbVAR_h' num2str(horizon) '.csv']);
writematrix(RPUpsiSeVAR, ['output/lemp/RPUpsiSeVAR_h' num2str(horizon) '.csv']);
writematrix(RPUvdVAR, ['output/lemp/RPUvdVAR_h' num2str(horizon) '.csv']);
writematrix(RPUvdCbVAR, ['output/lemp/RPUvdCbVAR_h' num2str(horizon) '.csv']);
writematrix(RPUvdSeVAR, ['output/lemp/RPUvdSeVAR_h' num2str(horizon) '.csv']);

% RPU - y
% R2
lagy = 0;
lagX = [3,3,3,3,3];
XVAROrder = [1,2,3,5];
yPlace = 4;
XtIndicatorRPU = [1, 0, 0, 0, 0];

[ RPUyvdR2, RPUyvdBcR2, RPUyvdSeR2, RPUyvdBcCbR2, RPUydetailsR2] ...
    = BootstrapVARR2( y, XRPU, horizon, ylevely, trend, lagy, lagX, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );

% LP A
 [ RPUyvdLPA, RPUyvdBcLPA, RPUyvdSeLPA, RPUyvdBcCbLPA, RPUydetailsLPA] ...
    = BootstrapVARLPA( y, XRPU, horizon, ylevely, trend, lagy, lagX, XtIndicatorRPU, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );

% LP B
 [ RPUyvdLPB, RPUyvdBcLPB, RPUyvdSeLPB, RPUyvdBcCbLPB, RPUydetailsLPB] ...
    = BootstrapVARLPB( y, XRPU, horizon, ylevely, trend, lagy, lagX, XtIndicatorRPU, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );

% Save R2-based results
writematrix(RPUyvdBcR2, ['output/lemp/RPUyvdBcR2_h' num2str(horizon) '.csv']);
writematrix(RPUyvdSeR2, ['output/lemp/RPUyvdSeR2_h' num2str(horizon) '.csv']);
writematrix(RPUyvdBcCbR2, ['output/lemp/RPUyvdBcCbR2_h' num2str(horizon) '.csv']);


% #4. Estimators correcting for biases based on simulating asymptotic
% distributions


% RPU - y
% R2
XtIndicatorRPU = [1, 0, 0, 0, 0];
lagyLP = 1;
lagXLP = [1,0,0,0,0];

[ RPUyvdAVarR2, RPUyvdBcAVarR2, RPUyvdSeAVarR2, RPUyvdBcAVarCbR2, RPUydetailsAVarR2] ...
    = SimulationAVarR2( y, XRPU, horizon, ylevely, trend, lagy, lagX, ...
    NBootstrap, quantiles, seIndicator, cbIndicator, boundIndicator );

% LP A
[ RPUyvdAVarLPA, RPUyvdBcAVarLPA, RPUyvdSeAVarLPA, RPUyvdBcAVarCbLPA, RPUydetailsAVarLPA] ...
    = SimulationAVarLPA( y, XRPU, horizon, ylevely, trend, lagy, lagX, XtIndicatorRPU, ...
    NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
    jointIndicator, XRPU, lagyLP, lagXLP, XtIndicatorRPU );


% LP B
[ RPUyvdAVarLPB, RPUyvdBcAVarLPB, RPUyvdSeAVarLPB, RPUyvdBcAVarCbLPB, RPUydetailsAVarLPB] ...
    = SimulationAVarLPB( y, XRPU, horizon, ylevely, trend, lagy, lagX, XtIndicatorRPU, ...
    NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
    jointIndicator, XRPU, lagyLP, lagXLP, XtIndicatorRPU );



% VAR - VAR

RPUpsiVARb = zeros(N, horizon+1, NBootstrap);
RPUvdVARb  = zeros(N, horizon+1, NBootstrap);

[~, ~, ~, ~, ~, ~, ~, ...
    cVAR, phiVAR, ~, ~, ~, ~, residualVAR] ...
    = var_chol(timeseries, lagVAR, horizon, NBootstrap, 1, 0.9, [], 0, 0);


for nb = 1:NBootstrap
    [timeseriesb] = var_bootstrap(cVAR, phiVAR, timeseries, residualVAR, T, TBurnIn);
    
    [RPUpsiVARb(:,:,nb), ~, ~, RPUvdVARb(:,:,nb)] ...
        = var_chol(timeseries, lagVAR, horizon, NBootstrap, 1, 0.9, [], 0, 0);

end

% bias-corrected VAR based vd
RPUvdVARBc  = 2 * RPUvdVAR - mean(RPUvdVARb,3);

% bootstrapped standard error
RPUpsiSeVAR = std(RPUpsiVARb, 0, 3);
RPUvdSeVAR  = std(RPUvdVARb, 0, 3);

% bootstrapped confidence interval in deviation. by adding the estimate, we
% can construct an interval centered around the estimate.
dRPUpsiVARCb = quantile(RPUpsiVARb - repmat(mean(RPUpsiVARb,3), [1,1,NBootstrap]), quantiles, 3); % (y variable, horizon, quantile), deviation from the mean
dRPUvdVARCb  = quantile(RPUvdVARb - repmat(mean(RPUvdVARb,3), [1,1,NBootstrap]), quantiles, 3); % (y variable, horizon, quantile), deviation from the mean


% #5. Reshaping

% ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls), LP-(lagyLP, lagXLP only x)] ), ir
irRPUy                 = [ RPUpsiVAR(2,:)', RPUydetailsLPB.stdx  * RPUydetailsLPB.ir,   RPUydetailsAVarLPB.stdx   * RPUydetailsAVarLPB.ir  ];

% ( quantile, h, [VAR, LP-(VAR equivalent set-up with controls)] ), ir - bootstrapped confidence band 
irRPUyCb               = zeros(length(quantiles), horizon+1, 2);

for idxq = 1:length(quantiles)
    irRPUyCb(idxq,:,:)  = [ RPUpsiVAR(2,:)' + dRPUpsiVARCb(2,:,idxq)', RPUydetailsLPB.irStrCb(idxq,:)' ];

end

% ( idxn, h, [VAR-Bootstrap, LP-(VAR equivalent set-up with controls, VAR Bootstrap), LP-(VAR equivalent set-up with controls, Asymptotic variance),
%                            LP-(lagyLP, lagXLP only x, Asymptotic variance)] ), s.e.
irSeRPUy               = [ RPUpsiSeVAR(2,:)', RPUydetailsLPB.stdx  * RPUydetailsLPB.irSeBootstrap,  RPUydetailsLPB.stdx  * RPUydetailsLPB.irSeAVar,  RPUydetailsAVarLPB.stdx  * RPUydetailsAVarLPB.irSeAVar  ];



% ( idxn, h, [VAR, R2, LP A/B-(VAR equivalent set-up with controls),
% LP A/B-(lagyLP, lagXLP only x)] ), vd w/o bias correction
vdRPUy                  = [ RPUvdVAR(2,:)', RPUyvdR2,  RPUyvdLPA,  RPUyvdLPB,  RPUyvdAVarLPA,  RPUyvdAVarLPB ];

% ( quantile, h, [VAR, R2, LPA, LPB] ), vd w/ bias correction - bootstrapped confidence band 
vdRPUyCb               = zeros(length(quantiles), horizon+1, 4);

for idxq = 1:length(quantiles)
    vdRPUyCb(idxq,:,:)  = [ RPUvdVAR(2,:)' + dRPUvdVARCb(2,:,idxq)', RPUyvdR2  + (RPUyvdBcCbR2(idxq,:)'  - RPUyvdBcR2) , RPUyvdLPA  + (RPUyvdBcCbLPA(idxq,:)'  - RPUyvdBcLPA) , RPUyvdLPB  + (RPUyvdBcCbLPB(idxq,:)'  - RPUyvdBcLPB)  ];
end


% ( idxn, h, [VAR, R2, LP A/B] ), vd w/ bias correction by simulating the estimated VAR model
vdVARBcRPUy             = [ RPUvdVARBc(2,:)', RPUyvdBcR2,  RPUyvdBcLPA,  RPUyvdBcLPB  ];

% ( quantile, h, [VAR, R2, LPA, LPB] ), vd w/ bias correction - bootstrapped confidence band 
vdVARBcRPUyCb               = zeros(length(quantiles), horizon+1, 4);

for idxq = 1:length(quantiles)
    vdVARBcRPUyCb(idxq,:,:)  = [ RPUvdVARBc(2,:)' + dRPUvdVARCb(2,:,idxq)', RPUyvdBcCbR2(idxq,:)' , RPUyvdBcCbLPA(idxq,:)' , RPUyvdBcCbLPB(idxq,:)'  ];
end


% ( idxn, h, [VAR, R2, LP A/B] ), s.e. by simulating the estimated VAR model
vdVARSeBootstrapRPUy    = [ RPUvdSeVAR(2,:)', RPUyvdSeR2,  RPUyvdSeLPA,  RPUyvdSeLPB  ];

% ( idxn, h, [R2, LP A/B] ), vd w/ bias correction by simulating the asymptotic distribution
vdAVarBcRPUy            = [ RPUyvdBcAVarR2,   RPUyvdBcAVarLPA,  RPUyvdBcAVarLPB  ];

% ( idxn, h, [R2, LP A/B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.
vdAVarSeRPUy            = [ RPUyvdSeAVarR2,  RPUyvdSeAVarLPA,  RPUyvdSeAVarLPB  ];

% ( idxn, h, [R2, LP A/B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.
vdAVarSeSimulationRPUy            = [ RPUydetailsAVarR2.vdSeBootstrap, RPUydetailsAVarLPA.vdSeBootstrap, RPUydetailsAVarLPB.vdSeBootstrap ];


% Save
%eval(['save App_T_' num2str(T1) '_' num2str(T2) ' ir* vd* horizon'])

writematrix(RPUpsiVAR, ['output/lemp/RPUpsiVAR_h' num2str(horizon) '.csv']);
writematrix(RPUpsiSeVAR, ['output/lemp/RPUpsiSeVAR_h' num2str(horizon) '.csv']);
writematrix(RPUvdVAR, ['output/lemp/RPUvdVAR_h' num2str(horizon) '.csv']);
writematrix(RPUvdSeVAR, ['output/lemp/RPUvdSeVAR_h' num2str(horizon) '.csv']);

writematrix(RPUyvdBcR2, ['output/lemp/RPUyvdBcR2_h' num2str(horizon) '.csv']);
writematrix(RPUyvdBcAVarCbR2, ['output/lemp/RPUyvdBcAVarCbR2_h' num2str(horizon) '.csv']);

