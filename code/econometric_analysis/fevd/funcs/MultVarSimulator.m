%Simulate the Smets and Wouters (2007) model and apply the methods
%introduced in Gorodnichenko and Lee (2019). 'Parfor' is used. So, using
% 'parpool' command would increase the efficiency of this code.
%
% NOTE: Need to specify T and lagFlag. Also, lagy and lagX are needed if
%   lagFlag == false.
%
% INPUT:
% idxy    = Out of seven endogenous variables, the selected one is
%   analyzed in relations to the specified exogenous variable.
%   1: y=RGDP, 2: c=Consumption, 3: inve=Investment, 4: w=Wage,
%   5: pinf=Price inflation, 6: r=Nominal interest rates, 7: lab=Employment
% Either 1 or 5 below
% idxx    = Among seven exogenous shocks, the selected one and its
%   relations with the selected endogenous variable is studied.
%   1: a=TFP, 2: b=Credit spread, 3: g=Government expenditure,
%   4: qs=Investment specific technology, 5: m=Monetary policy,
%   6: spinf=price mark up, 7: sw=wage mark up
% Fixed at [5] below
% idxobs  = Those variables are assumed to be in the information set.
%   1: y=GDP, 2: c=Consumption, 3: inve=Investment, 4: w=Wage,
%   5: pinf=Price inflation, 6: r=Nominal interest rates, 7: lab=Employment
%   Assumes that idxy is an element of idxobs.
% Fixed at [1,5,6] below
% horizon = impulse response coefficients and FEVDs are derived up to 
%   the horizon-th periods.
%
% Example of the information set:
%   idxy = 1, idxx = [5], idxobs = [1,5,6] -> Information set
%     = {MP shock, RGDP, Price Inflation, Nominal Interest Rate}.
%
% NReplication = replication size of the simulation study.
% T       = Consider length T vectors
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%	NBootstrap times to derive the bias and the standard errors.
% TBurnIn = First TBurnIn observations are dropped as burn-in
% lagVARSet   = Set of candidate lag lengthes. The lag length is determined
%   by information criteria. It should be either a column
%	or a row vector.
% lagy    = y(-1), ... , y(-lagy) will be included on the RHS to
%   derive the impulse response coefficients using the local projection
%   method. If ylevel == 0, then Delta y is used instead.
% lagX    = (1 * N) or (N * 1) vector. lagX(n) number of lagged values of
%  n-th variable in X is included in the forecasting error regression or
%  local projection.
%
% OUTPUT:
% Too Many. See descriptions in #3.1. Initialization.
% In the below,
%   ir     = impulse response coefficients in response to a one std shock
%   vd     = forecast error variance decompositions
%   Bc     = bias-corrected
%   Se     = Standard Error
%   Bootstrap / AVar / Simulation
%          = Bootstrap based / Asymptotic variance / simulation based
%          (Appendix A and B)

% #0.
% idxy = 1; % 1 = GDP, 5 = inflation

% #1. Some default parameters for our DGP and specification.
idxx = [5];
idxobs = [1,5,6];
lagVARSet = 1:10;            % candidate lag length for VAR.
horizon = 20;
NReplication = 2000;
NBootstrap = 2000;
TBurnIn = 100;
quantiles = [0.05, 0.95];
ylevely = 0;             % 0 = GDP, 1 = inflation
ylevelpi = 1;
trend = 0;               % Because we estimate models with Delta y, we have constants as trends.
ICIndicator = 1;         % HQIC


% Parameters for VAR
xLocation = 1;
differencedIndex = 2;
yLocation = [2,3];



% Parameters for BootstrapVAR methods
seIndicator = 0;         % default. see BootstrapVARR2 / SimulationAVarR2
cbIndicator = 0;         % default. see BootstrapVARR2 / SimulationAVarR2
boundIndicator = 1;      % default. see SimulationAVarR2

dfIndicator = 0;         % degrees of freedom adjustment when we estimate the asymptotic variance

% for gdp, y = gdp, X = [MP, pi, ffr]. Then VAR = [X(:,1), Delta y,
%   X(:,[2,3])]
XVAROrdery = [1,2,3];
yPlacey = 2;

% for pi, y = pi, X = [MP, y, ffr]. Then VAR = [X(:,[1,2]), y,
%   X(:,[3])]
XVAROrderpi = [1,2,3];
yPlacepi = 3;



% Parameters for SimulationAVar methods
XtIndicator = [1,0,0]; % MP is ordered first in the Cholesky structure.
jointIndicator = 0; % horizon by horizon approach when LP is used.

% To have a joint asymptotic distribution of ir's at horizons 0, 1, ... , horizon,
%  only intercept, x(t), x(t-1), and y(t-1) are included on the RHS
%  when estimating ir.
% We use the common lag specification for forecast errors.
%  See Appendix B for detatils.
lagyLP = 1;
lagXLP = [1,0,0];
XtIndicatorLP = [1,0,0];



% #2. Population ir and vd.
[ Population.iry, Population.vdFullInfoy, Population.vdMixedInfoy, Population.vdSimpleInfoy] ...
    = MultVarPopulation( horizon, 1, idxx, idxobs );
[ Population.irpi, Population.vdFullInfopi, Population.vdMixedInfopi, Population.vdSimpleInfopi] ...
    = MultVarPopulation( horizon, 5, idxx, idxobs );

% #3. Replication and applying our estimators
% #3.1. Initialization

lagVARResults      = zeros( NReplication, 1 );   % chosen lagVAR

iry       = zeros( NReplication, horizon + 1, 3 );  % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls), LP-(lagyLP, lagXLP only x)] ), ir
irpi      = zeros( NReplication, horizon + 1, 3 );  % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls), LP-(lagyLP, lagXLP only x)] ), ir

irBcy     = zeros( NReplication, horizon + 1, 2 );  % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls)] ), bias-corrected ir based on VAR-bootstrap
irBcpi    = zeros( NReplication, horizon + 1, 2 );  % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls)] ), bias-corrected ir based on VAR-bootstrap

irSey     = zeros( NReplication, horizon + 1, 4 );
irSepi    = zeros( NReplication, horizon + 1, 4 );
% ( idxn, h, [VAR-Bootstrap, LP-(VAR equivalent set-up with controls, VAR Bootstrap), LP-(VAR equivalent set-up with controls, Asymptotic variance),
%                            LP-(lagyLP, lagXLP only x, Asymptotic variance)] ), s.e.

vdy                     = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A-(VAR equivalent set-up with controls), LP B-(VAR equivalent set-up with controls)] ), vd w/o bias correction
vdpi                    = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A-(VAR equivalent set-up with controls), LP B-(VAR equivalent set-up with controls)] ), vd w/o bias correction

vdVARBcy                = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/ bias correction based on VAR-bootstrap
vdVARBcpi               = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/ bias correction based on VAR-bootstrap

vdVARSeBootstrapy       = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A, LP B] ), bootstrap s.e. 
vdVARSeBootstrappi      = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A, LP B] ), bootstrap s.e. 

vdVARCbBootstrapy       = zeros( NReplication, horizon + 1, 4, 2); % ( idxn, h, [VAR, R2, LP A, LP B], [5%, 95%] ), bootstrap confidence bands
vdVARCbBootstrappi      = zeros( NReplication, horizon + 1, 4, 2); % ( idxn, h, [VAR, R2, LP A, LP B], [5%, 95%] ), bootstrap confidence bands

vdAVarBcy               = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), vd w/ bias correction by simulating the asymptotic distribution (Appendix A and B)
vdAVarBcpi              = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), vd w/ bias correction by simulating the asymptotic distribution (Appendix A and B)

vdAVarSey               = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.
vdAVarSepi              = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.

vdAVarSeSimulationy     = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.
vdAVarSeSimulationpi    = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.


% It is a trick to define lagy and lagX inside parfor when they are
%   pre-specified, i.e. lagFlag = false.
if lagFlag  % lagy = lagX(1) = ... = lagX(N) = lagVAR, if not, use the pre-specified lags
else
    lagytemp = lagy;
    lagXtemp = lagX;
end


parfor idxn = 1:NReplication
    
    % #3.2. Data generation
    [~, x, endoAll] = MultVarDataGen( T, TBurnIn, 1, idxx );
    
    Xexo     = x;
    Xendowy  = endoAll(:,idxobs);
    XVAR  = [ Xexo, Xendowy ];   % shocks first in the order in idxx, then endogenous variables in the order in idxobs with y. Used for VARs
    XVAR(:,2) = [NaN; diff(XVAR(:,2))];
    timeseries = XVAR(2:end,:);  % [Mp, Delta GDP, inflation, ffr]
    
    % #3.3. VAR
    [ VARVARvd, VARVARvdBc, VARVARvdSeBootstrap, VARVARcb, VARVARdetails] ...
        = BootstrapVARVAR( timeseries, horizon, xLocation, differencedIndex, yLocation, trend, lagVARSet, ICIndicator, NBootstrap, TBurnIn, quantiles );
    
    lagVAR = VARVARdetails.lagVAR;
    
    
    
    % #3.4. GDP
    
    y = Xendowy(:,1);          % y
    X = [x, Xendowy(:,[2,3])]; % MP, pi, ffr
    
    % #3.4.1. Bias-corrected Estimators based on a VAR model
    if lagFlag  % lagy = lagX(1) = ... = lagX(N) = lagVAR, if not, use the pre-specified lags
        
        lagy = lagVAR;
        lagX = lagVAR * ones(size(X,2),1);
    else
        lagy = lagytemp;
        lagX = lagXtemp;
    end
    
    [ VARR2vd, VARR2vdBc, VARR2vdSe, VARR2cb, ~] ...
        = BootstrapVARR2( y, X, horizon, ylevely, trend, lagy, lagX, lagVAR, ...
        XVAROrdery, yPlacey, ICIndicator, NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );
    [ VARLPAvd, VARLPAvdBc, VARLPAvdSeBootstrap, VARLPAcb, ~] ...
        = BootstrapVARLPA( y, X, horizon, ylevely, trend, lagy, lagX, XtIndicator, lagVAR, ...
        XVAROrdery, yPlacey, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
    [ VARLPBvd, VARLPBvdBc, VARLPBvdSeBootstrap, VARLPBcb, VARLPBdetails] ...
        = BootstrapVARLPB( y, X, horizon, ylevely, trend, lagy, lagX, XtIndicator, lagVAR, ...
        XVAROrdery, yPlacey, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
    
    % #3.4.2 Bias-corrected Estimators based on the simulation of asymptotic
    % distributions
    
    [ ~, SimR2vdBc, ~, AVarR2cb, SimR2details] ...
        = SimulationAVarR2( y, X, horizon, ylevely, trend, lagy, lagX, ...
        NBootstrap, quantiles, seIndicator, cbIndicator, boundIndicator );
    [ ~, SimLPAvdBc, ~, AVarLPAcb, SimLPAdetails] ...
        = SimulationAVarLPA( y, X, horizon, ylevely, trend, lagy, lagX, XtIndicator, ...
        NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
        jointIndicator, X, lagyLP, lagXLP, XtIndicatorLP );
    [ ~, SimLPBvdBc, ~, AVarLPBcb, SimLPBdetails] ...
        = SimulationAVarLPB( y, X, horizon, ylevely, trend, lagy, lagX, XtIndicator, ...
        NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
        jointIndicator, X, lagyLP, lagXLP, XtIndicatorLP );
    
    
    % #3.4.3. Reshaping
    
    lagVARResults(idxn) = lagVAR;
    
    % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls), LP-(lagyLP, lagXLP only x)] ), ir
    iry(idxn,:,:)                 = [ VARVARdetails.ir(:,1),  VARLPBdetails.stdx * VARLPBdetails.ir,    SimLPBdetails.stdx * SimLPBdetails.ir   ];
    
    % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls)] ), VAR bias corrected ir
    irBcy(idxn,:,:)               = [ VARVARdetails.irBc(:,1), VARLPBdetails.stdx * VARLPBdetails.irBc  ];
    
    % ( idxn, h, [VAR-Bootstrap, LP-(VAR equivalent set-up with controls, VAR Bootstrap), LP-(VAR equivalent set-up with controls, Asymptotic variance),
    %                            LP-(lagyLP, lagXLP only x, Asymptotic variance)] ), s.e.
    irSey(idxn,:,:)               = [ VARVARdetails.irSeBootstrap(:,1), VARLPBdetails.stdx * VARLPBdetails.irSeBootstrap, ...
        VARLPBdetails.stdx * VARLPBdetails.irSeAVar, SimLPBdetails.stdx * SimLPBdetails.irSeAVar ];
    
    
    % ( idxn, h, [VAR, R2, LP A-(VAR equivalent set-up with controls), LP B-(VAR equivalent set-up with controls)] ), vd w/o bias correction
    vdy(idxn,:,:)                  = [ VARVARvd(:,1), VARR2vd, VARLPAvd, VARLPBvd ];
    
    % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/ bias correction based on VAR-bootstrap
    vdVARBcy(idxn,:,:)             = [ VARVARvdBc(:,1), VARR2vdBc, VARLPAvdBc, VARLPBvdBc ];
    
    % ( idxn, h, [VAR, R2, LP A, LP B] ), s.e. VAR-bootstrap
    vdVARSeBootstrapy(idxn,:,:)    = [ VARVARvdSeBootstrap(:,1), VARR2vdSe, VARLPAvdSeBootstrap, VARLPBvdSeBootstrap ];
    
    % ( idxn, h, [R2, LP A, LP B] ), vd w/ bias correction by simulating the asymptotic distribution
    vdAVarBcy(idxn,:,:)            = [ SimR2vdBc, SimLPAvdBc, SimLPBvdBc ];
    
    % ( idxn, h, [R2, LP A, LP B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.
    vdAVarSey(idxn,:,:)            = [ SimR2details.vdSeAVarWPW, SimLPAdetails.vdSeAVarWPW, SimLPBdetails.vdSeAVarWPW  ];
    
    % ( idxn, h, [R2, LP A, LP B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.
    vdAVarSeSimulationy(idxn,:,:)  = [ SimR2details.vdSeBootstrap, SimLPAdetails.vdSeBootstrap, SimLPBdetails.vdSeBootstrap ];
    
    
    vdVARCbBootstrapyTemp = zeros(horizon + 1, 4, 2);
    vdVARCbBootstrapyTemp(:,1,:) = VARVARcb(:,:,1)'; 
    vdVARCbBootstrapyTemp(:,2,:) = VARR2cb' ;
    vdVARCbBootstrapyTemp(:,3,:) = VARLPAcb' ;
    vdVARCbBootstrapyTemp(:,4,:) = VARLPBcb' ;
    vdVARCbBootstrapy(idxn,:,:,:) = vdVARCbBootstrapyTemp;    
    
    
    
   
    % #3.5. Inflation
    
    y = Xendowy(:,2);          % pi
    X = [x, Xendowy(:,[1,3])]; % MP, y, ffr
    
    % #3.5.1. Bias-corrected Estimators based on a VAR model
    
    [ VARR2vd, VARR2vdBc, VARR2vdSe, VARR2cb, ~] ...
        = BootstrapVARR2( y, X, horizon, ylevelpi, trend, lagy, lagX, lagVAR, ...
        XVAROrderpi, yPlacepi, ICIndicator, NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );
    [ VARLPAvd, VARLPAvdBc, VARLPAvdSeBootstrap, VARLPAcb, ~] ...
        = BootstrapVARLPA( y, X, horizon, ylevelpi, trend, lagy, lagX, XtIndicator, lagVAR, ...
        XVAROrderpi, yPlacepi, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
    [ VARLPBvd, VARLPBvdBc, VARLPBvdSeBootstrap, VARLPBcb, VARLPBdetails] ...
        = BootstrapVARLPB( y, X, horizon, ylevelpi, trend, lagy, lagX, XtIndicator, lagVAR, ...
        XVAROrderpi, yPlacepi, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
    
    % #3.5.2. Bias-corrected Estimators based on the simulation of asymptotic
    % distributions
    
    [ ~, SimR2vdBc, ~, AVarR2cb, SimR2details] ...
        = SimulationAVarR2( y, X, horizon, ylevelpi, trend, lagy, lagX, ...
        NBootstrap, quantiles, seIndicator, cbIndicator, boundIndicator );
    [ ~, SimLPAvdBc, ~, AVarLPAcb, SimLPAdetails] ...
        = SimulationAVarLPA( y, X, horizon, ylevelpi, trend, lagy, lagX, XtIndicator, ...
        NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
        jointIndicator, X, lagyLP, lagXLP, XtIndicatorLP );
    [ ~, SimLPBvdBc, ~, AVarLPBcb, SimLPBdetails] ...
        = SimulationAVarLPB( y, X, horizon, ylevelpi, trend, lagy, lagX, XtIndicator, ...
        NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
        jointIndicator, X, lagyLP, lagXLP, XtIndicatorLP );
    
    
    % #3.4.3. Reshaping
       
    % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls), LP-(lagyLP, lagXLP only x)] ), ir
    irpi(idxn,:,:)                 = [ VARVARdetails.ir(:,2),  VARLPBdetails.stdx * VARLPBdetails.ir,    SimLPBdetails.stdx * SimLPBdetails.ir   ];
    
    % ( idxn, h, [VAR, LP-(VAR equivalent set-up with controls)] ), VAR bias corrected ir
    irBcpi(idxn,:,:)               = [ VARVARdetails.irBc(:,2), VARLPBdetails.stdx * VARLPBdetails.irBc  ];
    
    % ( idxn, h, [VAR-Bootstrap, LP-(VAR equivalent set-up with controls, VAR Bootstrap), LP-(VAR equivalent set-up with controls, Asymptotic variance),
    %                            LP-(lagyLP, lagXLP only x, Asymptotic variance)] ), s.e.
    irSepi(idxn,:,:)               = [ VARVARdetails.irSeBootstrap(:,2), VARLPBdetails.stdx * VARLPBdetails.irSeBootstrap, ...
        VARLPBdetails.stdx * VARLPBdetails.irSeAVar, SimLPBdetails.stdx * SimLPBdetails.irSeAVar ];
    
    
    % ( idxn, h, [VAR, R2, LP A-(VAR equivalent set-up with controls), LP B-(VAR equivalent set-up with controls)] ), vd w/o bias correction
    vdpi(idxn,:,:)                  = [ VARVARvd(:,2), VARR2vd, VARLPAvd, VARLPBvd ];
    
    % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/ bias correction based on VAR-bootstrap
    vdVARBcpi(idxn,:,:)             = [ VARVARvdBc(:,2), VARR2vdBc, VARLPAvdBc, VARLPBvdBc ];
    
    % ( idxn, h, [VAR, R2, LP A, LP B] ), s.e. VAR-bootstrap
    vdVARSeBootstrappi(idxn,:,:)    = [ VARVARvdSeBootstrap(:,2), VARR2vdSe, VARLPAvdSeBootstrap, VARLPBvdSeBootstrap ];
    
    % ( idxn, h, [R2, LP A, LP B] ), vd w/ bias correction by simulating the asymptotic distribution
    vdAVarBcpi(idxn,:,:)            = [ SimR2vdBc, SimLPAvdBc, SimLPBvdBc ];
    
    % ( idxn, h, [R2, LP A, LP B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.
    vdAVarSepi(idxn,:,:)            = [ SimR2details.vdSeAVarWPW, SimLPAdetails.vdSeAVarWPW, SimLPBdetails.vdSeAVarWPW  ];
    
    % ( idxn, h, [R2, LP A, LP B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.
    vdAVarSeSimulationpi(idxn,:,:)  = [ SimR2details.vdSeBootstrap, SimLPAdetails.vdSeBootstrap, SimLPBdetails.vdSeBootstrap ];
    
    
    
    vdVARCbBootstrappiTemp = zeros(horizon + 1, 4, 2);
    vdVARCbBootstrappiTemp(:,1,:) = VARVARcb(:,:,2)';
    vdVARCbBootstrappiTemp(:,2,:) = VARR2cb';
    vdVARCbBootstrappiTemp(:,3,:) = VARLPAcb';
    vdVARCbBootstrappiTemp(:,4,:) = VARLPBcb';
    vdVARCbBootstrappi(idxn,:,:,:) = vdVARCbBootstrappiTemp;
    
    
end

