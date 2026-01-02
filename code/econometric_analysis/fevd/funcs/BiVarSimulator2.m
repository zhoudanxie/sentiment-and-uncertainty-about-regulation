function [ Population, ir, irBc, irSe, ...
    vd, vdVARBc, vdVARSeBootstrap, lagVAR, ...
    vdAVarBc, vdAVarSe, vdAVarSeSimulation, ...
    vdVARCbBootstrap, vdAVarCbBootstrap, vdBlockBc, vdBlockSeBootstrap, vdBlockCbBootstrap ] ...
    = BiVarSimulator2( dgpidx, horizon, NReplication, T, NBootstrap, TBurnIn, lagVARSet, lagVARBootstrap, lagy, lagx, sizeBlock )
%For the difference between BiVarSimulator and BiVarSimulator2, please
%look at the comment below on lagVARBootstrap as an input. 
% BiVarSimulator.m does not take lagVARBootstrap as an input. This code is
% used for DGP3 when lagVAR is fixed at either 5 or 10.
%
%Simulate the following process and apply the methods introduced in
%   Gorodnichenko and Lee (2019). 'Parfor' is used. So, using 
%   'parpool' command would increase the efficiency of this code.
%
%   DGP:
%   y(t)   = psix(L)x(t) + z(t),
%   z(t)   = p(t) + a(t),
%   ((1-L)p(t) -gy) = rhop * ((1-L)p(t) -gy) + sigmap * wnp(t),
%   a(t)   = rhoa * a(t-1) + sigmaa * wna(t),
%   x(t)   ~ wn(sigmax),
%   wnp(t) ~ wn(1),
%   wna(t) ~ wn(1).
%
% INPUT:
% dgpidx  = Chooses among three pre-specified psix(L).
%   if 1: psix(L) = hump-shaped, MA(100).
%   if 2: psix(L) = AR(1): (1 - 0.9L)^(-1)
%   if 3: psix(L) = Integrated AR(1): (1 - L)^(-1)*(1 - 0.9L)^(-1)
% horizon = ir and vd are derived up to the horizon-th
% NReplication = replication size of the simulation study.
% T       = Generate length T vectors
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%	NBootstrap times to derive the bias and the standard errors.
% TBurnIn = First TBurnIn observations are dropped as burn-in
% lagVARSet   = Set of candidate lag lengthes. The lag length is determined
%   by information criteria. It should be either a column
%	or a row vector.
% lagVARBootstrap = Used for DGP3 / Table 4. For example, if it is 5, 
%   then VAR(5) is estimated and used to simulate artificial data. But 
%   estimation of ir / vd are based on lagVARSet, lagy, and lagx. If not 
%   specified, lagVARBootstrap = lagVARSet.
% lagy    = y(-1), ... , y(-lagy) will be included on the RHS to
%   derive the impulse response coefficients using the local projection
%   method. If ylevel == 0, then Delta y is used instead.
% lagx    = x(0), x(-1), ... , x(-lagx) will be included on the RHS to
%  estimate the impulse response coefficients.
% 
% NOTE: If lagy and lagx are not specified, they are the same as lagVAR
%   selected via an information criterion.
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
% lagVAR   = Selected lag length for the VAR model by the IC.

% #0.
if nargin == 7
    lagVARBootstrap = lagVARSet;
    lagFlag = true;  % If lagFlag == 1, lagy = lagx = lagVAR.
    lagy = [];
    lagx = [];
elseif nargin == 8
    lagFlag = true;  % If lagFlag == 1, lagy = lagx = lagVAR.
    lagy = [];
    lagx = [];
else
    lagFlag = false;
end


% #1. Some default parameters for our DGP and specification.
ylevel = 0;              % we include Delta y on the RHS.
trend = 0;               % Because we estimate models with Delta y, we have constants as trends.
ICIndicator = 1;         % Hannan-Quinn information criterion
dfIndicator = 0;         % degrees of freedom adjustment when we estimate the asymptotic variance
lagyLP = 1;              % For the AVarLP method (which requires the joint distribution of LP ir estimators), the choice that lagyLP = 1, lagXLP = 1 requires 4 * (horizon) parameters including constants to be estimated which should be less than T.
lagXLP = 1;
quantiles = [0.05, 0.95];
seIndicator = 0;        % bootstrap
cbIndicator = 0;        % bootstrap
boundIndicator = 1;     % for SimulationAVarR2. 
jointIndicator = 0;     % For SimulationAVarLP. We do it horizon by horizon.
        
% Parameters required when bootstrapping the estimated VAR to determine the
%   ordering of variables. See explanation in BootstrapVARR2.m.
XtIndicator = 1;        
XVAROrder = 1;
yPlace = 2;
XtIndicatorLP = 1;

% Block Bootstrap
if nargin < 11
    sizeBlock = 4;
end

% #2. Population ir and vd.
[ Population.ir, Population.vd, Population.sigmax ] ...
    = BiVarPopulation( dgpidx, horizon );

% #3. Replication and applying our estimators
% #3.1. Initialization
ir     = zeros( NReplication, horizon + 1, 3 );  % ( idxn, h, [VAR, LP-(lagy, lagx), LP-(lagyLP, lagXLP)] ), ir
irBc   = zeros( NReplication, horizon + 1, 2 );  % ( idxn, h, [VAR, LP-(lagy, lagx)] ), bias-corrected ir based on VAR
irSe   = zeros( NReplication, horizon + 1, 4 );  
% ( idxn, h, [VAR-Bootstrap, LP-(lagy, lagx, VAR Bootstrap), LP-(lagy, lagx, Asymptotic variance),
%                            LP-(lagyLP, lagXLP, Asymptotic variance)] ), s.e.

vd                   = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/o bias correction
vdVARBc              = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/ bias correction based on VAR-bootstrap
vdVARSeBootstrap     = zeros( NReplication, horizon + 1, 4); % ( idxn, h, [VAR, R2, LP A, LP B] ), bootstrap s.e.
vdVARCbBootstrap     = zeros( NReplication, horizon + 1, 4, 2); % ( idxn, h, [VAR, R2, LP A, LP B], [5%, 95%] ), bootstrap confidence band

lagVAR = zeros( NReplication, 1 ); % Selected lag length among lagVARSet

vdAVarBc             = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), vd w/ bias correction by simulating the asymptotic distribution (Appendix A and B)
vdAVarSe             = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.
vdAVarSeSimulation   = zeros( NReplication, horizon + 1, 3); % ( idxn, h, [R2, LP A, LP B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.
vdAVarCbBootstrap    = zeros( NReplication, horizon + 1, 3, 2); % ( idxn, h, [R2, LP A, LP B], [5%, 95%] ), s.e. by simulating the asymptotic distribution

vdBlockBc            = zeros( NReplication, horizon + 1);
vdBlockSeBootstrap   = zeros( NReplication, horizon + 1);
vdBlockCbBootstrap   = zeros( NReplication, horizon + 1, 2);

parfor idxn = 1:NReplication
    
    % #3.2. Data generation
    [ y, x ] = BiVarDataGen( T, TBurnIn, dgpidx );
    
    % #3.3. Bias-corrected Estimators based on a VAR model
    [ VARVARvd, VARVARvdBc, VARVARvdSeBootstrap, VARVARcb, VARVARdetailsir] ...
        = BootstrapVARVAR( [x(2:end), diff(y)], horizon, 1, 2, 2, trend, lagVARBootstrap, ICIndicator, NBootstrap, TBurnIn, quantiles );
    
    
    if lagFlag   % lagx = lagy = lagVAR
        
        [ ~, ~, ~, ~, VARVARdetails] ...
            = BootstrapVARVAR( [x(2:end), diff(y)], horizon, 1, 2, 2, trend, lagVARSet, ICIndicator, NBootstrap, TBurnIn, quantiles );
        
        lagyVAR = VARVARdetails.lagVAR;
        lagxVAR = VARVARdetails.lagVAR;
        
        [ VARR2vd, VARR2vdBc, VARR2vdSe, VARR2cb, VARR2details] ...
            = BootstrapVARR2( y, x, horizon, ylevel, trend, lagyVAR, lagxVAR, VARVARdetails.lagVAR, ...
            XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );
        [ VARLPAvd, VARLPAvdBc, VARLPAvdSeBootstrap, VARLPAcb, ~] ...
            = BootstrapVARLPA( y, x, horizon, ylevel, trend, lagyVAR, lagxVAR, XtIndicator, VARVARdetails.lagVAR, ...
            XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
        [ VARLPBvd, VARLPBvdBc, VARLPBvdSeBootstrap, VARLPBcb, VARLPBdetails] ...
            = BootstrapVARLPB( y, x, horizon, ylevel, trend, lagyVAR, lagxVAR, XtIndicator, VARVARdetails.lagVAR, ...
            XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
        
        
        % #3.4. Bias-corrected Estimators based on the simulation of asymptotic
        % distributions
        [ ~, AVarR2vdBc, AVarR2vdSe, AVarR2cb, AVarR2details] ...
            = SimulationAVarR2( y, x, horizon, ylevel, trend, lagyVAR, lagxVAR, ...
            NBootstrap, quantiles, seIndicator, cbIndicator, boundIndicator );
        [ ~, AVarLPAvdBc, AVarLPAvdSe, AVarLPAcb, AVarLPAdetails] ...
            = SimulationAVarLPA( y, x, horizon, ylevel, trend, lagyVAR, lagxVAR, XtIndicator, ...
            NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
            jointIndicator, x, lagyLP, lagXLP, XtIndicatorLP );
        [ ~, AVarLPBvdBc, AVarLPBvdSe, AVarLPBcb, AVarLPBdetails] ...
            = SimulationAVarLPB( y, x, horizon, ylevel, trend, lagyVAR, lagxVAR, XtIndicator, ...
            NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
            jointIndicator, x, lagyLP, lagXLP, XtIndicatorLP );
        
        % #3.5. Block bootstrap
        [ ~, BlockR2vdBc, BlockR2vdSe, BlockR2cb, ~] ...
            = BootstrapBlockR2( y, x, horizon, ylevel, trend, lagyVAR, lagxVAR, sizeBlock, ...
            NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );
        
    else    % lagy and lagX are pre-specified.
                
        [ VARR2vd, VARR2vdBc, VARR2vdSe, VARR2cb, VARR2details] ...
            = BootstrapVARR2( y, x, horizon, ylevel, trend, lagy, lagx, lagVARSet, ...
            XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );
        [ VARLPAvd, VARLPAvdBc, VARLPAvdSeBootstrap, VARLPAcb, ~] ...
            = BootstrapVARLPA( y, x, horizon, ylevel, trend, lagy, lagx, XtIndicator, lagVARSet, ...
            XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
        [ VARLPBvd, VARLPBvdBc, VARLPBvdSeBootstrap, VARLPBcb, VARLPBdetails] ...
            = BootstrapVARLPB( y, x, horizon, ylevel, trend, lagy, lagx, XtIndicator, lagVARSet, ...
            XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator );
        
        % Bias-corrected Estimators based on the simulation of asymptotic
        % distributions
        [ ~, AVarR2vdBc, AVarR2vdSe, AVarR2cb, AVarR2details] ...
            = SimulationAVarR2( y, x, horizon, ylevel, trend, lagy, lagx, ...
            NBootstrap, quantiles, seIndicator, cbIndicator, boundIndicator );
        [ ~, AVarLPAvdBc, AVarLPAvdSe, AVarLPAcb, AVarLPAdetails] ...
            = SimulationAVarLPA( y, x, horizon, ylevel, trend, lagy, lagx, XtIndicator, ...
            NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
            jointIndicator, x, lagyLP, lagXLP, XtIndicatorLP );
        [ ~, AVarLPBvdBc, AVarLPBvdSe, AVarLPBcb, AVarLPBdetails] ...
            = SimulationAVarLPB( y, x, horizon, ylevel, trend, lagy, lagx, XtIndicator, ...
            NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
            jointIndicator, x, lagyLP, lagXLP, XtIndicatorLP );
        
        % Block bootstrap
        [ ~, BlockR2vdBc, BlockR2vdSe, BlockR2cb, ~] ...
            = BootstrapBlockR2( y, x, horizon, ylevel, trend, lagy, lagx, sizeBlock, ...
            NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator );
        
    end
    
    
    
    
    % #3.5. Reshaping
    
    ir(idxn,:,:)               = [ VARVARdetailsir.ir, VARLPBdetails.stdx * VARLPBdetails.ir, AVarLPBdetails.stdx * AVarLPBdetails.ir ];
    % ( idxn, h, [VAR, LP-(lagy, lagx), LP-(lagyLP, lagXLP)] ), ir
    
    irBc(idxn,:,:)             = [ VARVARdetailsir.irBc, VARLPBdetails.stdx * VARLPBdetails.irBc ];
    % ( idxn, h, [VAR, LP-(lagy, lagx)] ), VAR bias corrected ir
    
    irSe(idxn,:,:)             = [ VARVARdetailsir.irSeBootstrap, VARLPBdetails.stdx * VARLPBdetails.irSeBootstrap, VARLPBdetails.stdx * VARLPBdetails.irSeAVar,...
        AVarLPBdetails.stdx * AVarLPBdetails.irSeAVar ];
    % ( idxn, h, [VAR-Bootstrap, LP-(lagy, lagx, VAR Bootstrap), LP-(lagy, lagx, Asymptotic variance),
    %                            LP-(lagyLP, lagXLP, Asymptotic variance)] ), s.e.
    
    
    
    vd(idxn,:,:)               = [ VARVARvd, VARR2vd, VARLPAvd, VARLPBvd ];
    % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/o bias correction
    
    vdVARBc(idxn,:,:)          = [ VARVARvdBc, VARR2vdBc, VARLPAvdBc, VARLPBvdBc ];
    % ( idxn, h, [VAR, R2, LP A, LP B] ), vd w/ bias correction based on VAR-bootstrap
    
    vdVARSeBootstrap(idxn,:,:) = [ VARVARvdSeBootstrap, VARR2vdSe, VARLPAvdSeBootstrap, VARLPBvdSeBootstrap ];
    % ( idxn, h, [VAR, R2, LP A, LP B] ), s.e. VAR-bootstrap
    
    
    
    lagVAR(idxn) = VARR2details.lagVAR;
    % Selected lag length among lagVARSet

    
    
    vdAVarBc(idxn,:,:)         = [ AVarR2vdBc, AVarLPAvdBc, AVarLPBvdBc ]; 
% ( idxn, h, [R2, LP A, LP B] ), vd w/ bias correction by simulating the asymptotic distribution
    
    vdAVarSe(idxn,:,:)         = [ AVarR2details.vdSeAVarWPW, AVarLPAdetails.vdSeAVarWPW, AVarLPBdetails.vdSeAVarWPW ]; 
% ( idxn, h, [R2, LP A, LP B] ), s.e. from the asymptotic distribution. w/ pre-whitening. For LP, it is horizon by horizon, not joint inference.
    
    vdAVarSeSimulation(idxn,:,:)  = [ AVarR2vdSe, AVarLPAvdSe, AVarLPBvdSe ]; 
% ( idxn, h, [R2, LP A, LP B] ), s.e. from the simulated \xi(\theta^b)'s. See Appendix for details.

    



    vdBlockBc(idxn,:)          = BlockR2vdBc;
    
    vdBlockSeBootstrap(idxn,:) = BlockR2vdSe;
    
    vdBlockCbBootstrap(idxn,:,:) = BlockR2cb';
    

    vdVARCbBootstrapTemp = zeros(horizon + 1, 4, 2);
    vdVARCbBootstrapTemp(:,1,:) = VARVARcb' 
    vdVARCbBootstrapTemp(:,2,:) = VARR2cb' 
    vdVARCbBootstrapTemp(:,3,:) = VARLPAcb' 
    vdVARCbBootstrapTemp(:,4,:) = VARLPBcb' 
    vdVARCbBootstrap(idxn,:,:,:) = vdVARCbBootstrapTemp;
    

    vdAVarCbBootstrapTemp = zeros(horizon + 1, 3, 2);
    vdAVarCbBootstrapTemp(:,1,:) = AVarR2cb'
    vdAVarCbBootstrapTemp(:,2,:) = AVarLPAcb'
    vdAVarCbBootstrapTemp(:,3,:) = AVarLPBcb'
    vdAVarCbBootstrap(idxn,:,:,:) = vdAVarCbBootstrapTemp;

end


end