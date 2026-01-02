function [y, x, endoAll, exoAll] = MultVarDataGen( T, TBurnIn, idxy, idxx )
%Simulate a medium-sized DSGE model in Smets and Wouters (2007).
% The estimated parameter values in Smets and Wouters (2007) are used. 
%
% INPUT
% T       = Generate length T sample
% TBurnIn = First TBurnIn observations are dropped as burn-in
% idxy    = Out of seven endogenous variables, the selected one is
%   returned as y.
%   1: y=GDP, 2: c=Consumption, 3: inve=Investment, 4: w=Wage, 
%   5: pinf=Price inflation, 6: r=Nominal interest rates, 7: lab=Employment
% idxx    = Among seven exogenous shocks, the selected one is returned as
%   x.
%   1: a=TFP, 2: b=Credit spread, 3: g=Government expenditure, 
%   4: qs=Investment specific technology, 5: m=Monetary policy, 
%   6: spinf=price mark up, 7: sw=wage mark up
%
% OUTPUT
% y = T * 1 vector of the simulated endogenous variable
% x = T * 1 vector of the exogenous shock
% endoAll = (T * 7) All seven variables.  
% exoAll  = (T * 7) All seven exogenous variables.


warning('off', 'all')

% #1. Data generating process: Smets and Wouters (2007)
load sw_model_results       % Solved results using dynare given the parameter estimates reported in Smets and Wouters (2007). 

% Dynare saves the solution as follows:
% Y: log-linearized variables.
% State equation: Y     = A * state(-1) + B * ETA,
% Y     = M_.endo_names(oo_.dr.order_var,:),  33 * 1
% A     = oo_.dr.ghx
% state = M_.endo_names(oo_.dr.state_var,:),  20 * 1
% state = Y(8:27),
% B     = oo_.dr.ghu
% ETA   = (eta_a, eta_b, eta_g, eta_qs, eta_m, eta_pinf, eta_w)',  7 * 1
% ETA   ~  iid N( 0, Omega )
% Omega = M_.Sigma_e. It is a diagonal matrix.
% Measurement equation:
% YObservable = ( Delta y, Delta c, Delta inve, Delta w, pinf, r, lab)' 
%             = [diff(Y([11, 24, 25, 27]), Y([26, 12, 33])]' 
%               + Constant trend.


trend = [0.43; 0.43; 0.43; 0.43; -0.1; 0.81; M_.params(34)];
A = oo_.dr.ghx;
B = oo_.dr.ghu;
Omega = M_.Sigma_e;
OmegaRoot = Omega^(1/2);


% #2. Data generation
wn = normrnd(0,1,T + TBurnIn, 7);    % 7: seven shocks.
Y  = zeros(T + TBurnIn, 33);       

for idxt = 2:T + TBurnIn   % Note that the simulation begins at the steady-state. That is, the initial conditions are zeros.
    Y(idxt, :) = A * Y(idxt-1, 8:27)' + B * OmegaRoot * wn(idxt,:)';
end

YEndo = Y(TBurnIn + 1 : T + TBurnIn, [11,24,25,27,26,12,33]); 
% 1) Dropping the burn in periods
% 2) Saving only observable endogenous variables:
%    (y, c, inve, w, pinf, r, lab)
%    Note that they are in log-deviations from the trend.



endoAll = YEndo + [(1:1:T)' * trend(1:4)', ones(T,1) * trend(5:7)'];
% Adding the trend growth component.
% The first four variables are growing at the constant rate. The last three
% variables are inflation, nominal interest rates, and employment that are
% already stationary around constants.

exoAll = wn(TBurnIn + 1:T + TBurnIn, :) .* repmat(diag(OmegaRoot)', T, 1); 
% All seven shocks adjusted by the standard deviation. Thus, it leads to
% the actual magnitudes of the shocks.

y = endoAll(:, idxy); % selected variable among seven endogenous ones
x = exoAll(:, idxx);  % selected variable among seven exogenous shocks

