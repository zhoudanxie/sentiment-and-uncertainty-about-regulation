function [ vd, vdBc, vdSeBootstrap, cb, details] ...
    = BootstrapVARLPA( y, X, horizon, ylevel, trend, lagy, lagX, XtIndicator, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, dfIndicator )
%Estimate the FEVD using the LP A method with VAR-based bootstrap.
%
% Difference between LP A and LP B::
% A uses the denominator of Var(FE(t+h,t-1)) directly instead of
%   Sum ir_x(h)^2 * sigma_x^2 + Var(v(t+h,t-1)).
%   B method uses the latter.
%
% By bootstrapping the estimated VAR model, we derive the bias-corrected
% FEVD, vdBc, and the corresponding standard errors, vdSe.
% INPUT:
% y(t)    = (T * 1) dependent variable.
% X(t, n) = (T * N) shock and controls. The first column is the shock. The
%   shock value will be de-meaned.
% horizon = ir and vd are derived up to the horizon-th period
% ylevel  = Delta y on the RHS when deriving the forecast errors and 
%           estimating a VAR if ylevel == 0, y on the RHS if ylevel == 1.
% trend   = c                     if trend == 0,
%           c + dt                if trend == 1,
%           c + dt + et^2         if trend == 2,
%           c + dt + et^2 + ft^3  if trend == 3.
% lagy    = y(-1), ... , y(-lagy) will be included on the RHS to
%   build the forecast errors. If ylevel == 0, then Delta y is used instead.
% lagX    = (1 * N) or (N * 1) vector. lagX(n) number of lagged values of
%   n-th variable in X is included in the forecasting error regression. We
%   have
%   X_1(t), ...
%   X_1(t-1), ... , X_1(t-lagX(1)), ...
%   X_2(t-1), ... , X_2(t-lagX(2)), ...
%   ...
%   X_N(t-1), ... , X_N(t-lagX(N)) on the RHS.
% XtIndicator = (1 * N) or (N * 1) vector of zero or one. It selects which
%   time-t values of X variables are included. For example, if
%   XtIndicator = [1,1,0,1,0]', then we add
%   X_2(t) and X_4(t) on the RHS in addition to X_1(t) which is our shock
%   of interest. Given the idea of the local projections, it is clear that
%   the first element of XtIndicator should be 1.
%   In a bivariate case, it can be just 1.
%
% NOTE: When there are many control variables or specific ordering
%   assumption is required for the purpose of identification, it is easier
%   to work with lagy = 0 and y or Delta y included in X directly. Then
%   specification details can be implemented by choosing lagX and
%   XtIndicator.
%
% lagVARSet   = Set of candidate lag lengthes. The lag length is determined
%   in terms of the information criterion. It should be either a column
%	or row vector.
% XVAROrder   = A VAR model with a Cholesky ordering is used for bootstrap.
%   The timeseries used is XVAR = X(:,XVAROrder). We will also add y or 
%   Delta y based on yPlace.
%   (Default = 1:N).
% yPlace = It is about where to add the dependent variable in the VAR
%   model. We put y or Delta y depending on ylevel in the yPlace-th place. 
%   For example, if yPlace == 2, 
%   XVAR = [XVAR(:,1), y, XVAR(:,2:end)] if ylevel == 1,
%   XVAR = [XVAR(2:end,1), diff(y), XVAR(2:end,2:end)] if ylevel == 0.
%   (Default = N+1).
%
% NOTE: XVAROrder and yPlace are here to have a flexible ordering of
%   variables when estimating a VAR. For example, when we have y=GDP, 
%   pi=inflation, r=FFR, and MP=monetary policy shock, 
%   X would be Xwoy = [MP, pi, r] or Xwy = [MP, y or Delta y, pi, r]. 
%   If we use Xwoy, lagy might be positive. If we use Xwy, information in
%   y is directly controlled by y or Delta y in Xwy, so lagy should be 
%   zero to avoid multi-collinearity.
%   For the VAR, one might want to impose contemporaneous restrictions. 
%   In such a case, XVAR might be [y or Delta y, pi, MP, r]. To impose such 
%   Cholesky order and bootstrap the model with this ordering, one may let 
%   XVAROrder=[2,1,3], and yPlace = 1 if Xwoy is used. If Xwy is used
%   instead, then XVAROrder=[3,1,4], and yPlace = 1. Note that we add y
%   using yPlace, not directly by including y in XVAROrder when X contains
%   y or Delta y. This is to identify which variable in the VAR system is
%   the dependent variable.
%       BUT different orderings would have no effects on bootstrapped 
%	samples because we simulate the reduced form residuals.
%
% ICIndicator = Indicator for which information criterion to use.
%	0: BIC, 1: HQIC, 2: AIC.     (Default = 1).
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%	NBootstrap times to derive the bias, standard errors,
%	and confidence bands.        (Default = 2000).
% TBurnIn     = First TBurnIn observations are dropped as burn-in in 
%   bootstrap samples.
% qunatiles   = save the corresponding quantiles in cb. For example, if
%	quantiles = [0.05, 0.95], then cb will be a matrix whose size is
%	(2 * horizon+1) and is representing 90% confidence bands based on 
%   bootstrap. (Default = []).
% dfIndicator = Whether to adjust the degrees of freedom when building the
%   forecast error variance.
%   E[FE(t+h,t-1)^2] is estimated by
%       = \sum FE(t+h, t-1)^2 / denominator.
%       The denominator is
%       T_h - lagy - sum(lagX) - (trend + 1) if dfIndicator == 1, and
%       T_h o.w.
%   where T_h = T - t_0 - h + 1 = effective number of observations
%       for horizon h.
%
% OUTPUT:
% vd        = (1+horizon * 1) forecasting error variance decompositions
%   of y(t+h) - y(t-1) in relations to
%   x(t), x(t+1), x(t+2), ..., x(t+h) using LP.
% vdBc     = bias-corrected FEVD.
%	LP A estimator with VAR based bootstrap and bias correction.
% vdSeBootstrap = Bootstrap based standard errors for vd and vdBc.
% cb       = condifence bands. See above explanations for 'quantiles.' It
%   is based on bootstrap.
% If you want to have asymptotic variance and corresponding standard error,
%   you may use SimulationAVarLPA.
% details  = A structure consists of other results.
% details.ir / irBc / irSeBootstrap / irSeAVar :
%	Local Projection based impulse responses, bias-corrected estimates,
%	bootstrap based s.e., and asymptotic variance based s.e..
%       Input: a unit shock to x.
%       When one wants to have responses to a one std shock, 
%       details.stdx can be multiplied.
% details.stdx = std(x);  For conversions from a unit shock to a one s.d. shock
% details.lagVAR = Chosen lag length for the VAR model according to the IC.
% details.stdx = std(x).

% #1. Setting up the default parameters
if nargin < 9
    error('Not enough input arguments')
end

[Ty,N] = size(X);   % size of the time series

if nargin < 16
    dfIndicator = 0;
end
if nargin < 15
    quantiles = [];
end
if nargin < 14
    TBurnIn = 100;
end
if nargin < 13
    NBootstrap = 2000;
end
if nargin < 12
    ICIndicator = 1;
end
if nargin < 11
    yPlace = N+1;
end
if nargin < 10
    XVAROrder = 1:N;
end



% #2. Information criterion and lagVAR
% #2.1. Time series used to estimate a VAR
XVAR = X(:,XVAROrder);
if ylevel == 1
    XVAR = [XVAR(:,1:yPlace-1), y, XVAR(:,yPlace:end)];
else
    XVAR = [XVAR(2:end,1:yPlace-1), diff(y), XVAR(2:end,yPlace:end)];
end

NXVAR = size(XVAR,2); % Number of variables in the VAR system

if isscalar(lagVARSet)
    % If there's only one candidate, then that is lagVAR.
    lagVAR = lagVARSet;
    
else
    % When there are multiple candidates
    IC = zeros(size(lagVARSet));
    
    lagVARMax = max(lagVARSet);   % the largest candidate.
    T = Ty - lagVARMax - (1-ylevel); % if ylevel == 0, we lose one observation when taking the first differences in y.
    % For fair comparisons, we impose the same effective sample sizes for different lagVAR.

    for idxIC = 1: length(IC)
        lagVARtemp = lagVARSet(idxIC); % a candidate lag length
        [~, ~, omega, ~, k] = var_estimation(XVAR(lagVARMax - lagVARtemp + 1:end, :), lagVARtemp, 0, trend);
        % 'lagVARMax - lagVARtemp + 1': To equalize the effective sample
        % size for comparing the values of evaluated information criterion.
        
        if ICIndicator == 0      % BIC. # of parameters = k.
            IC(idxIC) = log(det(omega)) + k * log(T) / T;
        elseif ICIndicator == 1  % HQIC
            IC(idxIC) = log(det(omega)) + 2 * k * log(log(T)) / T;
        else                     % AIC
            IC(idxIC) = log(det(omega)) + 2 * k / T;
        end
        
    end
    [~,idxIC] = min(IC);
    lagVAR = lagVARSet(idxIC);   % lagVAR minimizes the IC among the candidates.
end




% #3.0. Finding which variable in XVAR is the shock and the dependent
%       variable.
XVAROrder = XVAROrder(:);
XVAROrderwy = [XVAROrder(1:yPlace-1); 0; XVAROrder(yPlace:end)];

OnetoN = 1:N;
OnetoNXVAR = 1:NXVAR;

xLocation = OnetoNXVAR(XVAROrderwy == 1); 
% 1, because the first vector in X is the shock.
% xLocation-th variable in XVAR in the VAR model is the shock.
yLocation = yPlace;
% yLocation-th variable in XVAR in the VAR model is the dependent variable.

% Finding inverse ordering
InvXVAROrder = zeros(N,1);

if N == NXVAR         % y or Delta y is included in X from the beginning
    % Becasue we add y or Delta y in XVAR via yPlace and lagy instead of
    % directly adding it via XVAROrder,
    % yLocationInX-th variable in X is either y or Delta y.
    yLocationInX = setdiff(1:N, XVAROrder);
    
    % we apply this InvXVAROrder to make bootstrapped X including y or Delta y.
    XVAROrderwy(yPlace) = yLocationInX; % It is a permutation of 1:N.
    for n= 1:N
        InvXVAROrder(n) = OnetoN(XVAROrderwy == n);
    end
    
elseif N == NXVAR - 1 % y or Delta y is not included in X. It is controlled through lagy.
    % in this case, XVAROrder is a permutation of 1:N. After dropping y
    % from the simulated time series, we apply this InvXVAROrder to make
    % bootstrapped X.
    for n= 1:N
        InvXVAROrder(n) = OnetoN(XVAROrder == n);
    end
end



% #3.1. Estimation: VAR for the purpose of bootstrapping
% If y is differenced in VAR, the vd is calculated for the cumulated
% series.
if ylevel == 0
    differencedIndex = yLocation;
else
    differencedIndex = [];
end
[~, ~, ~, vdVAR, ~, ~, ~, ...
    c, phi, ~, ~, ~, ~, residual] ...
    = var_chol(XVAR, lagVAR, horizon, NBootstrap, xLocation, 0, differencedIndex, trend, 0);
vdVAR = vdVAR(yLocation,:);
vdVAR = vdVAR(:);

% #3.2. Estimation: LP A method
[ details.ir, vd, ~, details.irSeAVar, ~, varx ] ...
    = SubLPA( y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator, 1, 0 );
% semIndicator = 0, i.e. ir is obtained in the horizon by horizon way.
% For VAR based bootstrap, we let XLP = X, lagyLP = lagy, etc. See SubLPA
%   for details.


% #4. Bootstrap
% initialization
irb = zeros(NBootstrap, horizon + 1);  % (b, h+1) = \psi_{x,h}^{(b)}
vdb = zeros(NBootstrap, horizon + 1);  % (b, h+1) = s_{h}^{(b)}


for idxb = 1:NBootstrap

    % #4.1. Data generation: Bootstrap using the estimated VAR
    [XVARb] = var_bootstrap(c, phi, XVAR, residual, Ty, TBurnIn);
    % Ty = length(y) or size(X,1)
    
    yb = XVARb(:,yLocation);
    % if Delta y(t) is simulated...
    if ylevel == 0
        yb = cumsum(yb);
    end
    
    % Making Xb
    if N == NXVAR         % y or Delta y is included in X from the beginning
        Xb = XVARb(:,InvXVAROrder);
        
    elseif N == NXVAR - 1 % y or Delta y is not included in X.
        XVARb = XVARb(:,setdiff(1:NXVAR,yLocation));  % Dropping y or Delta y
        Xb = XVARb(:,InvXVAROrder);
    end
    
    
    % #4.2. Estimate vd using simualated series
    [irb(idxb,:), vdb(idxb,:)] ...
        = SubLPA( yb, Xb, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator, 0, 0 );
    
end


% #5. Bias correction, standard errors, etc.
% #5.1. Impulse response
details.irBc = 2 * details.ir - mean(irb,1)';  % bias correction
details.irSeBootstrap = std(irb, 0, 1)'; % standard error
details.stdx = sqrt(varx);               % For conversions from a unit shock to a one s.d. shock


% #5.2. Variance decomposition
mvdb = mean(vdb,1)';   % average of vdb
vdBc = vd - ( mvdb - vdVAR ); % bias correction
vdSeBootstrap = std(vdb, 0, 1)'; % standard error 

% # 5.3. lagVAR
details.lagVAR = lagVAR;


% # 5.4. confidence bands
if isempty(quantiles)
    cb = [];
else
    qvd = quantile(vdb,quantiles,1);  % (quantiles, horizon) in the original space
    
    cb = qvd + repmat( (vd' - mvdb' - (mvdb' - vdVAR')), length(quantiles), 1 );
    % Parallely shifting the distribution of vdb's to make its new mean is
    % equal to the vdBc
    
end