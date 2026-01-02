function [ vd, vdBc, vdSe, cb, details] ...
    = BootstrapVARR2( y, X, horizon, ylevel, trend, lagy, lagX, lagVARSet, ...
    XVAROrder, yPlace, ICIndicator, NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator )
%Estimate the FEVD using the R2 method with VAR-based bootstrap.
% By bootstrapping the estimated VAR model, we derive the bias-corrected
% FEVD, vdBc, and the corresponding standard errors, vdSe.
%
% INPUT:
% y(t)    = (T * 1) dependent variable.
% X(t, n) = (T * N) shock and controls. The first column is the shock. The
%   shock value will be de-meaned.
% horizon = ir and vd are derived up to the horizon-th period
% ylevel  = Delta y on the RHS when deriving forecast errors and estimating 
%           a VAR if ylevel == 0, y on the RHS if ylevel == 1.
% trend   = c                     if trend == 0,
%           c + dt                if trend == 1,
%           c + dt + et^2         if trend == 2,
%           c + dt + et^2 + ft^3  if trend == 3.
% lagy    = y(-1), ... , y(-lagy) will be included on the RHS to
%   build the forecast errors. If ylevel == 0, then Delta y is used instead.
% lagX    = (1 * N) or (N * 1) vector. lagX(n) number of lagged values of
%   n-th variable in X is included in the forecasting error regression. We
%   have
%   X_1(t-1), ... , X_1(t-lagX(1)), ...
%   X_2(t-1), ... , X_2(t-lagX(2)), ...
%   ...
%   X_N(t-1), ... , X_N(t-lagX(N)) on the RHS.
% lagVARSet   = Set of candidate lag lengthes. The lag length is determined
%	by an information criterion. It should be either a column or a row 
%   vector.
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
%               0: BIC, 1: HQIC, 2: AIC.     (Default = 1).
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%               NBootstrap times to derive biases, standard errors,
%               and confidence bands.        (Default = 2000).
% TBurnIn     = First TBurnIn observations are dropped as burn-in in 
%               bootstrap samples.
% qunatiles   = save the corresponding quantiles in cb. For example, if
%               quantiles = [0.05, 0.95], then cb will be a matrix whose
%               size is (2 * horizon+1) and represents 90% confidence
%               bands based on bootstrap.       (Default = []).
% seIndicator = Indicator for which s.e. to report as vdSe and to use when
%               constructing cb among bootstrap based one (vdSeBootstrap),
%               asymptotic variance (AVar) with pre-whitening (PW)
%               (vdSeAVarWPW), and AVar without PW (vdSeAVarWoPW).
%               0 -> vdSe = details.vdSeBootstrap,
%               1 -> vdSe = details.vdSeAVarWPW
%               2 -> vdSe = details.vdSeAVarWoPW   (Default = 0).
% cbIndicator = Indicator for which confidence band to report as cb.
%   	0 -> Bootstrap based cb
%       1 -> Normal approximation with details.vdSeAVarWPW
%       2 -> Normal approximation with details.vdSeAVarWoPW  (Default = 0).
%
% OUTPUT:
% vd       = forecasting error variance decompositions of y(t+h)-y(t-1) in
%               relations to x(t), x(t+1), x(t+2), ..., x(t+h).
% vdBc     = bias-corrected FEVD.
%               R2 estimator with VAR based bootstrap and bias correction.
% vdSe = standard errors for vd and vdBc. See 'seIndicator' above.
% cb       = condifence bands. See above explanations for 'quantiles' and 
%               'cbIndicator.'
% details  = A structure consists of other results.
% details.vdSeBootstrap / vdSeAVarWPW / vdSeAVarWoPW:
%               Bootstrap: bootstrap, 
%               WPW: with pre-whitening, WoPW: without pre-whitening
% Asymptotic variance is based on the moment conditions. 
% details.lagVAR = Chosen lag length for the VAR model by an IC.


% #1. Setting up the default parameters
if nargin < 8
    error('Not enough input arguments')
end

[Ty,N] = size(X);   % size of the time series

if nargin < 16
    cbIndicator = 0;
end
if nargin < 15
    seIndicator = 0;
end
if nargin < 14
    quantiles = [];
end
if nargin < 13
    TBurnIn = 100;
end
if nargin < 12
    NBootstrap = 2000;
end
if nargin < 11
    ICIndicator = 1;
end
if nargin < 10
    yPlace = N+1;
end
if nargin < 9
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
    % If there's only one candidate, then that is the lagVAR.
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
    % In this case, one should not select y or Delta y from X through
    % XVAROrder. Instead, one adds y or Delta y via yPlace and lagy.
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

% #3.2. Estimation: R2 method
[ vd, ~, details.vdSeAVarWPW, details.vdSeAVarWoPW ] ...
    = SubR2( y, X, lagy, lagX, horizon, ylevel, trend, 1 );


% #4. Bootstrap
% initialization
vdb = zeros(NBootstrap, horizon+1);      % (b, h+1)   = s_{h}^{(b)}

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
    vdb(idxb,:) = SubR2( yb, Xb, lagy, lagX, horizon, ylevel, trend, 0 );

end

% #5. Bias correction, standard errors, etc.
% #5.1. Impulse response
%   In this version, we don't report ir.

% #5.2. Variance decomposition
mvdb = mean(vdb,1)';   % average of vdb
vdBc = vd - ( mvdb - vdVAR ); % bias correction 
details.vdSeBootstrap = std(vdb, 0, 1)'; % standard error

if seIndicator == 0
    vdSe = details.vdSeBootstrap;
elseif seIndicator == 1
    vdSe = details.vdSeAVarWPW;
else
    vdSe = details.vdSeAVarWoPW;
end

% # 5.3. lagVAR
details.lagVAR = lagVAR;


% # 5.4. confidence bands
if isempty(quantiles)
    cb = [];
else
    if cbIndicator == 0     % based on the distribution of s_h^{(b)}
        qvd = quantile(vdb,quantiles,1);  % (quantiles, horizon) in the original space
        
        cb = qvd + repmat( (vd' - mvdb' - ( mvdb' - vdVAR' )), length(quantiles), 1 );
        % Parallely shifting the distribution of vdb's to make its new mean is
        % equal to the vdBc
        
    elseif cbIndicator == 1    % use the AVar and normal approximation. with pre-whitening
        
        qvd = norminv(quantiles,0,1);
        qvd = qvd(:);
        
        cb = repmat(vdBc', length(qvd), 1) + qvd * details.vdSeAVarWPW';
        
    else % without prewhitening
        
        qvd = norminv(quantiles,0,1);
        qvd = qvd(:);
        
        cb = repmat(vdBc', length(qvd), 1) + qvd * details.vdSeAVarWoPW';
        
    end
end



