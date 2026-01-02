function [ vd, vdBc, vdSe, cb, details] ...
    = BootstrapBlockR2( y, X, horizon, ylevel, trend, lagy, lagX, sizeBlock, ...
    NBootstrap, TBurnIn, quantiles, seIndicator, cbIndicator )
%Estimate the FEVD using the R2 method with block bootstrap.
% By using block-bootstrap method of Kilian and Kim (2011), 
%   we derive the bias-corrected FEVD, vdBc, and the corresponding 
%   standard errors, vdSe.
%
% INPUT:
% y(t)    = (T * 1) dependent variable.
% X(t, n) = (T * N) shock and controls. The first column is the shock. The
%   shock value will be de-meaned.
% horizon = ir and vd are derived up to the horizon-th period
% ylevel  = Delta y on the RHS when deriving the forecast errors if 
%           ylevel == 0, y on the RHS if ylevel == 1.
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
% sizeBlock   = size of the block for the block bootstrap method. Kilian 
%   and Kim (2011) note that a fixed block size (four in their simulations)
%   produces more accurate intervals. So we follow their suggestion.
%   (Default = 4).
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%               NBootstrap times to derive the bias, standard errors,
%               and confidence bands.        (Default = 2000).
% TBurnIn     = First TBurnIn observations are dropped as burn-in in 
%               bootstrap samples.
% qunatiles   = save the corresponding quantiles in cb. For example, if
%               quantiles = [0.05, 0.95], then cb will be a matrix whose
%               size is (2 * horizon+1) and represents 90% confidence
%               bands based on bootstrap.               (Default = []).
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
%               R2 estimator with a block bootstrap and bias correction.
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
if nargin < 7
    error('Not enough input arguments')
end

[T,N] = size(X);   % size of the time series

if nargin < 13
    cbIndicator = 0;
end
if nargin < 12
    seIndicator = 0;
end
if nargin < 11
    quantiles = [];
end
if nargin < 10
    TBurnIn = 100;
end
if nargin < 9
    NBootstrap = 2000;
end
if nargin < 8
    sizeBlock = 4;
end




% #2. Estimation: R2 method
[ vd, ~, details.vdSeAVarWPW, details.vdSeAVarWoPW ] ...
    = SubR2( y, X, lagy, lagX, horizon, ylevel, trend, 1 );


% #3. Preparation

% de-meaning the shock
X(:,1) = X(:,1) - mean(X(:,1));

% effective sample starts at t0.
t0 = max([lagy + (1 - ylevel); lagX(:)]) + 1;
% observations with t = 1, 2, dots, t0-1 are required to initialize the
%  forecast error (FE) regression. In other words, t0 is the first 
%  'effective' t. If ylevel == 0, then we need one more observation to take 
%   the first difference.


if ylevel == 0
    yendo = [nan; diff(y)];
else
    yendo = y;
end
% This variable will be needed when we construct the RHS of the FE
% regression. When ylevel == 0, we use Delta y(t) = y(t) - y(t-1).

% Things can be defined out of the for loop which, will be used in #4.
% LHS
dhy = lagmatrix(y, 0:-1:-horizon) - repmat(lagmatrix(y, 1), 1, horizon+1);
% dhy(t,h+1) = y(t+h) - y(t-1) will be used for
% the Forecast Error regression

% RHS, y
rhsy = lagmatrix(yendo, 1:lagy);
% either y(t-1), ..., y(t-lagy)
%     or Delta y(t-1), ... , Delta y(t-lagy)

% RHS, X
rhsX = zeros(T, sum(lagX));

idxn = 1;
columnFilled = 0;
while idxn <= N
    columnFilledNew = columnFilled + lagX(idxn);
    % lagX(idxn) new columns will be filled. time: t-1, ... t-lagX(idxn)
    rhsX(:, columnFilled+1:columnFilledNew) = lagmatrix(X(:,idxn), 1:lagX(idxn));
    % idxn-th vector of X matrix. Values at lag 1, ... , lagX(idxn)
    % are included.
    columnFilled = columnFilledNew;
    idxn = idxn + 1;
end

% RHS, trend
if trend == 0
    rhsTrend = ones(T,1);
elseif trend == 1
    rhsTrend = [ones(T,1), (1:1:T)'];
elseif trend == 2
    rhsTrend = [ones(T,1), (1:1:T)', (1:1:T)'.^2];
else
    rhsTrend = [ones(T,1), (1:1:T)', (1:1:T)'.^2, (1:1:T)'.^3];
end

% RHS, all
rhsAll = [ rhsX, rhsy, rhsTrend ] ;
% right-hand side for the Forecast Error regression

% Future realizations of shocks
xFuture = lagmatrix(X(:,1), 0:-1:-horizon);
% x_t, x_{t+1}, x_{t+2}, ..., x_{t+horizon}




% #4. Bootstrap

% initialization
vdb = zeros(NBootstrap, horizon+1);      % (b, h+1)   = s_{h}^{(b)}


for h=0:horizon
    
    % for each h, we use T - h - t0 + 1 observations. Thus, we need to
    % simulate samples with T - h - t0 + 1 observations. 
    Th = T - h - t0 + 1;
    nBlockh = ceil((Th + TBurnIn) / sizeBlock); % number of blocks to 
    % generate a vector series whose length is Th. We will discard initial
    % TBurnIn number of observations and some last observations to match
    % the length.
    
    
    blockShuffleh = ceil((Th - sizeBlock + 1) * rand(NBootstrap,nBlockh)) + (t0 - 1);
    % randomly generated natural numbers 
    % between t0 and T - h - sizeBlock + 1.
    
    for idxb = 1:NBootstrap
        
        % #4.1. Data generation: Bootstrap using the block bootstrap
        blockShufflehb = blockShuffleh(idxb,:);
        hbShuffle = kron(blockShufflehb, ones(1,sizeBlock));
        hbShuffle = hbShuffle + kron(ones(1,nBlockh), 0:1:(sizeBlock-1));
        
        lhsb = dhy(hbShuffle, h+1);  % left-hand side. y(t+h, b) - y(t-1, b)
        rhsb = rhsAll(hbShuffle, :); % right-hand side. 
        fexhb = xFuture(hbShuffle, 1:h+1); % x_{t,b}, x_{t+1,b}, ... x_{t+h,b}

        % Dropping BurnIn periods and the last several ones to match the
        % effective sample size Th.
        lhsb = lhsb(TBurnIn+1: TBurnIn + Th, :);
        rhsb = rhsb(TBurnIn+1: TBurnIn + Th, :);
        fexhb = fexhb(TBurnIn+1: TBurnIn + Th, :); 
        
        
        % #4.2. Estimate vd using simualated series
        feb = lhsb - rhsb * ((rhsb' * rhsb) \ (rhsb' * lhsb));
        
        Sigmaxb = fexhb' * fexhb / Th;
        theta2b = fexhb' * feb / Th;
        theta1b = Sigmaxb \ theta2b;
        theta3b = feb' * feb / Th;
        
        % variance decomposition using non-centered R2: (yb'xb)(xb'xb)^(-1)(xb'yb) / (yb'yb).
        % Note that we de-meaned x and y is mean zero because we included
        % an intercept in the forecast error regression.
        vdb(idxb,h+1) = theta2b' * theta1b / theta3b;

    end
  
end

% #5. Bias correction, standard errors, etc.
% #5.1. Impulse response
%   In this version, we don't report ir.

% #5.2. Variance decomposition
mvdb = mean(vdb,1)';       % average of vdb
vdBc = vd - ( mvdb - vd ); % bias correction
details.vdSeBootstrap = std(vdb, 0, 1)'; % standard error

if seIndicator == 0
    vdSe = details.vdSeBootstrap;
elseif seIndicator == 1
    vdSe = details.vdSeAVarWPW;
else
    vdSe = details.vdSeAVarWoPW;
end


% # 5.3. confidence bands
if isempty(quantiles)
    cb = [];
else
    if cbIndicator == 0     % based on the distribution of s_h^{(b)}
        qvd = quantile(vdb,quantiles,1);  % (quantiles, horizon) in the original space
        
        cb = qvd + repmat( (vd' - mvdb' - ( mvdb' - vd' )), length(quantiles), 1 );
        % Parallely shifting the distribution of vdb's to make its new mean is
        % equal to the vdBc
        
    elseif cbIndicator == 1    % use the AVAR and normal approximation with pre-whitening
        
        qvd = norminv(quantiles,0,1);
        qvd = qvd(:);
        
        cb = repmat(vdBc', length(qvd), 1) + qvd * details.vdSeAVarWPW';
        
    else % without prewhitening
        
        qvd = norminv(quantiles,0,1);
        qvd = qvd(:);
        
        cb = repmat(vdBc', length(qvd), 1) + qvd * details.vdSeAVarWoPW';
        
    end
end



