function [ vd, forecastError, seAVarWPW, seAVarWoPW, idxt, theta, GOGTWPW, GOGTWoPW, effectiveT ] ...
    = SubR2( y, X, lagy, lagX, horizon, ylevel, trend, extraOutputIndicator )
%Estimate the FEVD using the R2 method
% where the forecast errors are estimated using projections.
% INPUT:
% y(t)    = (T * 1) dependent variable,
% X(t, n) = (T * N) shock and controls. The first column is the shock. The
%  shock value will be de-meaned.
% lagy    = number of lagged y(t) or Delta y(t) included in the forecasting
%  error (FE) regression. We have y(t-1), y(t-2), ... , y(t-lagy) on the
%  RHS.
% lagX    = (1 * N) or (N * 1) vector. lagX(n) number of lagged values of
%  n-th variable in X is included in the forecasting error regression. We
%  have
%  X_1(t-1), ... , X_1(t-lagX(1)), ...
%  X_2(t-1), ... , X_2(t-lagX(2)), ...
%  ...
%  X_N(t-1), ... , X_N(t-lagX(N)) on the RHS.
% horizon = vd are derived up to the horizon-th period. (s_0, ... , s_horizon)'.
% ylevel  = Delta y on the RHS when deriving the FE if ylevel == 0,
%           y on the RHS if ylevel == 1.
% trend   = c                     if trend == 0,
%           c + dt                if trend == 1,
%           c + dt + et^2         if trend == 2,
%           c + dt + et^2 + ft^3  if trend == 3.
% extraOutputIndicator
%       return seAVarWPW, seAVarWoPW, GOGTWPW, and GOGTWoPW if the 
%       Indicator == 1, o.w., those variables are [].
% OUTPUT:
% vd       = forecasting error variance decomposition of y(t+h) - y(t-1) in
%  relations to x(t), x(t+1), x(t+2), ..., x(t+h) using R2.
% forecastError = (T * horizon+1) matrix. 
%  Its (t,h+1)-th element is FE(t+h,t-1).
%  y(t+h) - y(t-1) = Projection( y(t-1), ... , X_(t-1), ... ) + FE(t+h,t-1).
% seAVarWPW  = standard error based on the asymptotic variance
%  using the moment conditions with pre-whitening by VAR(1) where vd is
%  used in the place of s_h. For those who want to have the s.e. based on 
%  vdBc instead of vd, please refer SimulationAVarR2.m.
% seAVarWoPW = standard error based on the asymptotic variance
%  using the moment condition without pre-whitening where vd is
%  used in the place of s_h. For those who want to have the s.e. based on 
%  vdBc instead of vd, please refer SimulationAVarR2.m.
%  When the Newey-West variance estimator is used, we usse the Bartlett
%  kernel. The lag length is determined by the simple rule suggested by
%  Stock and Watson (2010): round(0.75*T^(1/3)).
% idxt      = (horizon+1 * 2): h-th column of forecastError has the following
%   structure. [First idxt(h,1) - 1 elements      = NaN's ; 
%               From  idxt(h,1)     to  idxt(h,2) = forecast errors ;
%               From  idxt(h,2) + 1 to end        = NaN's]
% GOGTWPW    = (horizon+1 * 1) cell. (G)^-1 * OmegaWPW * (G')^-1 / T 
% GOGTWoPW   = (horizon+1 * 1) cell. (G)^-1 * OmegaWoPW * (G')^-1 / T
%  For each h, GOGT gives the asymptotic variance of 
%   ( (X'X)^-1 * (X'f), X'f, f'f ) with or without pre-whitening.
% theta      = (horizon+1 * 1) cell. For each h = 0, ... , horizon,
%   theta(h+1,1) is about s_{h} and it is a (2h + 3) dimensional 
%   vector of ( (X'X)^-1 * (X'f), X'f, f'f ). 
% effectiveT = Sample size for each horizon


% #1. Setup
[T,N] = size(X);

if N ~= length(lagX)
    error('lagX is not consistent with the number of variables in X.')
end

% initialization
vd = zeros(horizon+1,1);
forecastError = nan(T, horizon+1);
if extraOutputIndicator == 0   % In this case we do not calculate the AVar.
    seAVarWPW = [];
    seAVarWoPW = [];
    GOGTWPW = [];
    GOGTWoPW = [];
    effectiveT = [];
else
    seAVarWPW = zeros(horizon+1, 1);
    seAVarWoPW = zeros(horizon+1, 1);
    GOGTWPW = cell(horizon+1,1);
    GOGTWoPW = cell(horizon+1,1);
    effectiveT = zeros(horizon+1, 1);
end

idxt = zeros(horizon+1, 2);
theta = cell(horizon+1, 1);

% de-meaning the shock
X(:,1) = X(:,1) - mean(X(:,1));

% effective sample starts at t0.
t0 = max([lagy + (1 - ylevel); lagX(:)]) + 1;
% observations with t = 1, 2, dots, t0-1 are required to initialize the
%  FE regression. In other words, t0 is the first 'effective' t. If ylevel
%  == 0, then we need one more observation to take the first difference.
% It is designed to be at least 2, not 1. It is needed when we build local
%  projections, because we need to take y(t+h) - y(t-1), and y(t-1) is
%  defined only for t >= 2.


if ylevel == 0
    yendo = [nan; diff(y)];
else
    yendo = y;
end
% This variable will be needed when we construct the RHS of the FE
% regression. When ylevel == 0, we use Delta y(t) = y(t) - y(t-1).


% #2. Preparation
% Things can be defined out of the for loop which will be used in #3.
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
% right-hand side in the Forecast Error regression

% Future realizations of shocks
xFuture = lagmatrix(X(:,1), 0:-1:-horizon);
% x_t, x_{t+1}, x_{t+2}, ..., x_{t+horizon}



for h = 0:horizon
    
    % #3. Building the forecast errors.
    % Regress y(t+h) - y(t-1) on the information at the time t-1.
    
    % #3.1. LHS
    % As discussed above, the first effective observation is with t = t0.
    % Similarly, the last observation is with t = T - h.
    lhs = dhy(t0:T-h,h+1); % left-hand side
    
    % #3.2. RHS, y, X, trend
    rhs = rhsAll(t0:T-h,:); % right-hand side
    
    % #3.3. Regression and the FE
    fe = lhs - rhs * ((rhs' * rhs) \ (rhs' * lhs));
    % residual of the OLS = the forecast error
    
    forecastError(t0:T-h, h+1) = fe; 
    % horizon h forecast errors, FE(t+h,t-1)
    
    idxt(h+1,:) = [t0, T-h]; 
    % Effetive t's used in the regression
    
    % #4. R2 method, FEVD, and Standard errors using the moment conditions
    % #4.1. R2
    fexh = xFuture(t0:T-h, 1:h+1); % x_t, x_{t+1}, ... x_{t+h}
    
    % Some moments. See the paper for details.
    TTemp = T - h - t0 + 1;
    Sigmax = fexh' * fexh / TTemp;
    theta2 = fexh' * fe / TTemp;
    theta1 = Sigmax \ theta2;
    theta3 = fe' * fe / TTemp;
    
    theta{h+1,1} = [theta1; theta2; theta3];
    
    % variance decomposition using non-centered R2: (y'x)(x'x)^(-1)(x'y) / (y'y).
    % Note that we de-meaned x, and y is mean zero because we included the
    % intercept in the forecast error regression.
    vd(h+1) = theta2' * theta1 / theta3;
    
    
    % #3.2. Standard errors using the moment conditions
    if extraOutputIndicator ~= 0   
        % If the flag == 0, we do not calculate the AVar. Useful for
        % bootstraps.
        
        effectiveT(h+1,1) = TTemp;
        
        residual = fe - fexh*theta1;   % y - X * beta = residual
        % See Appendix of the paper for details
        G = - blkdiag(Sigmax, eye(h+2));
        xi = 1/theta3 * [theta2; theta1; -vd(h+1)];
        
        % With pre-whitening
        % VAR set-up
        ZFull = [repmat(residual,1,h+1).*fexh - repmat(mean(repmat(residual,1,h+1).*fexh),TTemp,1), repmat(fe,1,h+1).*fexh - repmat(theta2',TTemp,1), fe.^2 - repmat(theta3,TTemp,1)];
        Z_1 = ZFull(2:end,:);
        Z_0 = ZFull(1:end-1,:);
        A = (Z_0' * Z_0) \ (Z_0' * Z_1);
        A = A';
        % VAR(1) residual
        U = Z_1 - Z_0 * A';
        OmegaWPW = (eye(2*h+3) - A) \ lrv_nw(U, round(0.75*TTemp^(1/3)-1)) / (eye(2*h+3) - A');
        aVarWPW = xi' * (G \ OmegaWPW / G') * xi;
        seAVarWPW(h+1) = sqrt(aVarWPW/TTemp);
        GOGTWPW{h+1,1} = (G \ OmegaWPW / G') / TTemp;
        
        % Without pre-whitening
        OmegaWoPW = lrv_nw(ZFull, round(0.75*TTemp^(1/3)-1));
        aVarWoPW = xi' * (G \ OmegaWoPW / G') * xi;
        seAVarWoPW(h+1) = sqrt(aVarWoPW/TTemp);
        GOGTWoPW{h+1,1} = (G \ OmegaWoPW / G') / TTemp;
    end
    
    
end

