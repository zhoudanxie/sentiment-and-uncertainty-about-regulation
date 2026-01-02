function [ ir, vd, forecastError, irSeAVar, meanx, varx, mse, irAVar, idxt, dhy, rhsAllLP, residualLP, vFE, BetaHat, sigma2vHat, t0LP ] ...
    = SubLPB( y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator, extraOutputIndicator, semIndicator, XLP, lagyLP, lagXLP, XtIndicatorLP )
%Estimate the FEVD using the LP B method.
%
% Difference between A and B:
% A uses the denominator of Var(FE(t+h,t-1)) directly instead of
%   Sum ir_x(h)^2 * sigma_x^2 + Var(v(t+h,t-1)).
%   B method uses the latter.
%
% When (XLP, lagyLP, lagXLP, XtIndicatorLP) are not specified, we don't use two separated
%   regressions for the purposes of constructing forecast errors and
%   estimating impulse responses. This is the method described in the
%   paper.
% For VAR-based approaches to bias-correction, or simple implementation of
%   the method, one may simply assume that
%   XLP = X, lagyLP = lagy, lagXLP = lagX, and XtIndicatorLP = XtIndicator.
%
% The forecast errors and impulse responses can be obtained from
%   regressions with different lag lengthes and / or set of controls.
%   Therefore, we have two sets of regressors and lag lengthes, i.e.,
%   (X, lagy, lagX, XtIndicator) for forecast errors
%   and (XLP, lagyLP, lagXLP, XtIndicatorLP) for impulse responses.
% This feature would be useful when the joint asymptotic distribution is
%   required. For example, we may use the simultaneous equations method
%   for estimating impulse responses for their joint distribution. This
%   usually implies a very large system and its number of regressors can
%   easily exceed the effective sample size because the standard error is
%   obtained by Driscoll and Kraay (1998) method. See regress_dk.m for 
%   details. Therefore, we may drop other control variables or reduce lagy 
%   and lagX when estimating impulse responses. To allow for such an 
%   approach, we differentiate X and XLP where the former is used for 
%   constructing forecast errors and the latter for impulse responses.
%
% INPUT:
% y(t)    = (T * 1) dependent variable,
% X(t, n) = (T * N) shock and controls. The first column is the shock. The
%   shock value will be de-meaned.
% lagy    = y(-1), ... , y(-lagy) will be included on the RHS of
%   the local projections.
%   If ylevel == 0, then Delta y is used instead.
% lagX    = (1 * N) or (N * 1) vector. lagX(n) number of lagged values of
%   n-th variable in X is included in the local projections. We have
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
%   In an bivariate case, it can be just 1.
%
% NOTE: When there are many control variables or specific ordering
%   assumption is required for the purpose of identification, it is easier
%   to set lagy = 0 and include y or Delta y in X directly. Then
%   specification details can be implemented by choosing lagX and
%   XtIndicator.
%
% horizon = ir and vd are derived up to the horizon-th period.
% ylevel  = if 0: Delta y(t-1) and its lagged values on the RHS.
%           if 1: y(t-1) and its lagged values on the RHS.
% trend   = c                     if trend == 0,
%           c + dt                if trend == 1,
%           c + dt + et^2         if trend == 2,
%           c + dt + et^2 + ft^3  if trend == 3.
% dfIndicator = Whether to adjust the degrees of freedom when building the
%   forecast error variance.
%   E[FE(t+h,t-1)^2] is estimated by
%       = \sum_{i=0}^{h} \psi_{x,i}^2 * Var(x) + Var( v(t+h,t-1) ) where
%       v(t+h,t-1) = FE(t+h,t-1) - \sum_{i=0}^{h} \psi_{x,i} * x(t+h-i).
%   When we evaluate the second term, we calculate the following.
%       \sum ( v(t+h,t-1) )^2 / denominator
%       The denominator is
%       T_h - lagy - sum(lagX) - (trend + 1) if dfIndicator == 1, and
%       T_h o.w.,
%   where T_h = T - t_0 - h + 1 = effective number of observations
%       for the horizon h regression.
% extraOutputIndicator
%       return irSeAVar if the Indicator == 1,
%       o.w., it is [].
% semIndicator: An indicator for the simultaneous equations method. If the
%   indicator is zero, the projection equation for each horizon is
%   estimated separately, i.e., equation by equation. when the Indicator is 
%   one, the simultaneous equations system consisting of horizon + 1 number 
%   of equations is estimated at once instead. This gives us
%   a 'joint distribution' of the local projection estimators of ir across
%   horizons. The asymptotic variance when Indicator == 1 is obtained using
%   Driscoll and Kraay (1998) method. When this method is used, the
%   following specifications are assumed.
%   1) Horizon fixed effects ( a different constant for each h's ),
%   2) No pre-whitening when estimating the variance,
%   3) Small sample adjustment for the asymptotic variance with a factor of
%   (T/(T-1)) * (NT - 1)/(NT - # of parameters). For details, see
%   regress_dk.m.
% XLP    = ( T * NLP ) matrix. The first column is the shock. This is used
%   when estimating the impulse response coefficients. If not specified, it
%   is assumed to be the same as X. The first column of XLP should be the 
%   same as that of X.
% lagyLP = Number of lagged values of y or Delta y included on the RHS of
%   the local projection regressions when estimating impulse response
%   coefficients. If not specified, it is equal to lagy.
% lagXLP = Similar to lagyLP, this is lag lengthes used for XLP when
%   estimating impulse response coefficients. If not specified, it is lagX.
%
% OUTPUT:
% ir        = (1+horizon * 1) Local Projection based impulse responses. In
%   response to a unit shock in x(t).
% vd        = (1+horizon * 1) forecasting error variance decompositions
%   of y(t+h) - y(t-1) in relations to
%   x(t), x(t+1), x(t+2), ..., x(t+h) using LP.
% forecastError = (T * horizon+1) matrix. Its (t,h+1)-th element is
%   FE(t+h,t-1).
%   y(t+h) - y(t-1) = Projection( y(t-1), y(t-1), ..., x(t-1), ... )
%                       + FE(t+h,t-1).
% irSeAVar  = Standard error of ir using the Newey-West variance estimator
%   with the Bartlett kernel. The lag length is determined by the simple
%   rule suggested by Stock and Watson (2010): round(0.75*T^(1/3)). If
%   extraOutputIndicator == 0, it is not calculated.
% meanx     = E(shock)
% varx      = Var(shock)
% mse       = (horizon+1 * 1) E( FE(t+h,t)^2 )
%   = \sum_{i=0}^{h-1} \psi_{x,i}^2 * Var(x) +
%     Var( FE(t+h,t) - \sum_{i=0}^{h-1} \psi_{x,i} * x(t+h-i) )
% irAVar    = Variance-covariance matrix of ir. This represents the joint
% distribution of ir. It is obtained only when the simultaneous equations
% method is applied. If neither semIndicator nor extraOutputIndicator are
% one, this is not calculated.
% idxt      = (horizon+1 * 2): h+1-th column of forecastError representing
%   FE(t+h,t-1) has the following structure.
%               [First idxt(h+1,1)-1 elements    = NaN's ;
%               From idxt(h+1,1) to idxt(h+1,2)  = forecast errors ;
%               From idxt(h+1,2) + 1 to end      = NaN's]
% dhyLP, rhsAllLP, residualLP, vFE, BetaHat, sigma2vHat, t0LP:
%   Required as inputs to BootstrapAVarLPB.m to avoid deriving the same
%   quantities twice.
%   dhy: (T * horizon+1): (t,h+1) = y(t+h) - y(t-1). 'h+1' because we
%       have h=0.
%   rhsAllLP: (T * something): t-th row = q_t in the Appendix where x_t
%       comes first.
%   residualLP: (T * horizon+1): (t,h+1) = residual of regressing
%       y(t+h) - y(t-1) on t-th row of rhsAllLP, i.e. local projection
%       equation for the impulse response.
%   vFE: (T * horizon+1): (t,h+1) = \hat{fe}(t+h|t-1)
%       - \hat{\psi}_{x,0} x(t+h) - \cdots - \hat{\psi}_{x,h-1} x(t+1).
%   BetaHat: (something * horizon+1): h+1 -th column = coefficients of the
%       local projections, i.e. regressing h+1 -th column of dhyLP on
%       rhsAllLP.
%   sigma2vHat: (horizon+1 * 1): h+1-th element
%       = \hat{Var}(\hat{FE}(t+h|t-1)
%       - \hat{\psi}_{x,0} x(t+h) - \cdots - \hat{\psi}_{x,h} x(t))
%       = \hat{Var}(\hat{v}(t+h|t-1).
%   t0LP: When evaluating the moment conditions for the asymptotic
%       variance, the effective sample period is from max(t0LP, idxt(:,1))
%       to idxt(:,2).


% #1. Setup
% If not specified, the local projection regression uses the same set of
% controls.
if nargin < 12
    XLP = X;
end
if nargin < 13
    lagyLP = lagy;
end
if nargin < 14
    lagXLP = lagX;
end
if nargin < 15
    XtIndicatorLP = XtIndicator;
end

[T,N] = size(X);
[TLP, NLP] = size(XLP);

if N ~= length(lagX)
    error('lagX is not consistent with the number of variables in X.')
end

if N ~= length(XtIndicator)
    error('XtIndicator is not consistent with the number of variables in X.')
end

if XtIndicator(1) ~= 1
    error('XtIndicator(1) should be one.')
end

if T ~= TLP
    error('Check the sample sizes of X and XLP.')
end

if ~isequal(X(:,1), XLP(:,1)) % in both matrices, shocks should be the same.
    error('X(:,1) is not the same as XLP(:,1).')
end

if semIndicator == 1 && extraOutputIndicator == 1 && (horizon + 1) * (sum(lagXLP) + NLP + lagyLP + trend) > T
    % When simultaneous equations are estimated, the Driscoll and Kraay
    % method is basically clustering the observations by the time.
    % Therefore, the effective sample size is just T. However, the number
    % of variables in the regression increases linearly in horizon. Thus,
    % for example, if horizon = 20, lagXLP = 3, N = 1, lagy = 3, and
    % trend = 0, this already implies 140 variables. In such a case, the
    % estimated standard errors would not be reliable.
    error('Too many regressors relative to the effective sample size.')
end

% initialization
ir   = zeros(horizon+1, 1);
idxt = zeros(horizon+1, 2);
vFE  = nan(T, horizon+1);
sigma2vHat = zeros(horizon+1, 1);

forecastError = nan(T, horizon+1);
mse = zeros(horizon+1, 1);

if extraOutputIndicator == 0   % In this case we do not calculate the AVar.
    irSeAVar = [];
    irAVar = [];
elseif semIndicator == 0
    irSeAVar = zeros(horizon+1, 1);
    irAVar = [];
end

% First and second moments of shocks
meanx = mean(X(:,1));
varx  = var(X(:,1));

% effective sample starts at t0.
t0 = max([lagy + (1 - ylevel); lagX(:);1]) + 1;
t0LP = max([lagyLP + (1 - ylevel); lagXLP(:);1]) + 1;
% observations with t = 1, 2, dots, t0-1 are required to initialize the
%  local projections. In other words, t0 is the first 'effective' t. If ylevel
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

% de-meaning x
X(:,1) = X(:,1) - meanx;
XLP(:,1) = XLP(:,1) - meanx;

% #2. Preparation
% Things can be defined out of the for loop which will be used in #3 and 4.
% LHS
dhy = lagmatrix(y, 0:-1:-horizon) - repmat(lagmatrix(y, 1), 1, horizon + 1);
% dhy(t,h) = y(t+h) - y(t-1) will be used for local projections

% RHS, y
rhsyFE = lagmatrix(yendo, 1:lagy);
% either y(t-1), ..., y(t-lagy) or Delta y(t-1), ... , Delta y(t-lagy)
rhsyLP = lagmatrix(yendo, 1:lagyLP);
% either y(t-1), ..., y(t-lagyLP) or Delta y(t-1), ... , Delta y(t-lagyLP)

% RHS, X
% For Forecast errors
rhsXFE = zeros(T, sum(lagX) + sum(XtIndicator));

% contemporaneous X's are in the first few columns
columnFilled = 0;
columnFilledNew = sum(XtIndicator);
rhsXFE(:, columnFilled + 1 : columnFilledNew) = X(:, logical(XtIndicator));
columnFilled = columnFilledNew;

% Lagged values
idxn = 1;
while idxn <= N
    columnFilledNew = columnFilled + lagX(idxn);
    % XtIndicator(idxn) + lagX(idxn) new columns will be filled. time: t-1, ... t-lagX(idxn)
    rhsXFE(:, columnFilled+1:columnFilledNew) = lagmatrix(X(:,idxn), 1:lagX(idxn));
    columnFilled = columnFilledNew;
    idxn = idxn + 1;
end

% For Impulse Reponses
rhsXLP = zeros(T, sum(lagXLP) + sum(XtIndicatorLP));
% contemporaneous X's are in the first few columns
columnFilled = 0;
columnFilledNew = sum(XtIndicatorLP);
rhsXLP(:, columnFilled + 1 : columnFilledNew) = XLP(:, logical(XtIndicatorLP));
columnFilled = columnFilledNew;
% Lagged values
idxn = 1;
while idxn <= NLP
    columnFilledNew = columnFilled + lagXLP(idxn);
    rhsXLP(:, columnFilled+1:columnFilledNew) = lagmatrix(XLP(:,idxn), 1:lagXLP(idxn));
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
rhsAllFE = [ rhsXFE, rhsyFE, rhsTrend];
% right-hand side for the forecast errors
rhsAllLP = [ rhsXLP, rhsyLP, rhsTrend];
% right-hand side for the impulse responses.


% Future realizations of shocks
xFuture = lagmatrix(X(:,1), 0:-1:-horizon);
% x_t, x_{t+1}, x_{t+2}, ..., x_{t+horizon}

% Orthogonalization of X for the purpose of constructing forecast errors
Xperp = nan(size(X));
Xperp(t0:end, :) = X(t0:end, :) - rhsAllFE(t0:end, sum(XtIndicator)+1 : end ) * ...
    (( rhsAllFE(t0:end, sum(XtIndicator)+1 : end )' * rhsAllFE(t0:end, sum(XtIndicator)+1 : end ) ) \ ...
    ( rhsAllFE(t0:end, sum(XtIndicator)+1 : end )' * X(t0:end, :) ));


% initialization
BetaHat = zeros(size(rhsAllLP,2), horizon+1);
residualLP = nan(T, horizon+1);

if semIndicator ~= 1        % Equation by equation
    for h = 0:horizon
        
        % #3. Local Projection and ir
        % Regress y(t+h) - y(t-1) on x(t), x(t-1), ....
        TTempLP = T - h - t0LP + 1; % Sample size for the local projection regression with the horizon h.
        
        % #3.1. LHS
        % As discussed above, the first effective observation is with t = t0.
        % Similarly, the last observation is with t = T - h.
        lhsLP = dhy(t0LP:T-h, h+1); % left-hand side
        
        % #3.2. RHS, y, X, and trend
        rhsLP = rhsAllLP(t0LP:T-h,:); % right-hand side
        % We use rhsy from the second columnm, because y(t) or Delta y(t) is
        % not included on the RHS.
        
        % #3.3. ir and irSeAVar
        betah = ((rhsLP' * rhsLP) \ (rhsLP' * lhsLP));
        BetaHat(:,h+1) = betah;
        ir(h+1) = betah(1);      % Local projection estimator, psi_{x,h}
        
        if extraOutputIndicator == 1  % standard error using AVar.
            feLP = lhsLP - rhsLP * betah;
            % residual of the local projection regression
            
            residualLP(t0LP:T-h, h+1) = feLP;
            
            % AVar
            SigmaX = rhsLP' * rhsLP / TTempLP;
            Omega  = lrv_nw(rhsLP.*repmat(feLP, 1, sum(lagXLP) + sum(XtIndicatorLP) + lagyLP + trend + 1), round(0.75 * TTempLP^(1/3) - 1));
            V = (SigmaX \ Omega / SigmaX ) / TTempLP;
            irSeAVar(h+1,1) = sqrt(V(1,1));
        end
        
        % #4. Building the forecast errors.
        % if XLP == X, lagyLP == lagy, lagXLP == lagX, and XtIndicatorLP ==
        % XtIndicator, we use the above regression directly for forecast
        % errors too.
        
        if isequal(XLP, X) && isequal(lagyLP, lagy) && isequal(lagXLP, lagX) && isequal(XtIndicatorLP, XtIndicator)
            % Re-use the above regression
            lhsFE = lhsLP;
            rhsFE = rhsLP;
        else
            % Run a separate regression for the forecast error
            % Regress y(t+h) - y(t) on Information(t).
            
            % #4.1. LHS
            % As discussed above, the first effective observation is with t = t0.
            % Similarly, the last observation is with t = T - h.
            lhsFE = dhy(t0:T-h,h+1); % left-hand side
            
            % #4.2. RHS, y, X, and trend
            rhsFE = rhsAllFE(t0:T-h,:); % right-hand side
            
            % #4.3. Regression and the FE
            betah = ((rhsFE' * rhsFE) \ (rhsFE' * lhsFE));
        end
        
        
        feFE = lhsFE - rhsFE * betah + Xperp(t0:T-h, logical(XtIndicator)) * betah(1:sum(XtIndicator));
        % residual of the OLS and the forecasting error
        
        forecastError(t0:T-h, h+1) = feFE; % horizon h forecast errors, FE(t+h, t-1).
        idxt(h+1,:) = [t0, T-h];  % Saving periods of the forecast errors.
        
        % #4.4. MSE
        TTempFE = T - h - t0 + 1;
        % E [ FE(T+h, t)^2 ]. Whether to adjust the degrees of freedom or not.
        
        xFutureh = xFuture(t0:T-h, h+1:-1:1); % x_{t+h}, x_{t+h-1}, ... , x_{t+1}, x_t
        
        vh = feFE - xFutureh * ir(1:h+1);
        % v(t+h,t-1) = FE(t+h,t-1) - \sum_{i=0}^{h} \psi_{x,i} * x(t+h-i)
        
        vFE(t0:T-h,h+1) = vh;
        
        sigma2vHat(h+1) = sum(vh.^2) / ...
            (TTempFE - ( 1 - dfIndicator ) * ( lagy + sum(lagX) + (trend + 1) ));
        mse(h+1) = sum(ir(1:h+1).^2) * varx + sigma2vHat(h+1);
        %  if dfIndicator == 1, the denominator is TTempFE. If zero, the
        %  denominator is TTempFE - ( 1 - dfIndicator ) * ( lagy + sum(lagX) + (trend + 1) )
        
        idxt(h+1,:) = [t0, T-h];  % Saving periods of the forecast errors.
    end
    
else        % semIndicator == 1, simultaneous equations system
    
    % #3. Local Projection and ir: Simultaneous equations system
    % Initialization
    rhsSEM = [];
    lhsSEM = [];
    tidxSEM = [];
    iidxSEM = [];
    rhsSEMPerp = [];
    
    for h = 0:horizon
        
        % #3.1. LHS
        % As discussed above, the first effective observation is with t = t0.
        % Similarly, the last observation is with t = T - h.
        lhsLP = dhy(t0LP:T-h, h+1); % left-hand side
        
        % #3.2. RHS, y, X, and trend
        rhsLP = rhsAllLP(t0LP:T-h,:); % right-hand side
        rhsPerp = Xperp(t0LP:T-h, logical(XtIndicatorLP));
        
        % #3.3. Stacking the equations
        TTempLP = T - h - t0LP + 1;   % effective sample size
        
        lhsSEM = [lhsSEM; lhsLP];
        rhsSEM = blkdiag(rhsSEM, rhsLP);
        rhsSEMPerp = blkdiag(rhsSEMPerp, rhsPerp);
        
        tidxSEM = [tidxSEM; (t0LP + h : T)'];  % time index for the stacked vector for pooled OLS.
        % Note that the last observation would be like
        % y(T) - y(T-h-1) = ftn(x(t), ...) + error(T, T-h-1). This
        % equation, especially error(T,T-h-1) is included in time 'T' information set, not T-h.
        iidxSEM = [iidxSEM; h * ones(TTempLP, 1)];     % entity index
        
    end
    
    [b, bVar, residualDK] = regress_dk(lhsSEM, rhsSEM, iidxSEM, tidxSEM, 0, -2, 0, 0, extraOutputIndicator);
    % Use 'regress_dk.m' function to estimate the system.
    
    % re-shaping vectorized residualDK
    t1 = 0;
    for h = 0:horizon
        t2 = t1 + (T-h - t0LP) + 1;
        residualLP(t0LP:T-h, h+1) = residualDK(t1+1 : t2);
        t1 = t2;
    end
    
    % selecting coefficients on x(t)'s only.
    iridx       = (0:1:horizon)*( sum(XtIndicatorLP) + sum(lagXLP) + lagyLP + trend + 1) + 1;
    % Each block diagonal component of rhsLP's have 'sum(XtIndicatorLP)
    %   + sum(lagXLP) + lagyLP + trend + 1' number of elements and x(t)'s
    %   appear on the first column of each rhsLP.
    
    BetaHat = reshape(b, size(BetaHat));
    
    ir      = b(iridx);
    
    if extraOutputIndicator == 1  % standard error using AVar.
        
        irAVar = bVar(iridx, iridx); % estimated variance for impulse responses
        
        % Due to numerical issues, this may not be positive semi-definite.
        [~,chk]=cholcov(irAVar);
        while chk~=0
            irAVar = (irAVar + irAVar')/2;  % make it symmetric
            [VM,DM] = eig(irAVar);          % Diagonalize
            DM = max(DM,0);                 % Eliminate negative eigenvalues
            irAVar  = VM * DM * VM';
            [~,chk] = cholcov(irAVar);
        end
        
        irSeAVar = sqrt(max(diag(irAVar),eps));
        
    end
    
    % if XLP == X, lagyLP == lagy, lagXLP == lagX, and XtIndicatorLP ==
    % XtIndicator, we use the above regression directly for forecast errors.
    
    if isequal(XLP, X) && isequal(lagyLP, lagy) && isequal(lagXLP, lagX) && isequal(XtIndicatorLP, XtIndicator)
        % Re-use the above regression
    else
        % Estimate a separate system of equations
        
        % Initialization
        rhsSEM = [];
        lhsSEM = [];
        tidxSEM = [];
        iidxSEM = [];
        rhsSEMPerp = [];
        
        for h = 0:horizon
            
            % #4.1. LHS
            % As discussed above, the first effective observation is with t = t0.
            % Similarly, the last observation is with t = T - h.
            lhsFE = dhy(t0:T-h, h+1); % left-hand side
            
            % #4.2. RHS, y, X, and trend
            rhsFE = rhsAllFE(t0:T-h,:); % right-hand side
            rhsPerp = Xperp(t0:T-h, logical(XtIndicator));
            
            % #3.3. Stacking the equations
            TTempFE = T - h - t0 + 1;   % effective sample size
            
            lhsSEM = [lhsSEM; lhsFE];
            rhsSEM = blkdiag(rhsSEM, rhsFE);
            rhsSEMPerp = blkdiag(rhsSEMPerp, rhsPerp);
            
            tidxSEM = [tidxSEM; (t0 + h : T)'];  % time index for the stacked vector for pooled OLS.
            % Note that the last observation would be like
            % y(T) - y(T-h-1) = ftn(x(t), ...) + error(T, T-h-1). This
            % equation, especially error(T,T-h-1) is included in time 'T' information set, not T-h.
            iidxSEM = [iidxSEM; h * ones(TTempFE, 1)];     % entity index
            
        end
        
        [b, ~, residualDK] = regress_dk(lhsSEM, rhsSEM, iidxSEM, tidxSEM, 0, -2, 0, 0, extraOutputIndicator);
        % Use 'regress_dk.m' function to estimate the system.
    end
    
    % Adding unnecessarily purified components to transform the residuals
    % into the 'forecast errors.'
    perpidx       = repmat((0:1:horizon)*( sum(XtIndicator) + sum(lagX) + lagy + trend + 1 ), sum(XtIndicator), 1) ...
        + repmat( (1:sum(XtIndicator))', 1, horizon+1 );
    perpidx = perpidx(:);  % perpidx-th variables in the large system corresponds to time t variables.
    
    residualFE = residualDK + rhsSEMPerp * b(perpidx);
    
    t1 = 0;
    for h = 0:horizon
        TTempFE = T - h - t0 + 1;
        
        t2 = t1 + TTempFE;
        
        % re-shaping vectorized forecast errors
        feFE = residualFE(t1+1 : t2);
        forecastError(t0:T-h, h+1) = feFE;
        
        t1 = t2;
        
        % Saving periods of the forecast errors.
        idxt(h+1,:) = [t0, T-h];
        
        % #3.4. MSE
        xFutureh = xFuture(t0:T-h, h+1:-1:1); % x_{t+h}, x_{t+h-1}, ... , x_{t+1}, x_t
        vh = feFE - xFutureh * ir(1:h+1);
        % v(t+h,t-1) = FE(t+h,t-1) - \sum_{i=0}^{h} \psi_{x,i} * x(t+h-i)
        
        vFE(t0:T-h,h+1) = vh;
        
        sigma2vHat(h+1) = sum(vh.^2) / ...
            (TTempFE - ( 1 - dfIndicator ) * ( lagy + sum(lagX) + (trend + 1) ));
        mse(h+1) = sum(ir(1:h+1).^2) * varx + sigma2vHat(h+1);
        %  if dfIndicator == 1, the denominator is TTempFE. If zero, the
        %  denominator is TTempFE - ( 1 - dfIndicator ) * ( lagy + sum(lagX) + (trend + 1) )
    end
end


% #5. variance decomposition
vd = varx * cumsum(ir.^2) ./ mse ;
