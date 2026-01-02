function [ vd, vdBc, vdSe, cb, details] ...
    = SimulationAVarR2( y, X, horizon, ylevel, trend, lagy, lagX, ...
    NBootstrap, quantiles, seIndicator, cbIndicator, boundIndicator )
%The FEVD is obtained by the R2 method.
%
% By simulating ((X'X)^-1 * (X'f), X'f, f'f) simultaneously from their 
% estimated asymptotic distribution, we derive the bias-corrected
% FEVD, vdBc, and the corresponding standard errors, vdSe.
% Given a large number of moment conditions for each horizon, joint
% inference is not available in this code, although it is possible in
% theory.
% See the paper and Appendix A for details.
%
% INPUT:
% y(t)    = (T * 1) dependent variable.
% X(t, n) = (T * N) shock and controls. The first column is the shock. The
%  shock value will be de-meaned.
% horizon = vd are derived up to the horizon-th period
% ylevel  = 1) if ylevel == 0
%   i) Delta y(t-1) and its lagged values on the RHS when constructing
% 	the forecast errors.
%           2) if ylevel == 1
%   i) y(t-1) and its lagged values on the RHS when constructing
% 	the forecast errors.
% trend   = c                     if trend == 0,
%           c + dt                if trend == 1,
%           c + dt + et^2         if trend == 2,
%           c + dt + et^2 + ft^3  if trend == 3.
% lagy    = number of lagged y(t) or Delta y(t) included in the forecasting
%   error (FE) regression. We have y(t-1), y(t-2), ... , y(t-lagy) on the
%   RHS.
% lagX    = (1 * N) or (N * 1) vector. lagX(n) number of lagged values of
%   n-th variable in X is included in the forecasting error regression. We
%   have
%   X_1(t-1), ... , X_1(t-lagX(1)), ...
%   X_2(t-1), ... , X_2(t-lagX(2)), ...
%   ...
%   X_N(t-1), ... , X_N(t-lagX(N)) on the RHS.
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%   NBootstrap times to derive the bias, standard errors, and confidence
%   bands.        (Default = 2,000).
% qunatiles   = save the corresponding quantiles in cb. For example, if
%	quantiles = [0.05, 0.95], then cb will be a matrix whose size is
%   (2 * horizon+1) and is representing 90% confidence bands. (Default = []).
% seIndicator = Indicator for which s.e. to report as vdSe and to use when
%   constructing cb among bootstrap based one (vdSeBootstrap), asymptotic
%   variance (AVar) with pre-whitening (PW) (vdSeAVarWPW), and AVar
%   without PW (vdSeAVarWoPW).
%   	0 -> vdSe = details.vdSeBootstrap,
%       1 -> vdSe = details.vdSeAVarWPW
%       2 -> vdSe = details.vdSeAVarWoPW   (Default = 1).
% cbIndicator = Indicator for which confidence band to report as cb.
%   	0 -> Bootstrap based cb
%       1 -> Normal approximation with details.vdSeAVarWPW
%       2 -> Normal approximation with details.vdSeAVarWoPW  (Default = 1).
% boundIndicator: If 1 -> vd based on simulated sample is forced to 
%   be bounded by 0 and 1. (Default = 1).
%
% OUTPUT
% vd    = forecasting error variance decompositions of y(t+h) - y(t-1) in
%   relations to x(t), x(t+1), x(t+2), ..., x(t+h) using R2.
% vdBc  = bias-corrected FEVD.
%           R2 estimator with asymptotic distribution based simulation 
%           and bias correction in Appendix A.
% vdSe  = standard errors of vd and vdBc. See 'seIndicator' above.
% cb    = condifence bands. See above explanations for 'quantiles' and 'cbIndicator.'
%  If seIndicator is either 1 or 2, then normal approximation is used. For
%  example, if qunatile = 0.975, then vdBc + 1.96 * vdSe is reported.
%  If seIndicator is 0, corresponding quantiles from the simulated sample
%  are reported. We translate the bands according to the bias estimated to 
%  make the confidence band centered around vdBc.
% details  = A structure consists of other results.
% details.vdSeBootstrap / vdSeAVarWPW / vdSeAVarWoPW:
%           Simulation based se, 
%           asymptotic variance base se, WPW: with pre-whitening, 
%           asymptotic variance base se, WoPW: without pre-whitening
% details.preEstimates = (horizon+1 * 1) cell array. (h+1)-th element is
%  ((X'X)^-1 * (X'f), X'f/T, f'f/T) : h+1 + h+1 + 1 dimensional vector
% details.preAVar = (horizon+1 * 1) cell array. (h+1)-th element is
%  Estimated variance of ((X'X)^-1 * (X'f), X'f/T, f'f/T): (2h+3 * 2h+3)
%  matrix. If seIndicator == 0 or 1, it is based on pre-whitened series. 
%  If seIndicator == 2, it is based on the estimates without pre-whitening.


% #1. Setting up the default parameters  
if nargin < 7
    error('Not enough input arguments')
end

if nargin < 12
    boundIndicator = 1;
end
if nargin < 11
    cbIndicator = 1;
end
if nargin < 10
    seIndicator = 1;
end
if nargin < 9
    quantiles = [];
end
if nargin < 8
    NBootstrap = 2000;
end


% de-meaning the shock
x = X(:,1) - mean(X(:,1));
X(:,1) = x;


% #2. Variance decomposition and R2 method
[ vd, ~, ~, ~, ~, ...
    details.preEstimates, GOGTWPW, GOGTWoPW ] ...
    = SubR2( y, X, lagy, lagX, horizon, ylevel, trend, 1 );

if seIndicator == 2
    details.preAVar = GOGTWoPW;
else
    details.preAVar = GOGTWPW;
end

% #3. Bias-correction
% initialization
vdb = zeros(NBootstrap, horizon+1);      % (b, h+1)   = s_{h}^{(b)}
NBootstrapIdx = 1:NBootstrap;            % (1,2, ..., NBootstrap)


for h = 0:horizon
    thetah = details.preEstimates{h+1,1}; % \hat{\theta} = estimates of (X'X)^-1 * (X'f), X'f/T, f'f/T.
    GOGTh  = details.preAVar{h+1,1};      % Corresponding variance
    
    % Check validity for simulation: GOGTh should be real symmetric
    % positive definite.
    [~,chk] = cholcov(GOGTh);
    while chk ~= 0
        GOGTh = (GOGTh + GOGTh')/2;
        [VM,DM] = eig(GOGTh);
        DM = max(DM,eps);
        GOGTh = VM * DM * VM';
        [~,chk] = cholcov(GOGTh);
    end
    
    % thetahb ~ N ( thetah, GOGTh ).
    thetahb = mvnrnd(thetah, GOGTh, NBootstrap); 
    
    % f'f/T should be strictly positive.
    ffIdx = NBootstrapIdx(thetahb(:,end) <= 0);
    
    while ~isempty(ffIdx) % as long as at least one simulated fb'fb >= 0
        % Replace with new draws
        thetahb(ffIdx,:) = mvnrnd(thetah, GOGTh, length(ffIdx)); 
       
        % Check
        ffIdx = ffIdx( thetahb( ffIdx,end ) <= 0 );
    end
        
    % Estimate vdb using R2 method with the simualaetd series
    vdb(:,h+1) = sum( thetahb(:,1:h+1) .* thetahb(:,((h+2):(2*h+2))), 2 ) ./ thetahb(:,end);
    % thetahb2' * thetahb1 / thetahb3.
    
end

% #4. Bias correction, standard errors, etc.
% #4.1. Variance decomposition
if boundIndicator == 1
    % although R2 is bounded between 0 and 1, this simulated vdb might be
    % out of the unit interval. This transformation makes bias-correction
    % step more robust.
    vdb = min(max(vdb,0),1);
end

mvdb = mean(vdb,1)';                     % average of vdb
vdBc = vd - ( mvdb - vd );               % bias correction 


details.vdSeAVarWPW  = zeros(horizon,1);
details.vdSeAVarWoPW = zeros(horizon,1);

% Asymptotic variance: Delta (G^-1 Omega G'^-1) Delta' where Delta is based
%   on vdBc, not vd. What SubR2 reports is based on vd, but we believe that
%   vdBc is closer to the true value than vd. So, using it would be better.
for h = 0:horizon
    thetah = details.preEstimates{h+1,1};  % \hat{\theta} = estimates of (X'X)^-1 * (X'f), X'f/T, f'f/T.
    
    xih = 1/thetah(2*h+3)* [thetah(h+2:2*(h+1)); thetah(1:h+1); -vdBc(h+1)]';

    details.vdSeAVarWPW(h+1)  = sqrt(xih * GOGTWPW{h+1,1} * xih');
    details.vdSeAVarWoPW(h+1) = sqrt(xih * GOGTWoPW{h+1,1} * xih');
end


% #4.2. Standard error
details.vdSeBootstrap = std(vdb, 0, 1)'; % simulation based standard error 

if seIndicator == 0
    vdSe = details.vdSeBootstrap;
elseif seIndicator == 1
    vdSe = details.vdSeAVarWPW;
else
    vdSe = details.vdSeAVarWoPW;
end


% #4.3. confidence bands
if isempty(quantiles)
    cb = [];
else
    
    if cbIndicator == 0     % based on the distribution of s_h^{(b)}
        qvd = quantile(vdb,quantiles,1);  % (quantiles, horizon) in the original space
        
        cb = qvd + repmat( (vd' - mvdb' - ( mvdb' - vd' )), length(quantiles), 1 );
        % Parallely shifting the distribution of vdb's to make its new mean is
        % equal to the vdBc
       
    elseif cbIndicator == 1    % use the AVar and normal approximation with pre-whitening
        
        qvd = norminv(quantiles,0,1);
        qvd = qvd(:);
        
        cb = repmat(vdBc', length(qvd), 1) + qvd * details.vdSeAVarWPW';
        
    else % without pre-whitening
        
        qvd = norminv(quantiles,0,1);
        qvd = qvd(:);
        
        cb = repmat(vdBc', length(qvd), 1) + qvd * details.vdSeAVarWoPW';
        
    end
end




