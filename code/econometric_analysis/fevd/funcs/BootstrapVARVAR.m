function [ vd, vdBc, vdSeBootstrap, cb, details] ...
    = BootstrapVARVAR( timeseries, horizon, xLocation, differencedIndex, yLocation, trend, lagVARSet, ICIndicator, NBootstrap, TBurnIn, quantiles )
%Estimate a VAR model with the Cholesky decomposition.
% By bootstrapping the estimated VAR model, we derive the bias-corrected
% FEVD, vdBc, and the corresponding standard errors, vdSeBootstrap.
% INPUT:
% timeseries = (T * N). Cholesky decomposition follows the order here.
%   For example, if it is a bi-variate system of x and Delta y, then
%   timeseries should be [x(2:end), diff(y)].
% horizon = ir and vd are derived up to the horizon-th period
% xLocation = ir and vd are in responses to a structural shock to
%   the xLocation-th variable. In the above example, xLocation = 1. If we
%   impose a minimum delay assumption, then it should be 2 where timeseries
%   becomes [diff(y), x(2:end)].
% differencedIndex  =  those variables are understood as in first
%   differences. Impulse responses and variance decompositions are 
%   calculated in considerations of such integration. For example, 
%   if it is [2,4], the second and fourth variables are differenced in
%   'timeseries,' where the impulse responses and variance decompositions 
%   are calculated for the cumulated variables.
% yLocation = similar to xLocation. In the bi-variate example without the
%   minimum delay assumption, it should be 2.
% trend   = c                     if trend == 0,
%           c + dt                if trend == 1,
%           c + dt + et^2         if trend == 2,
%           c + dt + et^2 + ft^3  if trend == 3.
% lagVARSet   = Set of candidate lag lengthes. The lag length is determined
%               by an information criterion. It should be either a
%               column or a row vector.
% ICIndicator = Indicator for which information criterion to use.
%               0: BIC, 1: HQIC, 2: AIC.     (Default = 1).
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%               NBootstrap times to derive biases, standard errors,
%               and confidence bands.        (Default = 2000).
% TBurnIn     = First TBurnIn observations are dropped as burn-in in 
%               bootstrap samples.
% qunatiles   = save the corresponding quantiles in cb. For example, if
%               quantiles = [0.05, 0.95], then cb will be a matrix whose
%               size is (2 * horizon) and represents 90% confidence
%               bands based on bootstrap.      (Default = []).
%
% OUTPUT:
% vd       = forecasting error variance decompositions of y(t+h)-y(t-1) in
%               relations to x(t),x(t+1), x(t+2), ..., x(t+h).
% vdBc     = bias-corrected FEVD.
%               VAR based estimator with VAR based bootstrap.
% vdSeBootstrap = Bootstrap based standard errors for vd and vdBc.
% cb       = condifence bands. See above explanations for 'quantiles.'
% details  = A structure consists of other results.
% details.ir / irBc / irSeBootstrap:
%           VAR based impulse response, bias-corrected impulse response,
%           and bootstrap based se.
%           Input: one std shock to x
% details.lagVAR = Chosen lag length for the VAR model by an IC.


% #1. Setting up the default parameters
if nargin < 6
    error('Not enough input arguments')
end

if nargin < 10
    quantiles = [];
end
if nargin < 9
    TBurnIn = 100;
end
if nargin < 8
    NBootstrap = 2000;
end
if nargin < 7
    ICIndicator = 1;
end

Ty = size(timeseries,1);   % length of the time series


% #2. Information criterion and lagVAR
if isscalar(lagVARSet)
    % If there's only one candidate, then that is lagVAR.
    lagVAR = lagVARSet;
    
else
    % When there are multiple candidates
    IC = zeros(size(lagVARSet));
    
    lagVARMax = max(lagVARSet);   % the largest candidate.
    T = Ty - lagVARMax;
    % For fair comparisons, we impose the same effective sample sizes for different lagVAR.
    
    for idxIC = 1: length(IC)
        lagVARtemp = lagVARSet(idxIC); % a candidate lag length
        [~, ~, omega, ~, k] = var_estimation(timeseries(lagVARMax - lagVARtemp + 1:end, :), lagVARtemp, 0, trend);
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


% #3. Estimation: ir and vd using VAR
[ir, ~, ~, ...
    vd, ~, ~, ~, ...
    c, phi, ~, ~, ~, T, residual] ...
    = var_chol(timeseries, lagVAR, horizon, NBootstrap, xLocation, 0, differencedIndex, trend, 0);
ir = ir(yLocation,:)';
vd = vd(yLocation,:)';

% #4. Bootstrap

% initialization
irb = zeros(NBootstrap, horizon + 1, length(yLocation));  % (b, h+1, y) = \psi_{x,h}^{(b)} for yLocation(y)-th dependent variable
vdb = zeros(NBootstrap, horizon + 1, length(yLocation));  % (b, h+1, y) = s_{h}^{(b)} for yLocation(y)-th dependent variable


for idxb = 1:NBootstrap
    
    % #4.1. Data generation: Bootstrap using the estimated VAR
    [timeseriesb] = var_bootstrap(c, phi, timeseries, residual, T+lagVAR, TBurnIn);
    % T = length(timeseries) - lagVAR. So, T + lagVAR = length(timeseries)
    
    % #4.2. Estimate a VAR model using simualated series
    [irbb, ~, ~, vdbb] ...
        = var_chol(timeseriesb, lagVAR, horizon, NBootstrap, xLocation, 0, differencedIndex, trend, 0);
    irbb = irbb(yLocation,:)';
    vdbb = vdbb(yLocation,:)';
    
    irb(idxb,:,:) = irbb;
    vdb(idxb,:,:) = vdbb;
    
end

% #5. Bias correction, standard errors, etc.
% #5.1. Impulse response
details.ir = ir;
if isscalar(yLocation)  % single dependant variable
    details.irBc = 2 * ir - squeeze(mean(irb,1)');    % bias correction
    details.irSeBootstrap = squeeze(std(irb, 0, 1))'; % standard error
    
else                    % multiple dependant variable
    details.irBc = 2 * ir - squeeze(mean(irb,1));    % bias correction
    details.irSeBootstrap = squeeze(std(irb, 0, 1)); % standard error
    
end


% # 5.2. Variance decomposition
mvdb = squeeze(mean(vdb,1));   % average of vdb
if isscalar(yLocation)         % single dependant variable
    vdBc = 2 * vd - mvdb';     % bias correction
    vdSeBootstrap = squeeze(std(vdb, 0, 1))'; % standard error
    
else                           % multiple dependant variable
    vdBc = 2 * vd - mvdb;      % bias correction
    vdSeBootstrap = squeeze(std(vdb, 0, 1)); % standard error
    
end



% # 5.3. lagVAR
details.lagVAR = lagVAR;

% # 5.4. confidence bands
if isempty(quantiles)
    cb = [];
else
    qvd = quantile(vdb,quantiles,1);  % (quantiles, horizon, y)
    
    cb = qvd;
    
    if ismatrix(cb)   % one endogenous variable
        for idx = 1: length(quantiles)
            cb(idx,:) = squeeze(cb(idx,:)) + 2 * (vd - mvdb')';
        end
        
    else              % multiple endogenous variable
        for idx = 1: length(quantiles)
            cb(idx,:,:) = squeeze(cb(idx,:,:)) + 2 * (vd - mvdb);
        end
    end
        % Parallely shifting the distribution of vdb's to make its new mean is
        % equal to the vdBc
    
end

end

