function [beta, variance, residual] = regress_dk(y, X, iidx, tidx, fe_flag, L, PW_flag, ase_flag, extraOutputIndicator)
% Driscoll and Kraay (1998) regression for unbalanced panel data
% following Hoechle (2017).
% Input
% y       = NT * 1: vectorized dependent variable.
% X       = NT * b: b number of parameters, total NT observations
% iidx    = NT * 1: entity index.  i = 1,2, ..., N(t)
% tidx    = NT * 1: time index.    t = 1,2, ..., T. NT = sum N(t).
% e.g. iidx(1) = 2, tidx(1) = 1-> y(1) = y(i=2, t=1).
% fe_flag = 0: no fixed effect
%           1: entity fixed effect
%           2: time fixed effect
%           3: both entity and time fixed effect
%           default = 1.
% L       = number of lags used for the Newey-West type variance estimator
%         = -1: floor(4*(T/100)^(2/9))    Hoechle (2017) for xtscc in STATA
%         = -2: round(0.75*T^(1/3))-1     Stock and Watson (2010).
%           default = -1.
% PW_flag = 1: pre-whitening using VAR(1)
%         = 0: No pre-whitening
%           default = 0.
% ase_flag= 1: use asymptotic variance
%         = 0: Small sample adjustment with a factor of
%           (T/(T-1)) * (NT - 1)/(NT - # of parameters)
%           default = 0.
% extraOutputIndicator = 1: variance is calculated. (default = 1)
%                        0: variance = [];
% Output
% beta    = Estimated coefficients
% variance= Estimated Var-Cov matrix for beta. Not normalized by sqrt N,
%           sqrt T, etc. That is, beta~N(beta_0, variance) asymptotically.
% residual= regression residual.

% required: lrv_nw.m
% Written by Byoungchan Lee, 7/1/2017

if nargin < 5
    fe_flag = 1;
end
if nargin < 6
    L = -1;
end
if nargin < 7
    PW_flag = 0;
end
if nargin < 8
    ase_flag = 0;
end
if nargin < 9
    extraOutputIndicator = 1;
end

% Finding averages for the fixed effects
Tlist = unique(tidx);  % Set of t
T = length(Tlist);     % Total number of t's.
Nlist = unique(iidx);
N = length(Nlist);

[NT,b] = size(X);
NFE = zeros(N,b+1);
for idx = 1:N
    NFE(idx,:) = mean([y(iidx == Nlist(idx)), X(iidx == Nlist(idx),:)],1);
    % [y(i,.), X(i,.)] where . denotes for the average across time horizon
end
TFE = zeros(T,b+1);
for idx = 1:T
    TFE(idx,:) = mean([y(tidx == Tlist(idx)), X(tidx == Tlist(idx),:)],1);
    % [y(.,t), X(.,t)] where . denotes for the average across entities
end
NTFE = [mean(y,1), mean(X,1)];  % overall average

if L == -1
    L = floor(4*(T/100)^(2/9));
elseif L == -2
    L = round(0.75*T^(1/3))-1;
end

% Adjusting y and X for the fixed effects. (de-meaning)
if fe_flag == 1       % entity FE
    for idx = 1:N
        y(iidx == Nlist(idx))   = y(iidx == Nlist(idx))   - NFE(idx,1);
        X(iidx == Nlist(idx),:) = X(iidx == Nlist(idx),:) - repmat(NFE(idx,2:end), sum(iidx == Nlist(idx)), 1);
    end
elseif fe_flag == 2   % time FE
    for idx = 1:T
        y(tidx == Tlist(idx))   = y(tidx == Tlist(idx))   - TFE(idx,1);
        X(tidx == Tlist(idx),:) = X(tidx == Tlist(idx),:) - repmat(TFE(idx,2:end), sum(tidx == Tlist(idx)), 1);
    end
elseif fe_flag == 3   % both FE
    for idx = 1:N
        y(iidx == Nlist(idx))   = y(iidx == Nlist(idx))   - NFE(idx,1);
        X(iidx == Nlist(idx),:) = X(iidx == Nlist(idx),:) - repmat(NFE(idx,2:end), sum(iidx == Nlist(idx)), 1);
    end
    for idx = 1:T
        y(tidx == Tlist(idx))   = y(tidx == Tlist(idx))   - TFE(idx,1);
        X(tidx == Tlist(idx),:) = X(tidx == Tlist(idx),:) - repmat(TFE(idx,2:end), sum(tidx == Tlist(idx)), 1);
    end
    y = y + NTFE(1);
    X = X + repmat(NTFE(2:end),NT,1);
end

% regression
beta     = (X'*X) \ (X'*y);
residual = y - X*beta;

% Var-Cov
if extraOutputIndicator == 1
    Xu       = X.*repmat(residual,1,b);    % x(i,t)*r(i,t)
    Xucluster= nan(T,b);
    for idx  = 1:T
        %     Xucluster(idx,:) = sum(Xu(tidx == Tlist(idx),:));
        Xucluster(idx,:) = mean(Xu(tidx == Tlist(idx),:));
        % h_t(beta) = sum_i h_it(beta). Clustering by time.
    end
    
    if PW_flag == 0   % Without Pre-whitening
        lrv = lrv_nw(Xucluster, L);
    else            % With Pre-Whitening
        % VAR set-up
        Z_full = Xucluster - repmat(mean(Xucluster),T,1);
        Z_1 = Z_full(2:end,:);
        Z_0 = Z_full(1:end-1,:);
        A = (Z_0' * Z_0) \ (Z_0' * Z_1);
        A = A';
        % VAR(1) residual
        U = Z_1 - Z_0 * A';
        lrv = lrv_nw(U, L);
        lrv = (eye(size(A)) - A) \ lrv / (eye(size(A)) - A');
    end
    
    variance = (X' * X / NT) \ (lrv/T) / (X' * X / NT);
    
    if ase_flag == 0
        variance = (T / (T-1)) * ((NT-1) / (NT - b)) * variance;
    end
    
else
    variance = [];
end

end