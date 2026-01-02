function [ y, x ] ...
    = BiVarDataGen( T, TBurnIn, dgpidx, sigmax, gy, rhop, sigmap, rhoa, sigmaa )
%Simulate the following process to generate a length T time series:
% y(t)   = psix(L)x(t) + z(t),
% z(t)   = p(t) + a(t),
% ((1-L)p(t) -gy) = rhop * ((1-L)p(t) -gy) + sigmap * wnp(t),
% a(t)   = rhoa * a(t-1) + sigmaa * wna(t),
% x(t)   ~ wn(sigmax),
% wnp(t) ~ wn(1),
% wna(t) ~ wn(1).
%
% INPUT:
% T       = Generate length T vectors
% TBurnIn = First TBurnIn observations are dropped as burn-in
% dgpidx  = Chooses among three pre-specified psix(L).
%   if 1: psix(L) = hump-shaped, MA(100).
%   if 2: psix(L) = AR(1): (1 - 0.9L)^(-1)
%   if 3: psix(L) = Integrated AR(1): (1 - L)^(-1)*(1 - 0.9L)^(-1)
% sigmax = std(x(t).
%   DEFAULT: 1 if dgpidx == 1 or 3, 3 if dgpidx == 2.
% gy     = See the above model.  DEFAULT: 0.5.
% rhop   = See the above model.
%   DEFAULT: 0.9 if dgpidx == 1 or 2, 0.1 if dgpidx == 3.
% sigmap = See the above model.
%   DEFAULT: 0.5 if dgpidx == 1 or 2, 0.2 if dgpidx == 3.
% rhoa   = See the above model.
%   DEFAULT: 0.9 if dgpidx == 1 or 3, N/A if dgpidx == 2.
% sigmaa = See the above model.
%   DEFAULT: 3 if dgpidx == 1 or 3, N/A if dgpidx == 2.
%
% OUTPUT:
% y = T * 1 vector
% x = T * 1 vector

% #1. Setting up the default parameters
if nargin < 3
    error('Not enough input arguments')
end

if nargin == 3
    if dgpidx == 1
        sigmax  = 1;
        gy      = 0.5;
        rhop    = 0.9;
        sigmap  = 0.5;
        rhoa    = 0.9;
        sigmaa  = 3;
        
    elseif dgpidx == 2
        sigmax  = 3;
        gy      = 0.5;
        rhop    = 0.9;
        sigmap  = 1.5;   
        rhoa    = 0;
        sigmaa  = 0;
        
    else
        sigmax  = 1;
        gy      = 0.5;
        rhop    = 0.5;
        sigmap  = 2;
        rhoa    = 0.9;
        sigmaa  = 3;
    end
end

if nargin > 3 && nargin < 9
    error('Provide the entire parameter values')
end



% #2. Data generation
% #2.1. dgpidx == 1
if dgpidx == 1  % Hump-shaped psix(L)
    % white noise
    wn = normrnd(0,1,T + TBurnIn + 100, 3);
    
    % xpart
    psix = chi2pdf((0:1:100), 10);
    psix = psix * 3 / max(psix) * sigmax;   % Maximum response = 3 with the unit input.
    
    psixt = [fliplr(psix), zeros(1, T + TBurnIn - 1)];  % Convolution. We need to flip the MA coefficients.
    lagpsix = lagmatrix(psixt,0:1:T + TBurnIn + 100 - 1)';
    lagpsix(isnan(lagpsix)) = 0;
    
    xpart = lagpsix(TBurnIn+1:T+TBurnIn,:) * wn(:,1); 
    % Because we will drop the first TBurnIn observations, we don't need to
    % generate them from the beginning.
    x     = sigmax * wn(101+TBurnIn:end,1);
    
    % ppart and apart
    ppart = zeros(T + TBurnIn + 1, 1);
    apart = zeros(T + TBurnIn + 1, 1);
    for idxt = 2:T + TBurnIn + 1
        ppart(idxt) = rhop * ppart(idxt-1) + sigmap * wn(idxt,2);
        apart(idxt) = rhoa * apart(idxt-1) + sigmaa * wn(idxt,3);
    end
    ppart = cumsum(ppart) + (0:1:T+TBurnIn)'*gy;
    
    y     = xpart + ppart(end-T+1:end) + apart(end-T+1:end);
    
    
elseif dgpidx == 2 % psix(L) = (1 - 0.9L)^(-1)
    % white noise
    wn = normrnd(0,1,T + TBurnIn + 1, 3);
    
    xpart = zeros(T + TBurnIn + 1, 1);
    ppart = zeros(T + TBurnIn + 1, 1);
    apart = zeros(T + TBurnIn + 1, 1);
    for idxt = 2:T + TBurnIn + 1
        xpart(idxt) = 0.9 * xpart(idxt-1) + sigmax * wn(idxt,1);
        ppart(idxt) = rhop * ppart(idxt-1) + sigmap * wn(idxt,2);
        apart(idxt) = rhoa * apart(idxt-1) + sigmaa * wn(idxt,3);
    end
    ppart = cumsum(ppart) + (0:1:T+TBurnIn)'*gy;

    y     = xpart(end-T+1:end) + ppart(end-T+1:end) + apart(end-T+1:end);
    x     = sigmax * wn(end-T+1:end,1);
    
else    % psix(L) = (1-L)^(-1)(1 - 0.9L)^(-1)
    % white noise
    wn = normrnd(0,1,T + TBurnIn + 1, 3);
    
    xpart = zeros(T + TBurnIn + 1, 1);
    ppart = zeros(T + TBurnIn + 1, 1);
    apart = zeros(T + TBurnIn + 1, 1);
    for idxt = 2:T + TBurnIn + 1
        xpart(idxt) = 0.9 * xpart(idxt-1) + sigmax * wn(idxt,1);
        ppart(idxt) = rhop * ppart(idxt-1) + sigmap * wn(idxt,2);
        apart(idxt) = rhoa * apart(idxt-1) + sigmaa * wn(idxt,3);
    end
    xpart = cumsum(xpart);
    ppart = cumsum(ppart) + (0:1:T+TBurnIn)'*gy;

    y     = xpart(end-T+1:end) + ppart(end-T+1:end) + apart(end-T+1:end);
    x     = sigmax * wn(end-T+1:end,1);
end