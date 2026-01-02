function [ psix, vdx, sigmax, gy, rhop, sigmap, rhoa, sigmaa ] ...
    = BiVarPopulation( dgpidx, horizon, sigmax, gy, rhop, sigmap, rhoa, sigmaa, figureOn )
%Examine the population properties of the following process:
% y(t)   = psix(L)x(t) + z(t),
% z(t)   = p(t) + a(t),
% ((1-L)p(t) -gy) = rhop * ((1-L)p(t) -gy) + sigmap * wnp(t),
% a(t)   = rhoa * a(t-1) + sigmaa * wna(t),
% x(t)   ~ wn(sigmax),
% wnp(t) ~ wn(1),
% wna(t) ~ wn(1).
%
% INPUT:
% dgpidx = Chooses among three pre-specified psix(L).
%   if 1: psix(L) = hump-shaped, MA(100).
%   if 2: psix(L) = AR(1): (1 - 0.9L)^(-1)
%   if 3: psix(L) = Integrated AR(1): (1 - L)^(-1)*(1 - 0.9L)^(-1)
% horizon    = psix and vd will be returned up to (horizon)-th periods.
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
% figureOn: return a figure if 1, do nothing o.w. DEFALUT: 0.
%
% OUTPUT:
% psix : [psix_0, psix_1, \dots, psix_horizon]' * sigmax
% vdx  : [s_0, s_1, s_h, \dots, s_horizon]' where s_i is the forecast
% error variance decomposition at the i-th horizon.

% #1. Setting up the default parameters
if nargin == 1
    error('Not enough input arguments')
end

if nargin == 2
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

if nargin > 2 && nargin < 8
    error('Provide the entire parameter values')
end

if nargin < 9
    figureOn = 0;
end



% #2. Finding the population impulse responses and variance decomposition
% with the full information: psix, psip, psia, variance share of x, p, and
% a, respectively.

x = 0:1:horizon;
x = x';

% #2.1. Impulse responses to the one standard deviation shock
% #2.1.1 psix
if dgpidx == 1              % DGP1
    psix = chi2pdf(x, 10);
    psix = psix * 3 / max(psix) * sigmax;   % Maximum response = 3 with the unit input. Normalized by the std.
    
elseif dgpidx == 2          % DGP2
    psix = 0.9.^x * sigmax;
    
else                        % DGP3
    psix = 0.9.^x;
    psix = cumsum(psix) * sigmax;

end

% #2.1.2. psip
psip = rhop.^x;
psip = cumsum(psip) * sigmap;

% #2.1.3. psia
psia = rhoa.^x * sigmaa;


% #2.2. Variance decomposition
% Population variance decomposition with the full information, i.e., not
% only y and x, but also either p or a is also observable.
varx = cumsum(psix.^2);
varp = cumsum(psip.^2);
vara = cumsum(psia.^2);

vdFullInfox = varx./ (varx + varp + vara);
vdFullInfop = varp./ (varx + varp + vara);
vdFullInfoa = vara./ (varx + varp + vara);


% #3. Identifiable representation
% #3.1. (1-L)z(t) = psie(L)e(t). That is, 
% MA(infinite) representation of Delta p + Delta a.
% See Hamilton (1993, Ch. 13) for the detailed explanation.

F = [rhop, 0, 0;
    0, rhoa, -sigmaa;
    0, 0, 0];
B = [sigmap, 0;
    0, sigmaa;
    0, 1];
H = [1;1;0];
R = 0;
Q = B * B';

P = dare(F',H,Q,R); % Solution to the discrete algebraic Ricatti equation:
% P = FPF' - FPH(H'PH+R)^(-1)H'PF' + Q
K = F * P * H * (H' * P * H + R)^(-1); % gain matrix

sigmae = sqrt(H' * P * H + R);

psie = zeros(horizon+1,1);
psie(1) = 1;
for idxl = 1:horizon
   psie(idxl+1) = H' * F^(idxl-1) * K; 
end

psie = cumsum(psie) * sigmae; % one standard deviation shock

% #3.2.Identifiable variance decomposition when only y and x are
% observable.
vare = cumsum(psie.^2);
vdx = varx./ (varx + vare);    % s_0, ..., s_horizon
vde = vare./ (varx + vare);

% #4. Figure

if figureOn == 1
figure,
subplot(1,4,1), plot(0:1:horizon, psix), title('IR: x, one sd shock', 'fontsize', 12)
subplot(1,4,2), plot(0:1:horizon, psip), title('IR: p, one sd shock', 'fontsize', 12)
subplot(1,4,3), plot(0:1:horizon, psia), title('IR: a, one sd shock', 'fontsize', 12)
subplot(1,4,4), plot(0:1:horizon, psie), title('IR: e, one sd shock', 'fontsize', 12)

figure,
subplot(1,2,1), plot(1:1:horizon, vdFullInfox(1:horizon), 'b-', ...
    1:1:horizon, vdFullInfox(1:horizon)+vdFullInfop(1:horizon), 'r:', ...
    1:1:horizon, vdFullInfox(1:horizon)+vdFullInfop(1:horizon)+vdFullInfoa(1:horizon), 'g-.', ...
    'linewidth', 2), title('VD: x, p, and a', 'fontsize', 12)
subplot(1,2,2), plot(1:1:horizon, vdx(1:horizon), 'b-', ...
    1:1:horizon, vdx(1:horizon)+vde(1:horizon), 'g-.', ...
    'linewidth', 2), title('VD: x, e', 'fontsize', 12)
end

end
