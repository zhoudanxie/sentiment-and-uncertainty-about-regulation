function [ir, vdFullInfo, vdMixedInfo, vdSimpleInfo] = MultVarPopulation( horizon, idxy, idxx, idxobs )
%Study population properties of a medium-sized DSGE model in
% Smets and Wouters (2007).
% The estimated parameter values in Smets and Wouters (2007) are used. 
%
% INPUT
% horizon = impulse response coefficients and FEVDs are derived up to 
%   horizon-th periods.
% idxy    = Out of seven endogenous variables, the selected one is
%   analyzed in relations to the specified exogenous variable.
%   1: y=GDP, 2: c=Consumption, 3: inve=Investment, 4: w=Wage, 
%   5: pinf=Price inflation, 6: r=Nominal interest rates, 7:
%   lab=Employment. Must be a scalar.
% idxx    = Among seven exogenous shocks, the selected ones and their
%   relations to the selected endogenous variable are studied.
%   1: a=TFP, 2: b=Credit spread, 3: g=Government expenditure, 
%   4: qs=Investment specific technology, 5: m=Monetary policy, 
%   6: spinf=price mark up, 7: sw=wage mark up
% idxobs  = Those variables are assumed to be inside the information set.
%   1: y=GDP, 2: c=Consumption, 3: inve=Investment, 4: w=Wage, 
%   5: pinf=Price inflation, 6: r=Nominal interest rates, 7: lab=Employment
%
% OUTPUT
% ir           = ( (horizon + 1) * length(idxx) ) vector / matrix.
%   For example, 
%   if idxy == 1 and idxx == 5, ir(h) 
%   = partial GDP (t+h) / partial Monetary policy shock (t) * std(MP shock)
%   If idxx is a vector, columns of ir represents impulses responses to
%   the shocks in order.
% vdFullInfo   = ( horizon * 1 ) vector. For example, let idxy == 1 and 
%   idxx == 5. Let FE(t+h,t) = GDP(t+h) - E[GDP(t+h) | Information t].
%   Then, vdFullInfo(h) = Variance of components of FE(t+h,t) explained by
%   MP shock(t+1), MP shock(t+2), ... , MP shock(t+h)   /   Var(FE(t+h,t)).
%   Information t has histories of all seven shocks realized until t.
% vdModerateInfo = ( horizon * 1 ) vector. Variables in idxy, idxobs, and
%   shocks in idxx are assumed to be observable.
%   Example of the information set:
%   idxy = 1, idxx = [1,5], idxobs = [1,5,6] -> Information set
%     = {TFP shock, MP shock, GDP, Price Inflation, Nominal Interest Rate}.
% vdSimpleInfo = ( horizon * 1 ) vector. It is defined similarly. The only
%   difference is that the information set at time t has only GDP and MP shock.



warning('off', 'all')

% #1. Data generating process: Smets and Wouters (2007)
load sw_model_results       % Solved results using dynare given the parameter estimates reported in Smets and Wouters (2007). 


% #2. Transforming the model into VMA(infinity)
% YLogDev = PSI(L) * Omega^(-1/2) *  ETA. where
% YLogDev = (y, c, inve, w, pinf, r, lab)' log deviation from the trend,
% ETA = (ea, eb, eg, eqs, em, epinf, ew)' ~ N(0, Omega). Thus,
% Str. shock = Omega^(-1/2) *  ETA ~ N(0, I).
IRF = [oo_.irfs.y_ea; oo_.irfs.c_ea; oo_.irfs.inve_ea; oo_.irfs.w_ea; oo_.irfs.pinf_ea; oo_.irfs.r_ea; oo_.irfs.lab_ea]';
IRF(:,:,2) = [oo_.irfs.y_eb; oo_.irfs.c_eb; oo_.irfs.inve_eb; oo_.irfs.w_eb; oo_.irfs.pinf_eb; oo_.irfs.r_eb; oo_.irfs.lab_eb]';
IRF(:,:,3) = [oo_.irfs.y_eg; oo_.irfs.c_eg; oo_.irfs.inve_eg; oo_.irfs.w_eg; oo_.irfs.pinf_eg; oo_.irfs.r_eg; oo_.irfs.lab_eg]';
IRF(:,:,4) = [oo_.irfs.y_eqs; oo_.irfs.c_eqs; oo_.irfs.inve_eqs; oo_.irfs.w_eqs; oo_.irfs.pinf_eqs; oo_.irfs.r_eqs; oo_.irfs.lab_eqs]';
IRF(:,:,5) = [oo_.irfs.y_em; oo_.irfs.c_em; oo_.irfs.inve_em; oo_.irfs.w_em; oo_.irfs.pinf_em; oo_.irfs.r_em; oo_.irfs.lab_em]';
IRF(:,:,6) = [oo_.irfs.y_epinf; oo_.irfs.c_epinf; oo_.irfs.inve_epinf; oo_.irfs.w_epinf; oo_.irfs.pinf_epinf; oo_.irfs.r_epinf; oo_.irfs.lab_epinf]';
IRF(:,:,7) = [oo_.irfs.y_ew; oo_.irfs.c_ew; oo_.irfs.inve_ew; oo_.irfs.w_ew; oo_.irfs.pinf_ew; oo_.irfs.r_ew; oo_.irfs.lab_ew]';

% VMA. Re-organizing
PSI = zeros(7,7,horizon+1);
% Y-idx, ETA-idx, time-idx, normalized by shock std, i.e., one unit input to PSI = one standard deviation shock to ETA.
% partial Y-idx variable(t + time-idx) / partial Eta-idx shock(t) * std(Eta-idx).
for idx = 1:horizon+1
    for idx2 = 1:7
        PSI(:,idx2,idx) =  IRF(idx,:,idx2);
    end
end

% Saving 'ir'
ir = PSI(idxy, idxx, :);
if length(idxx) > 1
ir = squeeze(ir)';
else
ir = squeeze(ir);
end

% #3. Variance decomposition, Full Information.
% #3.1. denominator
PSI2 = zeros(7,7,horizon+1);
for idx = 1:horizon+1
    PSI2(:,:,idx) = PSI(:,:,idx) * PSI(:,:,idx)'; % Var( PSI_h * Str. shock )
end

% Var( PSI_0 * Str. shock(t+h) + PSI_1 * Str. shock(t+h-1) + ... PSI_{h-1} * Str. shock(t+1) )
VD_denom = cumsum(PSI2,3);


for idx = 1:horizon+1
    VD_denom(:,:,idx) = repmat(diag(VD_denom(:,:,idx)),1,7);
end

% #3.2. numerator
VD_num = zeros(7,7,horizon+1);

for idx2 = 1:7
    PSI2_num = zeros(7,7,horizon+1);
    for idx = 1:horizon+1
        PSI2_num(:,:,idx) = PSI(:,:,idx) * diag([zeros(1,idx2-1), 1, zeros(1,7-idx2)]) * PSI(:,:,idx)';
    end
    PSI2_num = cumsum(PSI2_num,3);
    % Var( PSI_0 * Str. shock(t+h) + PSI_1 * Str. shock(t+h-1) + ...
    % PSI_{h-1} * Str. shock(t+1) ) when all the other Str. shocks are
    % fixed to be zeros.

    for idx = 1:horizon+1
        VD_num(:,idx2,idx) = diag(PSI2_num(:,:,idx));
    end
    
end

% #3.3. vdFullInfo
VD = VD_num./VD_denom;  % Y-idx, ETA-idx, time-idx

% Saving 'vdFullInfo'
vdFullInfo = VD(idxy, idxx, :);
vdFullInfo = squeeze(vdFullInfo)';
if length(idxx) > 1
vdFullInfo = vdFullInfo(1:horizon+1,:);
end


% #4. Variance decomposition, Simple information

% Dynare saves the solution as follows:
% Y: log-linearized variables.
% State equation: Y     = A * state(-1) + B * ETA,
% Y     = M_.endo_names(oo_.dr.order_var,:),  33 * 1
% A     = oo_.dr.ghx
% state = M_.endo_names(oo_.dr.state_var,:),  20 * 1
% state = Y(8:27),
% B     = oo_.dr.ghu
% ETA   = (eta_a, eta_b, eta_g, eta_qs, eta_m, eta_pinf, eta_w)',  7 * 1
% ETA   ~  iid N( 0, Omega )
% Omega = M_.Sigma_e. It is a diagonal matrix.
% Measurement equation:
% YObservable = ( Delta y, Delta c, Delta inve, Delta w, pinf, r, lab)' 
%             = [diff(Y([11, 24, 25, 27]), Y([26, 12, 33])]' 
%               + Constant trend.



% #4.1. MA(infinite) representation of Simple information set case.
% For example, suppose that idxy == 1 and idxx == 5. Then, we want the 
%   following representation of GDP(t).
%   log deviation of GDP from trend(t) = psi_{MP}(L) * MP shock(t) + psi_e(L) * e(t)
%
% We aggregate the process driven by the other six shocks into just one
%   MA(infinite) process, psi_e(L) * e(t). 
% We use a technique of the stationary Kalman Filter.
% See Hamilton (1993, Ch. 13) for the detailed explanation.

% State equation
% Y = F * Y(-1) + B * Omega_{-idxx}^(1/2) * Str. shock_{-idxx}.
%   In the above process, we intentionally dropped the component driven by
%   the idxx-th structural shock denoted by '-idxx' notation.
% Observation equation
% Obs = H' * Y. 

vdSimpleInfo = [];

for idxidxx = 1:length(idxx)
    idxxtemp = idxx(idxidxx);
    F = [zeros(33,7), oo_.dr.ghx, zeros(33,6)];
    
    B = oo_.dr.ghu;
    sigma_eta = diag(sqrt(M_.Sigma_e));
    sigma_eta(idxxtemp) = 0;     % Excluding the shock being analyzed.
    B = B * diag(sigma_eta); % B * Omega_{-idxx}^(1/2)
    
    idx_endo = [11, 24, 25, 27, 26, 12, 33];  % For example, GDP is the 11st element in Y
    
    idx = idx_endo(idxy);
    H = zeros(33,1);
    H(idx) = 1;                               % Obs = H' * Y.
    
    R = 0;          % Measurement error variance in the observation equation
    Q = B * B';     % Var ( B * Omega_{-idxx}^(1/2) * Str. shock_{-idxx} ).
    
    
    
    P = dare(F',H,Q,R); % Solution to the discrete algebraic Ricatti equation:
    % P = FPF' - FPH(H'PH+R)^(-1)H'PF' + Q
    K = F * P * H * (H' * P * H + R)^(-1); % gain matrix
    
    sigmae = sqrt(H' * P * H + R);    % e(t) ~ WN( sigmae )
    
    psie     = zeros(horizon+1, 1);
    psie(1)  = 1;
    for idxt = 2:horizon+1
        psie(idxt) = H' * F^(idxt-2) * K;
    end
    
    vdSimpleInfo = [ vdSimpleInfo, cumsum(ir(:, idxidxx).^2) ./ (cumsum(ir(:, idxidxx).^2) + cumsum(psie.^2 * sigmae^2)) ];
    % Note that 'ir' is already in response to one s.d. shock.
end





% #5. Moderate information set
%   Example of the information set:
%   idxy = 1, idxx = [1,5], idxobs = [1,5,6] -> Information set
%     = {TFP shock, MP shock, GDP, Price Inflation, Nominal Interest Rate}.

idxobs = union(idxobs, idxy);
Numbering = 1:length(idxobs);
idxyPosition = Numbering(idxobs == idxy);

F = [zeros(33,7), oo_.dr.ghx, zeros(33,6)];

B = oo_.dr.ghu;
sigma_eta = diag(sqrt(M_.Sigma_e));
sigma_eta(idxx) = 0;     % Excluding the shock being analyzed.
B = B * diag(sigma_eta); % B * Omega_{-idxx}^(1/2)

idx_endo = [11, 24, 25, 27, 26, 12, 33];  % For example, GDP is the 11st element in Y

H = zeros(33,length(idxobs));
for idxcol = 1:length(idxobs)
H(idx_endo(idxobs(idxcol)), idxcol) = 1;      % Obs = H' * Y. 
end

R = zeros(length(idxobs));          % Measurement error variance in the observation equation
Q = B * B';     % Var ( B * Omega_{-idxx}^(1/2) * Str. shock_{-idxx} ).



P = dare(F',H,Q,R); % Solution to the discrete algebraic Ricatti equation:
% P = FPF' - FPH(H'PH+R)^(-1)H'PF' + Q
K = F * P * H * (H' * P * H + R)^(-1); % gain matrix

sigmae = chol(H' * P * H + R, 'lower');    % e(t) ~ WN( sigmae )

psie         = zeros(horizon+1, length(idxobs), length(idxobs));
psie(1,:,:)  = eye(3) * sigmae;
for idxt = 2:horizon+1
    psie(idxt,:,:) = H' * F^(idxt-2) * K * sigmae;
end
% VMA(infinity) representation of the components of observables which are 
% orthogonal to the observable shocks. In the example, it is the part
% driven by the other five structural shocks except for TFP and MP.
% Because we have three observable, we would have the Wold decomposition
% based on a three-dimensional vector white noise process.

vdMixedInfo = cumsum(ir.^2) ./ repmat((sum(cumsum(ir.^2),2) + cumsum(sum(squeeze(psie(:, idxyPosition, :).^2), 2))), 1, length(idxx));
% Note that 'ir' is already in response to a one s.d. shock.
