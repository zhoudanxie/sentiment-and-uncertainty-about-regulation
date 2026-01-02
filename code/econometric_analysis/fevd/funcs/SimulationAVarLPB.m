function [ vd, vdBc, vdSe, cb, details] ...
    = SimulationAVarLPB( y, X, horizon, ylevel, trend, lagy, lagX, XtIndicator, ...
    NBootstrap, dfIndicator, quantiles, seIndicator, cbIndicator, ...
    jointIndicator, XLP, lagyLP, lagXLP, XtIndicatorLP )
%The FEVD is obtained by the LP B method.
%
% By simulating (\hat{psi}_{x,h}'s, \hat{\sigma}_x^2,
% \hat{\sigma}_{v,h}^2) simultaneously from their 
% estimated asymptotic distribution, we derive the bias-corrected
% FEVD, vdBc, and the corresponding standard errors, vdSe.
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
% See Appendix B for details.
%
% INPUT:
% y(t)    = (T * 1) dependent variable,
% X(t, n) = (T * N) shock and controls. The first column is the shock. The
%  shock value will be de-meaned.
% horizon = ir and vd are derived up to the horizon-th period
% ylevel  = if 0: Delta y(t-1) and its lagged values on the RHS.
%           if 1: y(t-1) and its lagged values on the RHS.
% trend   = c                     if trend == 0,
%           c + dt                if trend == 1,
%           c + dt + et^2         if trend == 2,
%           c + dt + et^2 + ft^3  if trend == 3.
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
%   In a bivariate case, it can be just 1. (Default = [1, 0, 0, ...])
%
% NOTE: When there are many control variables or specific ordering
%   assumption is required for the purpose of identification, it is easier
%   to set lagy = 0 and include y or Delta y in X directly. Then
%   specification details can be implemented by choosing lagX and
%   XtIndicator.
%
% NBootstrap  = size of the bootstrap. We simulate the estimated model by
%   NBootstrap times to derive the bias, standard errors, and confidence
%   bands.        (Default = 2000).
% dfIndicator = Whether to adjust the degrees of freedom when building the
%   forecast error variance.
%   E[FE(t+h,t-1)^2] is estimated by
%       = \sum_{i=0}^{h} \psi_{x,i}^2 * Var(x) + Var( v(t+h,t-1) ) where
%       v(t+h,t-1) = FE(t+h,t-1) - \sum_{i=0}^{h} \psi_{x,i} * x(t+h-i).
%   When we evaluate the second term, we calculate the following.
%       \sum ( v(t+h,t-1) )^2 / denominator
%       The denominator is T_h if dfIndicator == 1 and
%       T_h - (lagy + 1) - (sum(lagX) + N) - (trend + 1) o.w. (Default = 0)
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
% jointIndicator: An indicator for joint distribution. 
%   If the indicator is zero, vd, bias, vdbc, and vdse are derived 
%   horizon by horizon. If the indicator is one, we estimate a one very 
%   large system and derive joint asymptotic variance of 
%   (\hat{s}_0, \hat{s}_1, ... , \hat{s}_horizon)'          (Default = 1).
% XLP    = ( T * NLP ) matrix. The first column is the shock. This is used
%   when estimating the impulse response coefficients. If not specified, it
%   is assumed to be the same as X. The first column of XLP should be the 
%   same as that of X.  (Default = X)
% lagyLP = Number of lagged values of y or Delta y included on the RHS of
%   the local projection regressions when estimating impulse response
%   coefficients. If not specified, it is equal to lagy. ( Defalut = lagy)
% lagXLP = Similar to lagyLP, this is lag lengthes used for XLP when
%   estimating impulse response coefficients. If not specified, it is lagX.
%   (Default = lagX)
% XtIndicatorLP = Similar to above. ( Default = XtIndicator )
%
% OUTPUT
% vd        = (1+horizon * 1) forecasting error variance decompositions
%   of y(t+h) - y(t-1) in relations to
%   x(t), x(t+1), x(t+2), ..., x(t+h) using LP.
% vdBc  = bias-corrected FEVD.
%           LP B estimator with asymptotic distribution based simulation 
%           and bias correction.
% vdSe  = standard errors for vd and/or vdBc. See 'seIndicator' above.
% cb    = condifence bands. See above explanations for 'quantiles' and 'cbIndicator.'
% details  = A structure consists of other results.
% details.ir / irSeAVar / irAVar
% ir = (1+horizon * 1) Local Projection based impulse response. It
%   is the response when the input is one, the unit shock.
% irSeAVar  = Standard error of ir using the Newey-West variance estimator
%   with the Bartlett kernel. The lag length is
%   determined by the simple rule suggested by Stock and Watson (2010):
%   round(0.75*T^(1/3)).
% irAVar    = Variance-covariance matrix for ir. This represents the joint
%   distribution of ir. 
% details.stdx = std(x);  For conversions from unit shock to one s.d. shock
% details.vdSeBootstrap / vdSeAVarWPW / vdSeAVarWoPW / 
%         vdJointVarBootstrap / vdJointVarWPW / vdJointVarWoPW :
%           vdSeBootstrap: Simulation based se, 
%           vdSeAVarWPW: asymptotic variance base se, WPW: with pre-whitening, 
%           vdSeAVarWoPW: asymptotic variance base se, WoPW: without pre-whitening
%   Remaining three items are variance of (\hat{s}_0, \dots, 
%    \hat{s}_horizon)'. Derived only when jointIndicator == 1.
%           vdJointVarBootstrap: based on vdb, i.e. simulated vd's.
%           vdJointVarWPW: asymptotic variance covariance matrix, WPW: with pre-whitening, 
%           vdJointVarWoPW: asymptotic variance covariance matrix, WoPW:
%           without pre-whitening.


% #1. Setting up the default parameters
if nargin < 7
    error('Not enough input arguments')
end

% sample size
[T, N] = size(X);

if nargin < 8
    XtIndicator = zeros(N,1);
    XtIndicator(1) = 1;
end
if nargin < 9
    NBootstrap = 2000;
end
if nargin < 10
    dfIndicator = 0;
end
if nargin < 11
    quantiles = [];
end
if nargin < 12
    seIndicator = 1;
end
if nargin < 13
    cbIndicator = 1;
end
if nargin < 14
    jointIndicator = 1;
end
if nargin < 15
    XLP = X;
end
if nargin < 16
    lagyLP = lagy;
end
if nargin < 17
    lagXLP = lagX;
end
if nargin < 18
    XtIndicatorLP = XtIndicator;
end


% de-meaning the shock
x = X(:,1) - mean(X(:,1));
X(:,1) = x;

XLP(:,1) = XLP(:,1) - mean(XLP(:,1));



% #2. Local projection, Variance decomposition, and LP B method
[ details.ir, vd, ~, details.irSeAVar, ~, varx, forecastErrorVar, details.irAVar, forecastErrorIdx, ~, q, residualLP, vFE, BetaHat, sigma2vHat, t0LP ] ...
    = SubLPB( y, X, lagy, lagX, XtIndicator, horizon, ylevel, trend, dfIndicator, 1, jointIndicator, XLP, lagyLP, lagXLP, XtIndicatorLP );

% Asymptotic distribution / Bias-correction / Asymptotic s.e.

% initialization
vdb = zeros(NBootstrap, horizon+1);      % (b, h+1)   = s_{h}^{(b)}
vdBc = zeros(horizon+1,1);

Sigmaq = q(t0LP:end,:)' * q(t0LP:end,:) / (T - t0LP + 1);  % E[q_t q_t'] used in G
Nq = size(q,2);
NBootstrapIdx = 1:NBootstrap;  % (1,2, ..., NBootstrap)

if jointIndicator == 0 % horizon by horizon, not simultaneously
    % initialization
    GOGTWPW  = cell(horizon+1,1);
    GOGTWoPW = cell(horizon+1,1);
    details.vdSeAVarWPW  = zeros(horizon+1, 1);
    details.vdSeAVarWoPW = zeros(horizon+1, 1);
    
    for h = 0:horizon
        
        % #3. Asymptotic distribution
        thetah = BetaHat(:,1:h+1);
        thetah = [ thetah(:); varx; sigma2vHat(h+1) ]; % \hat{\theta} = (estimates of Beta_0, ..., Beta_{h}, varx, sigma2v(t+h, t-1)).
        
        Gh = -blkdiag( kron(eye(h+1), Sigmaq), eye(2) );  % estimates of G
        
        % Effective sample is from tminh to tmaxh
        tminh = max( [ forecastErrorIdx(1:h+1,1); t0LP ] );
        tmaxh = min(  forecastErrorIdx(1:h+1,2)  );
        
        Th = tmaxh-tminh+1;  % effective sample size

        % the moment conditions evaluated at the parameter estimates to
        % derive its long-run variance. Each row of Zh corresponds to each
        % observation.
        Zhtemp = [];
        for htemp = 0:h
            Zhtemp = [ Zhtemp, q(tminh:tmaxh,:) .* repmat(residualLP(tminh:tmaxh,htemp+1), 1, Nq)];
        end
        Zh = [Zhtemp, x(tminh:tmaxh).^2 - varx, vFE(tminh:tmaxh, h+1).^2 - sigma2vHat(h+1)];
        
        % Omegah = AVar(Zh)
        % With pre-whitening
        % VAR set-up
        ZFull = Zh - repmat(mean(Zh), Th, 1); % demeaning
        Z_1 = ZFull(2:end,:);
        Z_0 = ZFull(1:end-1,:);
        A = (Z_0' * Z_0) \ (Z_0' * Z_1);
        A = A';
        % VAR(1) residual
        U = Z_1 - Z_0 * A';
        OmegaWPW = (eye(Nq*(h+1)+2) - A) \ lrv_nw(U, round(0.75*(Th)^(1/3)-1)) / (eye(Nq*(h+1)+2) - A');
        GOGTWPW{h+1,1} = (Gh \ OmegaWPW / Gh') / Th;
        
        % Without pre-whitening
        OmegaWoPW = lrv_nw(ZFull, round(0.75*Th^(1/3)-1));
        GOGTWoPW{h+1,1} = (Gh \ OmegaWoPW / Gh') / Th;
        
        
        if seIndicator ~=2   % w/ pre-whitening
            GOGTh =     GOGTWPW{h+1,1} ;
        else % w/o pre-whitening
            GOGTh =     GOGTWoPW{h+1,1}; 
        end
        
        
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
        
        % varx and sigma2v should be strictly positive.
        ffIdx = NBootstrapIdx(sum(thetahb(:,end-1:end) <= 0, 2) ~= 0);
        
        while ~isempty(ffIdx) % when simulated variances <= 0
            % Replace with new draws
            thetahb(ffIdx,:) = mvnrnd(thetah, GOGTh, length(ffIdx));
            
            % Check
            ffIdx = ffIdx( sum(thetahb( ffIdx,end-1:end ) <= 0, 2) ~= 0 );
        end
        
        
        
        
       % Simulation
       psixhb = thetahb(:, 1+(0:h)*Nq);
       varxhb = thetahb(:,end-1);
       varvhb = thetahb(:,end);
       
       spsixhb2 = sum(psixhb.^2, 2); % sum psixhb^2
       vdb(:,h+1) = ( spsixhb2 .* varxhb ) ./  ( spsixhb2 .* varxhb + varvhb );
       
       
       % #4. Bias-correction
       vdBc(h+1) = 2 * vd(h+1) - mean(vdb(:,h+1));
       
       % #5. Standard errors based on AVar. Bootstrap based ones would be
       % defined later using vdb at once.
       
       iota = zeros(Nq,1);
       iota(1) = 1;  % [1;0;...;0]
       
       Deltah = (1 - vdBc(h+1)) / forecastErrorVar(h+1) * ...
           [ kron( 2 * details.ir(1:h+1)' * varx, iota' ), ...
           sum( details.ir(1:h+1).^2 ), ...
           - vdBc(h+1) / (1 - vdBc(h+1)) ];
       % Required for the Delta method. See Appendix B for details.
       
       details.vdSeAVarWPW(h+1)  = sqrt(Deltah * GOGTWPW{h+1,1} * Deltah');
       
       details.vdSeAVarWoPW(h+1) = sqrt(Deltah * GOGTWoPW{h+1,1} * Deltah');
          
    end
    
    details.vdSeBootstrap = std(vdb, 0, 1)'; % Simulation based standard error 
     
    
else  % Joint distribution of vd's.
        
    % #3. Asymptotic distribution
    
    theta = BetaHat(:,1:horizon+1);
    theta = [ theta(:); varx; sigma2vHat ];
    % \hat{\theta} = (estimates of Beta_0, ..., Beta_{horizon}, varx, sigma2v(t, t-1), ... , sigma2v(t+horizon, t-1)).
    
    G = -blkdiag( kron(eye(horizon+1), Sigmaq), eye(horizon+2) );  % estimates of G
    
    % Effective sample is from tminh to tmaxh
    tmin = max( [ forecastErrorIdx(1:horizon+1,1); t0LP ] );
    tmax = min(  forecastErrorIdx(1:horizon+1,2)  );
    
    Teffective = tmax-tmin+1;  % effective sample size
    
    % the moment conditions evaluated at the parameter estimates to
    % derive its long-run variance. Each row of Z corresponds to each
    % observation.
    Ztemp = [];
    vtemp = [];
   
    for htemp = 0:horizon
        Ztemp = [ Ztemp, q(tmin:tmax,:) .* repmat(residualLP(tmin:tmax,htemp+1), 1, Nq)];
        vtemp = [ vtemp, vFE(tmin:tmax,htemp+1).^2 - sigma2vHat(htemp+1) ];
    end
    Z = [ Ztemp, x(tmin:tmax).^2 - varx, vtemp ];
    
    % Omegah = AVar(Zh)
    % With pre-whitening
    % VAR set-up
    ZFull = Z - repmat(mean(Z), Teffective, 1); % demeaning
    Z_1 = ZFull(2:end,:);
    Z_0 = ZFull(1:end-1,:);
    A = (Z_0' * Z_0) \ (Z_0' * Z_1);
    A = A';
    % VAR(1) residual
    U = Z_1 - Z_0 * A';
    OmegaWPW = (eye(Nq*(horizon+1) + horizon + 2) - A) \ lrv_nw(U, round(0.75*Teffective^(1/3)-1)) / (eye(Nq*(horizon + 1) + horizon + 2) - A');
    GOGTWPW = (G \ OmegaWPW / G') / Teffective;
    
    % Without pre-whitening
    OmegaWoPW = lrv_nw(ZFull, round(0.75*Teffective^(1/3)-1));
    GOGTWoPW = (G \ OmegaWoPW / G') / Teffective;
    
    
    if seIndicator ~=2   % w/ pre-whitening
        GOGT =     GOGTWPW ;
    else % w/o pre-whitening
        GOGT =     GOGTWoPW;
    end
    
    
    % Check validity for simulation: GOGTh should be real symmetric
    % positive definite.
    [~,chk] = cholcov(GOGT);
    while chk ~= 0
        GOGT = (GOGT + GOGT')/2;
        [VM,DM] = eig(GOGT);
        DM = max(DM,eps);
        GOGT = VM * DM * VM';
        [~,chk] = cholcov(GOGT);
    end
    
    
    % thetab ~ N ( theta, GOGT ).
    thetab = mvnrnd(theta, GOGT, NBootstrap);
    
    % varx and sigma2v should be strictly positive.
    ffIdx = NBootstrapIdx(sum(thetab(:,end-horizon:end) <= 0, 2) ~= 0);
    
    while ~isempty(ffIdx) % when simulated variances <= 0
        % Replace with new draws
        thetab(ffIdx,:) = mvnrnd(theta, GOGT, length(ffIdx));
        
        % Check
        ffIdx = ffIdx( sum(thetab( ffIdx,end-horizon:end ) <= 0 ,2) ~=0 );
    end
    
    
    
    
    % Simulation
    psixb = thetab(:, 1+(0:horizon)*Nq);
    varxb = thetab(:,end-horizon-1);
    varvb = thetab(:,end-horizon:end);
        
    for h = 0:horizon
        spsixhb2 = sum(psixb(:,1:h+1).^2, 2); % sum psixb^2 from i=0 to h
        vdb(:,h+1) = ( spsixhb2 .* varxb ) ./  ( spsixhb2 .* varxb + varvb(:,h+1) );
    end
    
           

    % #4. Bias-correction
    vdBc = 2 * vd - mean(vdb,1)';
    
    % #5. Standard errors 
    % Bootstrap / Simulation based ones
    details.vdSeBootstrap = std(vdb, 0, 1)';          % point-wise
    details.vdVarBootstrap = vdb' * vdb / NBootstrap; % Joint
    
    % Asymptotic variance
    
    iota = zeros(Nq,1);
    iota(1) = 1;  % [1;0;...;0]
    
    Delta = [];
    for h = 0:horizon
    
        Deltah = (1 - vdBc(h+1)) / forecastErrorVar(h+1) * ...
        [ kron( 2 * details.ir(1:h+1)' * varx, iota' ), ...
        zeros( 1, (horizon - h) * Nq ), ...
        sum( details.ir(1:h+1).^2 ), ...
        zeros( 1, h ), ...
        - vdBc(h+1) / (1 - vdBc(h+1)), ...
        zeros( 1, horizon-h ) ];
    % Required for the Delta method. See Appendix B for details.
    
    Delta = [ Delta; Deltah ];
    
    end
    
    details.vdAVarWPW = Delta * GOGTWPW * Delta';
    details.vdAVarWoPW = Delta * GOGTWoPW * Delta';

    details.vdSeAVarWPW  = sqrt(diag(details.vdAVarWPW));
    details.vdSeAVarWoPW = sqrt(diag(details.vdAVarWoPW));
    
    
end
    
    
    
% #5. Bias correction, standard errors, etc.
% #5.1. Impulse response
%   In this version, we don't 'correct' the biases of ir.
details.stdx = sqrt(varx);
% For conversions from unit shock to one s.d. shock

% #5.2. Variance decomposition

if seIndicator == 0
    vdSe = details.vdSeBootstrap;
elseif seIndicator == 1
    vdSe = details.vdSeAVarWPW;
else
    vdSe = details.vdSeAVarWoPW;
end



% #5.3. confidence bands
if isempty(quantiles)
    cb = [];
else
    
    if cbIndicator == 0     % based on the distribution of s_h^{(b)}
       
        mvdb = mean(vdb,1)';                     % average of vdb
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



