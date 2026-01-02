function  [timeseriesb] = var_bootstrap(c, phi, timeseries, residual, T, TBurnIn)
% It simulates an artificial (vector) times series using an estimated VAR
% model from var_estimation.m. It draws the reduced form residuals with
% replacements.
% INPUT:
% c = First column = constant term.
%     Second column = Linear trend term. etc.
% phi = VAR coefficient matrices. It is a N by N by lag dimensional array.
% The last dimension correspondes to the lag order.
% timeseries = (T * N) times series used to estimate the model. It is
% required for the initial conditions.
% residual = Estimated reduced form residuals. (N by time)
% T = Sample size
% TBurnIn = Firt TBurnIn observations generated will be discarded.
% OUTPUT:
% timeseriesb: (T * N) matrix. time by variable.

% Written by Byoungchan Lee, 8/27/2017


% Parmaters
[ N, trend ] = size(c);
trend = trend - 1;     % 0 = constant, 1 = linear, 2 = quadratic, 3 = cubic
lagVAR = size(phi,3);
Tinit = TBurnIn + 1;
Tend  = TBurnIn + T;


% Initialization
zb = zeros(Tend, N*lagVAR); % Simulate in companion form first. z(t,:) = [Y(t)', Y(t-1)', ..., Y(t-lagVAR+1)']


% companion matrix
F = [reshape(phi, N, N * lagVAR); eye(N*(lagVAR-1)), zeros(N*(lagVAR-1),N)];


% shuffled residual. (k * Tend) matrix.
residualb = residual(:, ceil( size(residual, 2) * rand( Tend, 1 )));


% Initial condition
initialb = ceil((size(timeseries, 1) - lagVAR + 1) * rand(1));
% if initialb = 2, y(2), y(3), ..., y(2+lagVAR-1)'s are initial conditions.

z = timeseries(initialb + lagVAR - 1 : -1 : initialb, :)';
z = z(:);

zb(1,:) = z';


% data generation
% simulation using reshuffled residual and estimated phi's.
for idxtb = 2:Tend
    z = [ c * idxtb.^((0:1:trend)'); zeros(N*(lagVAR-1),1) ] ...  % trend
        + F*z + [residualb(:,idxtb); zeros(N*(lagVAR-1),1) ];     % Endogenous component + residual
    zb(idxtb,:) = z';
end


% Dropping the burn-in periods
timeseriesb = zb(Tinit:Tend,1:N);

