function [psi, monte_carlo_psi_bounds, monte_carlo_psi_se, ...
    vd, monte_carlo_vd_bounds, monte_carlo_vd_se, shock, ...
    c, phi, omega, psiReduced, k, T, residual] ...
    = var_chol(raw_data, lag, impulse_response_order, monte_carlo_replication, var_place, con_level, first_difference_index, trend, MCIndicator)
% Impulse response and variance decompositionn with Cholesky decompostion
% Standard error bands are also calculated following the parametric Monte
% Carlo method. See Yuriy's lecture note #3 for Econ 235A, UC Berkeley.
% INPUT:
% raw_data = T by N matrix of time series. Cholesky decomposition follows the order here.
% lag      = Number of lags for VAR
% impulse_response_order = the results are derived up to this order
% monte_carlo_replication = number of replications for the Monte Carlo
% simulation generating standard error bands
% var_place = Outputs are in responses to structural shocsk to the var_place th variable
% con_level = confidence level for s.e. ex. If it is 0.9, then 0.05 and
% 0.95 quantiles are derived.
% first_difference_index = those variables are understood as in first
% differences. Impulse response and variance decomposition is calculated in
% considerations for this integration. For example, if it is [2,4], then
% the second and fourth variables are considered as differenced where the
% impulse responses and variance decompositions are calculated for
% cumulated variables.
% Trend :  0 = constant, 1 = linear, 2 = quadratic, 3 =  cubic.
% MCIndicactor: if MCIndicactor == 1, s.e.'s are obtained using parameteric
% monte carlo method. If not, those variables are [].
% OUTPUT:
% psi = impulse responses to unit structural shocks, i.e. one std shock. N by impulse_response_order + 1.
% (i,t) element is response of i th variable
% to a unit structural shock to var_place th variable after t-1 periods.
% monte_carlo_psi_bands = lower and upper bounds for confidence interval.
% For example, if con_level = 0.9, then it gives 0.05 and 0.95 quantiles
% from simulated impulse responses.
% monte_carlo_psi_se    = Simulated standard errors
% vd  = variance decomposition.
% monte_carlo_vd_bounds and monte_carlo_vd_se are similar.
% shock = structural shock estimated
% c, phi, omega, psiReduced, k, T, residual: Results obtained from
% var_estimation.m
% required: var_estimation.m
% Written by Byoungchan Lee, 8/27/2017

if nargin < 7
    first_difference_index = [];
end

if nargin < 8
    trend = 0;
end

if nargin < 9
    MCIndicator = 1;
end

[c, phi, omega, psiReduced, k, T, residual] = var_estimation(raw_data, lag, impulse_response_order, trend);

number_of_variables = size(c,1);
select = zeros(number_of_variables,1);
select(var_place) = 1;
R = chol(omega, 'lower');

% Population

% Impulse response
for idx = 1:impulse_response_order+1
    psi_R(:,:,idx) = psiReduced(:,:,idx) * R;
end
psi_R(first_difference_index,:,:) = cumsum(psi_R(first_difference_index,:,:), 3);
psi = squeeze(psi_R(:,var_place,:));
% IR to a unit shock-structural shock

% Variance decomposition
vd = zeros(number_of_variables, impulse_response_order+1);
mse = omega;
variance = R(:,var_place) * R(:,var_place)';

for idx = 1:impulse_response_order
    vd(:, idx) = diag(variance).* (diag(mse).^(-1));
    mse = mse + psi_R(:,:,idx+1) * psi_R(:,:,idx+1)';
    variance = variance + psi(:,idx+1) * psi(:,idx+1)';
end
vd(:, impulse_response_order+1) = diag(variance).* (diag(mse).^(-1));

if MCIndicator == 1
    % Parametric Monte-Carlo
    % Q hat
    lag_matrix = lagmatrix(raw_data, 1:lag);
    X = [ones(T,1), lag_matrix(lag+1:end,:)];
    Q = X' * X / T;
    
    % Random Phi and Omega generating
    pi = c;
    for idx = 1:size(phi,3)
        pi = [pi, phi(:,:,idx)];
    end
    pi = pi';
    pi = pi(:);
    
    Sigma11 = kron(omega, inv(Q))/T;
    
    Dn = dupmat(number_of_variables);
    Sigma22 = 2*(Dn' * Dn) \ (Dn' * kron(omega, omega) * Dn) / (Dn' * Dn);
    Sigma22 = Sigma22 / T;
    
    
    [~,chk]=cholcov(Sigma11);
    while chk~=0
        Sigma11 = (Sigma11 + Sigma11')/2;
        [VM,DM] = eig(Sigma11);
        DM = max(DM,0);
        Sigma11 = VM * DM / VM;
        [~,chk]=cholcov(Sigma11);
    end
    
    [~,chk]=cholcov(Sigma22);
    while chk~=0
        Sigma22 = (Sigma22 + Sigma22')/2;
        [VM,DM] = eig(Sigma22);
        DM = max(DM,0);
        Sigma22 = VM * DM / VM;
        [~,chk]=cholcov(Sigma22);
    end
    
    
    monte_carlo_phi = mvnrnd(pi', Sigma11, monte_carlo_replication);
    monte_carlo_omega = mvnrnd(vech(omega)', Sigma22, monte_carlo_replication);
    % Each row corresponds each replication.
    
    % Simulating Psi and vd
    monte_carlo_impulse_response = zeros(number_of_variables, impulse_response_order+1, monte_carlo_replication);
    monte_carlo_vd = zeros(number_of_variables, impulse_response_order+1, monte_carlo_replication);
    
    for idx = 1: monte_carlo_replication
        
        
        phi_temp = monte_carlo_phi(idx,:);
        phi_temp = phi_temp';
        phi_temp = reshape(phi_temp, 1 + lag * number_of_variables, number_of_variables);
        phi_temp = phi_temp';
        
        % VMA calculation - Impulse Response
        companion = [phi_temp(:,2:end);eye(number_of_variables * (lag - 1)), zeros(number_of_variables * (lag - 1), number_of_variables)];
        % Companion matrix
        
        monte_carlo_omega_temp = monte_carlo_omega(idx,:);
        monte_carlo_omega_temp = reshape(Dn * monte_carlo_omega_temp', number_of_variables, number_of_variables);
        
        [~,chk]=cholcov(monte_carlo_omega_temp);
        while chk~=0
            monte_carlo_omega_temp = mvnrnd(vech(omega)', Sigma22, 1);
            monte_carlo_omega_temp = reshape(Dn * monte_carlo_omega_temp', number_of_variables, number_of_variables);
            
            [~,chk]=cholcov(monte_carlo_omega_temp);
        end
        
        R_temp = chol(monte_carlo_omega_temp, 'lower');
        
        
        % initialize
        psi_R_temp = zeros(number_of_variables, number_of_variables, impulse_response_order+1);
        psi_temp = zeros(number_of_variables, number_of_variables, impulse_response_order+1);
        
        % IR simulation
        for idx2 = 1:impulse_response_order+1
            temporary = companion^(idx2-1);
            psi_temp(:,:,idx2) = temporary(1:number_of_variables,1:number_of_variables);
            psi_R_temp(:,:,idx2) = psi_temp(:,:,idx2) * R_temp;
        end
        % First differences
        psi_R_temp(first_difference_index,:,:) = cumsum(psi_R_temp(first_difference_index,:,:), 3);
        psi_temp = squeeze(psi_R_temp(:,var_place,:));
        
        monte_carlo_impulse_response(:,:,idx) = psi_temp;
        
        
        % VD simulation
        mse_temp = monte_carlo_omega_temp;
        variance_temp = R_temp(:,var_place) * R_temp(:,var_place)';
        for idx2 = 1:impulse_response_order
            monte_carlo_vd(:,idx2,idx) = diag(variance_temp).* (diag(mse_temp).^(-1));
            mse_temp = mse_temp + psi_R_temp(:,:,idx2+1) * psi_R_temp(:,:,idx2+1)';
            variance_temp = variance_temp + psi_temp(:,idx2+1) * psi_temp(:,idx2+1)';
        end
        monte_carlo_vd(:,impulse_response_order+1,idx) = diag(variance_temp).* (diag(mse_temp).^(-1));

    end
    
    monte_carlo_psi_bounds = quantile(monte_carlo_impulse_response, [(1-con_level)/2, 1-(1-con_level)/2], 3);
    monte_carlo_psi_se = std(monte_carlo_impulse_response, 0, 3);
    
    monte_carlo_vd_bounds = quantile(monte_carlo_vd, [(1-con_level)/2, 1-(1-con_level)/2], 3);
    monte_carlo_vd_se = std(monte_carlo_vd, 0, 3);
    
else
    
    monte_carlo_psi_bounds = [];
    monte_carlo_psi_se = [];
    
    monte_carlo_vd_bounds = [];
    monte_carlo_vd_se = [];
end

shock = R * residual;
shock = R(var_place,var_place) * shock(var_place,:);