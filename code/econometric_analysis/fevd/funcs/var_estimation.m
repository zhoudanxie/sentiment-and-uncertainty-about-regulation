function  [c, phi, omega, psi, k, T, residual] = var_estimation(time_series, lag, impulse_response_order, trend)
% It estimates a VAR(lag) model using the data in time_series.
% Each column of time_series corresponds to each time series starting from
% the first row.
% lag = The order of VAR model.
% impulse_response_order = It is the maximum order of impulse response.
% Trend :  0 = constant, 1 = linear, 2 = quadratic, 3 = cubic
% We have n variables, and T+lag observations. Because the first 'lag'
% number of observations are used as an initial condition, we have only T
% relationships when our total sample size is T + lag.
% k = n^2 * lag : number of estimated parameters.
% c = First column = constant term. 
%     Second column = Linear trend term. etc.
% phi = VAR coefficient matrices. It is a k by k by lag dimensional array.
% The last dimension correspondes to the lag order.
% omega = Estimated variance of the white noise term. It is not structural.
% That is, omega = var(e) = var(Rv) where v is the structural shock having
% the identity variance matrix. Thus, It is before any decomposition.
% psi = Estimated Impulse Responses. It is a k by k by
% impulse_response_order+1 dimensional array. It is the impulse response
% with respect to e, not v. Is last dimension correspondes to the lead(-1)
% of impulse response. That is, the first matrix is contemporary response.

% Written by Byoungchan Lee, 8/26/2017

if nargin < 3
    impulse_response_order = 20;        % Default order is 20.
    trend = 0;
end


if lag ~= 0             % lag >= 1
    [T,n] = size(time_series);
    T = T - lag;
    k = n^2 * lag;
    
    % Estimation
    lag_matrix = lagmatrix(time_series, 1:lag);
    Y = time_series(lag+1:end,:);
    Y = Y(:);
    
    if trend == 0 
        X = [ones(T,1), lag_matrix(lag+1:end,:)];
    elseif trend == 1
        X = [ones(T,1), lag_matrix(lag+1:end,:), (1:1:T)'];
    elseif trend == 2
        X = [ones(T,1), lag_matrix(lag+1:end,:), (1:1:T)', (1:1:T)'.^2];
    else
        X = [ones(T,1), lag_matrix(lag+1:end,:), (1:1:T)', (1:1:T)'.^2, (1:1:T)'.^3];
    end
        
    XX = kron(eye(n),X);
    phi_hat = (XX' * XX) \ (XX' * Y);
    residual = Y - XX * phi_hat;
    
    if trend == 0
        phi_hat = reshape(phi_hat, n*lag+1, n)';
    elseif trend == 1
        phi_hat = reshape(phi_hat, n*lag+2, n)';
    elseif trend == 2
        phi_hat = reshape(phi_hat, n*lag+3, n)';
    else
        phi_hat = reshape(phi_hat, n*lag+4, n)';
    end
    
    residual = reshape(residual, T, n)';  % Its t-th column is \hat{e_t}.
    
    % Assigning c and psi
    c = [phi_hat(:,1), phi_hat(:,end+1-trend:end)];

    phi_hat = phi_hat(:,2:end-trend);  % dropping trend related terms

    phi = zeros(n,n,lag);
    
    for idx = 1:lag
        phi(:,:,idx) = phi_hat(:, n * (idx-1) + 1 : n * idx );
    end
    
    % omega
    omega = zeros(n);
    for idx = 1 : T
        omega = omega + residual(:,idx) * residual(:,idx)';
    end
    omega = omega / T;
    
    % VMA calculation - Impulse Response
    companion = [phi_hat(:,1:end);eye(n * (lag - 1)), zeros(n*(lag-1), n)];        % Companion matrix
    psi = zeros(n,n,impulse_response_order+1);
    for idx = 1:impulse_response_order+1
        temporary = companion^(idx-1);
        psi(:,:,idx) = temporary(1:n,1:n);
    end
    
else                % lag = 0
    [T,n] = size(time_series);
    T = T - lag;
    k = n^2 * lag;
    
    % Estimation
    Y = time_series(lag+1:end,:);
    Y = Y(:);
    
    
    if trend == 0 
        X = [ones(T,1)];
    elseif trend == 1
        X = [ones(T,1), (1:1:T)'];
    elseif trend == 2
        X = [ones(T,1), (1:1:T)', (1:1:T)'.^2];
    else
        X = [ones(T,1), (1:1:T)', (1:1:T)'.^2, (1:1:T)'.^3];
    end
    
    XX = kron(eye(n),X);
    phi_hat = (XX' * XX) \ (XX' * Y);
    residual = Y - XX * phi_hat;
    
    if trend == 0
        phi_hat = reshape(phi_hat, n*lag+1, n)';
    elseif trend == 1
        phi_hat = reshape(phi_hat, n*lag+2, n)';
    elseif trend == 2
        phi_hat = reshape(phi_hat, n*lag+3, n)';
    else
        phi_hat = reshape(phi_hat, n*lag+4, n)';
    end
    
    residual = reshape(residual, T, n)';  % Its t-th column is \hat{e_t}.
    
    % Assigning c and psi
    c = [phi_hat(:,1), phi_hat(:,end+1-trend:end)];

    phi = [];
    
    % omega
    omega = zeros(n);
    for idx = 1 : T
        omega = omega + residual(:,idx) * residual(:,idx)';
    end
    omega = omega / T;
    
    % VMA calculation - Impulse Response
    psi = [];
end
