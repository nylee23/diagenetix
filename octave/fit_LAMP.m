function [theta, cost] = fit_LAMP(X, Y, lambda)
%% ==================== Comments ====================
%
%  fit_LAMP takes an array of features and array of answers
%  and returns the parameter array that provides the best fit. 
%
%  INPUTS:
%
%     X - Array of features of size (m x n)  [Should already
%     include X0 column]
%     Y - Column vector of answers of size (m) 
%     
%  OUTPUTS: 
%
%     THETA - Parameter Array that minimizes cost function, of size (n)  
%     COST - Cost associated with theta (single number)
%

%% ==================== Normalize and Initialize ====================
%% Properties of sample
% m = number of training examples
% n = number of parameters
[m, n] = size(X);

% Initialize Theta
initial_theta = zeros(n, 1);

%% ============= Compute Cost using Initial Theta  =============
% Compute Cost Function of initial theta

%[cost, grad] = lrCostFunction(initial_theta, X, Y, lambda); 
%fprintf('Cost at initial theta (zeros): %f\n', cost);
%fprintf('Gradient at initial theta (zeros): \n');
%fprintf(' %f \n', grad);

%% ============= Optimizing using fminunc  =============
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(lrCostFunction(t, X, Y, lambda)), initial_theta, ...
                options);

% Print theta to screen
%fprintf('Cost at theta found by fminunc: %f\n', cost);
%fprintf('theta: \n');
%fprintf(' %f \n', theta);

