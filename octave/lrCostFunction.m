function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% Initialize Outputs
J = 0;
grad = zeros(size(theta));

%%% Logistic Regression 
hyp = sigmoid(X*theta);
J = 1/m * sum((-y .* log(hyp)) - ((1-y).*log(1-hyp)));
grad = 1/m * X'*(hyp-y);

%%% Add in Regularization 
reg_var = theta;
reg_var(1) = 0;  % Don't regularize Theta_0

J = J + (lambda/(2*m))*sum(reg_var.^2);
grad = grad + (lambda/m)*reg_var;

% =============================================================

grad = grad(:);

end
