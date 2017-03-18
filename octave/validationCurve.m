function [error_train, error_val] = ...
    validationCurve(X, y, Xval, yval, lambda_vec)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. 
%

% Initialize Outputs
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% Loop over lambda vector
for ii = 1:length(lambda_vec)
    
    % Select lambda
    lambda = lambda_vec(ii);
    
    % Train linear regression using lambda
    theta = fit_LAMP(X,y,lambda);
    
    % Calculate Training Errors and CV errors, with no
    % regularization
    [cost_train, grad_train] = lrCostFunction(theta,X,y,0);
    [cost_cv, grad_cv] = lrCostFunction(theta,Xval,yval,0);
    
    % Store in arrays
    error_train(ii) = cost_train;
    error_val(ii) = cost_cv;
end






% =========================================================================

end
