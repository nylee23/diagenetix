function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   Loops over training examples 1 at a time, may need to increase
%   step size for larger sets. 

% Number of training examples
m = size(X, 1);

% Initialize Matrices
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% Loop over training example sets from 1 to full set
for i = 1:m

    Xtrain = X(1:i,:);
    ytrain = y(1:i);
    
    % Calculate Theta using Linear Regression on Training Set
    [theta, cost] = fit_LAMP(Xtrain,ytrain,lambda);
    
    % Calculate Training Error and CV Error. 
    [cost_train, grad_train] = lrCostFunction(theta,Xtrain,ytrain,0);
    [cost_cv, grad_cv] = lrCostFunction(theta,Xval,yval,0);
    
    % Save Costs to vectors
    error_train(i) = cost_train;
    error_val(i) = cost_cv;
end


% =========================================================================

end
