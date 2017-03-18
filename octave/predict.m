function [pred_y, hyp] = predict(X, mu, sigma, theta)
%PREDICT(X,mu,sigma) generates the hypothesis of a set of data
%given the mean/standard deviation for normalization and the chosen
%theta matrix. 

% Find Dimensions of Input Array
[m, n] = size(X);

% Normalize Input Array
mu_arr = ones(m,1)*mu;
sig_arr = ones(m,1)*sigma;
norm_x = (X-mu_arr)./sig_arr;

% Add ones
full_x = [ones(m, 1) norm_x];

% Calculate Hypothesis
hyp = sigmoid(full_x*theta);
pred_y = zeros(m,1);
pred_y(hyp >= 0.5) = 1;


