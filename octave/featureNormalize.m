function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. 

% Initialize arrays
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% Normalize
for ii=1:size(X,2),
    mu(ii) = mean(X(:,ii));
    sigma(ii) = std(X(:,ii));
    if mu(ii) != 0 && sigma(ii) !=0,
        X_norm(:,ii) = (X(:,ii)-mu(ii))/sigma(ii);
    else
        sigma(ii) = 1;
end


% ============================================================

end
