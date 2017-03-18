function [F1_vec, Prec_vec, Recall_vec] = ...
    calc_f1(X, y, Xval, yval, lambda, threshold_vec)

%% ==================== Comments ====================
%
%  CALC_F1 takes a training set, a cross validation set, and a
%  vector of thresholds to calculate the threshold that provides
%  the largest F1 score.  Can also plot precision vs. recall as a
%  function of threshold. 
%
%  INPUTS:
%
%     X - Training set given as array of features of size (m x n)
%     [Should already include X0 column]
%     
%     y - Column vector of training set answers of size (m) 
%
%     Xval - Cross validation set given as array of features of
%     size (m_val, n) [Should already include X0 column]
%     
%     yval - Cross validation set answers, given as column vector
%     of size (m_val)
%
%     Lambda - Regularization parameter to use
%
%     threshold_vec - vector of thresholds to try (ranging between
%     0 and 1), of length N_THRESHOLD 
%     
%  OUTPUTS: 
%
%     F1_vec - vector of F1 values associated with threshold_vec,
%     of length N_THRESHOLD
%   
%     Prec_vec - vector of precision values associated with
%     threshold_vec, of length N_THRESHOLD
%
%     Recall_vec - vector of recall values associated with
%     threshold_vec, of length N_THRESHOLD


% Initialize Outputs
F1_vec = zeros(length(threshold_vec), 1);
Prec_vec = zeros(length(threshold_vec),1);
Recall_vec = zeros(length(threshold_vec),1);

% Train linear regression using lambda
theta = fit_LAMP(X,y,lambda);

% Find size of cross validation set
m_val = size(Xval,1);

% Loop over threshold vector
for ii = 1:length(threshold_vec)
    
    % Find hypothesis by applying theta to cross validation set 
    hyp = sigmoid(Xval*theta);
    
    % Figure out which hypothesis predicts a positive:
    pred_y_val = zeros(m_val,1);
    pred_y_val(hyp >= threshold_vec(ii)) = 1;

    % Find number of true positives, false positives, and false
    % negatives 
    true_pos = sum(pred_y_val==1 & yval==1);
    false_pos = sum(pred_y_val==1 & yval==0);
    false_neg = sum(pred_y_val==0 & yval==1);
    
    % Calculate Precision & recall
    Prec_vec(ii) = true_pos/(true_pos+false_pos*1.);
    Recall_vec(ii) = true_pos/(true_pos+false_neg*1.);
    
end

% Calculate F1 score
F1_vec = 2. .* (Prec_vec.*Recall_vec)./(Prec_vec.+Recall_vec.*1.);



%
