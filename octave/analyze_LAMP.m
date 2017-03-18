%% Powdery Mildew Logistic Regression Model

%  ------------
% 
%  This file contains code to analyze and debug the machine
%  learning code for LAMP. Will either create and plot a learning
%  curve, an error vs. regularization curve, or an error
%  vs. threshold plot. 
%

%% Initialization
clear ; close all; clc

%% ==================== Load Data ====================
frei_data = load('catalogs/frei_feat.txt');
laguna_data = load('catalogs/laguna_feat.txt');
two_rock_data = load('catalogs/two_rock_feat.txt');

frei_y = frei_data(:,end);
frei_x = frei_data(:,1:end-1);
laguna_y = laguna_data(:,end);
laguna_x = laguna_data(:,1:end-1);
two_rock_y = two_rock_data(:,end);
two_rock_x = two_rock_data(:,1:end-1);
all_y = [frei_y; laguna_y; two_rock_y];
all_x = [frei_x; laguna_x; two_rock_x];
m_all = length(all_y);

% ======= Normalize the X array ====== 
[normX mu sigma] = featureNormalize(all_x);
% Add column of 1s to normalized X array
normX = [ones(m_all,1) normX];


%% Testing
[m, n] = size(normX);
theta0 = zeros(n, 1)+1;
[cost, grad] = lrCostFunction(theta0,normX,all_y,1);

[theta, cost] = fit_LAMP(normX,all_y,0.2);
keyboard()



%  ======= Split Data into Training, CV, and Test Set ======
%% Constants
% Values of lambda to try
%lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10 20 30]';
%lambda_vec = [0.1 0.3 1 2 3 4 5 6 7 8 9 10]';
lambda_vec = 0.5:0.25:6;
%% Threshold Values 
%threshold_vec = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
threshold_vec = 0.1:0.02:0.4;
%% Regularization to use when testing threshold. Set to best fit
%% from Regularization tests.
lambda = 4;

% Number of randomizations to try
n_iter = 100;
n_lambda = length(lambda_vec);
n_thres = length(threshold_vec);

%% Variables to hold data
lambda_err_train = zeros(n_iter,n_lambda);
lambda_err_val = zeros(n_iter,n_lambda);
prec_vec_all = zeros(n_iter,n_thres);
recall_vec_all = zeros(n_iter,n_thres);
F1_vec_all = zeros(n_iter,n_thres);

% Begin Loop
for ii = 1:n_iter
    
    % Randomize indices
    sel = randperm(m_all);
    
    % First 60% of data is Training set:
    sel_train = sel(1:round(m_all*0.6));
    X_train = normX(sel_train,:);
    y_train = all_y(sel_train);
    m_train = size(X_train,1);
    
    % Next 20% is Cross Validation set:
    sel_cv = sel(round(m_all*0.6)+1:round(m_all*0.8));
    X_cv = normX(sel_cv,:);
    y_cv = all_y(sel_cv);
    
    % Everything else is Test set
    sel_test = sel(round(m_all*0.8)+1:end);
    X_test = normX(sel_test,:);
    y_test = all_y(sel_test);
    
    %  ======= Create and Plot Learning Curves ======
    %lambda = 0.1;
    %[error_train, error_val] = learningCurve(X_train, y_train, ... 
    %X_cv, y_cv, lambda); 
    
    %  ======= Find lambda that minimizes error ======
    %  ======= Lambda = 2 is approximate best value =====
    [error_train, error_val] = ...
        validationCurve(X_train, y_train, X_cv, y_cv, lambda_vec);
    lambda_err_train(ii,:) = error_train;
    lambda_err_val(ii,:) = error_val;
    
    % ======= Find threshold that minimizes F1 ========
    [F1, precision, recall] = ...
        calc_f1(X_train, y_train, ... 
                [X_cv; X_test], [y_cv; y_test], ... 
                lambda, threshold_vec); 
    prec_vec_all(ii,:) = precision;
    recall_vec_all(ii,:) = recall;
    F1_vec_all(ii,:) = F1;
end

% ====== Calculate average from different iterations =======
lambda_train_err = mean(lambda_err_train);
lambda_val_err = mean(lambda_err_val);

for jj=1:n_thres
    p = prec_vec_all(:,jj);
    prec_vec(jj) = mean(p(finite(p)));
    
    r = recall_vec_all(:,jj);
    recall_vec(jj) = mean(r(finite(r)));
    
    %f = F1_vec_all(:,jj);
    %F1_vec(jj) = mean(f(finite(f)));
end
%prec_vec = nanmean(prec_vec_all);
%recall_vec = nanmean(recall_vec_all);
%F1_vec = nanmean(F1_vec_all);
F1_vec = 2. .* (prec_vec.*recall_vec)./(prec_vec.+recall_vec.*1.);


% ====== Calculate best regularization and threshold values =======
[min_lambda, ind_lambda] = min(lambda_val_err);
best_lambda = lambda_vec(ind_lambda);

[min_thres, ind_thres] = max(F1_vec);
best_thres = threshold_vec(ind_thres);

% ====== Find parameters using best regularization and threshold values =======
[best_theta, best_cost] = fit_LAMP(X_train,y_train,best_lambda);


%  =========== PLOTS ==============
%% Learning Curve
%plot(1:m_train, error_train, 1:m_train, error_val);
%title('Learning curve for linear regression')
%legend('Train', 'Cross Validation')
%xlabel('Number of training examples')
%ylabel('Error')
%axis([0 22 0 4])

%% Error as function of lambda
%close all;
plot(lambda_vec, lambda_train_err, lambda_vec, lambda_val_err);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('Best Regularization value:\n');
fprintf( '   %f   \n',best_lambda);
fprintf('Program paused. Press enter to continue.\n');
pause;


%% Precision, Recall, and F1 vs. Threshold  
close all;
plot(threshold_vec, prec_vec, threshold_vec, recall_vec, threshold_vec, ... 
     F1_vec);
legend('Precision','Recall','F1 score');
xlabel('Threshold');
ylabel('Value');
    
fprintf('Best Threshold value:\n');
fprintf( '   %f   \n',best_thres);