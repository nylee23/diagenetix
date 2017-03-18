%% Powdery Mildew Logistic Regression Model

%  ------------
% 
%  This file contains code to run a machine learning logistic
%  regression algorithm to predict LAMP results from powdery mildew
%  data. 
%

%% Initialization
clear ; close all; clc

%% ==================== Load Data ====================
frei_data = load('frei_properties.txt');
laguna_data = load('laguna_upper_properties.txt');
two_rock_data = load('two_rock_properties.txt');
y_col = 23 ;

frei_y = frei_data(:,y_col);
frei_x = frei_data(:,1:22);
laguna_y = laguna_data(:,y_col);
laguna_x = laguna_data(:,1:22);
two_rock_y = two_rock_data(:,y_col);
two_rock_x = two_rock_data(:,1:22);
all_y = [frei_y; laguna_y; two_rock_y];
all_x = [frei_x; laguna_x; two_rock_x];
n_all = length(all_y);

% Add column of 1s to all_x
all_x = [ones(n_all, 1) all_x];

%% ==================== Set Options for Fit ====================
lambda = 0.1;  % For Regularization

%% ==================== Find accuracy vs. fraction data ====================
% Set values for looping
min_frac = 0.2;
frac_step = 0.05;
n_frac = (1-min_frac)/0.05;
n_iter = 50;  % how many times to run each fraction

% Empty arrays
frac_arr = zeros(n_frac,1);
acc_arr = zeros(n_frac,4);

% Loop over all fractions
for ii=1:n_frac,
    frac_arr(ii) = min_frac+(ii*frac_step);

    % Empty arrays to hold results of all iterations
    all_acc = zeros(4,n_iter);
    
    % Loop over all iterations to find average
    for jj=1:n_iter,
        
        % Randomize indices
        sel = randperm(n_all);
        sel_mod = sel(1:round(n_all*frac_arr(ii)));
        sel_test = sel(round(n_all*frac_arr(ii))+1:n_all);
        
        % Normalize features so mean = 0, stddev = 1
        [normX mu sigma] = featureNormalize(all_x(sel_mod,:));
        
        % Minimize Cost Function
        [theta, cost] = fit_LAMP(normX,all_y(sel_mod),lambda);
        
        % Test Model Accuracy on Frei
        %[frei_pred_y,frei_hyp] = predict(frei_x,mu,sigma,theta);
        %all_acc(1,jj) = mean(double(frei_pred_y == frei_y)) * 100;

        % Test Model Accuracy on Laguna
        %[laguna_pred_y,laguna_hyp] = predict(laguna_x,mu,sigma,theta);
        %all_acc(2,jj) = mean(double(laguna_pred_y == laguna_y)) * 100;

        % Test Model Accuracy on Laguna
        %[two_rock_pred_y,two_rock_hyp] = predict(two_rock_x,mu,sigma,theta);
        %all_acc(3,jj) = mean(double(two_rock_pred_y == two_rock_y)) * 100;

        % Test Model Accuracy on Overall Results
        [all_pred_y,all_hyp] = predict(all_x(sel_test,:),mu,sigma,theta);
        all_acc(4,jj) = mean(double(all_pred_y == all_y(sel_test))) * 100;
    end

    % Average accuracies 
    acc_arr(ii,:) = mean(all_acc,dim=2);
end


%% ==================== Plots ====================
% Plotting Options
col_arr = ['b', 'r', 'g', 'k'];

h = figure(1);
%for ii=1:4,
%    plot(frac_arr,acc_arr(:,ii),col_arr(ii),'Linewidth',4);
%    hold on;
%end
plot(frac_arr,acc_arr(:,4),col_arr(4),'Linewidth',4);


% Plot Annotations
title('Accuracy of LAMP Predictions (+ or -)')
legend('Frei','Laguna','Two Rock','Overall',"location",'southeast');
xlabel('Fraction of Data Used');
ylabel('Accuracy of Predictions');

% Re-size figure
W = 5; H = 4;
set(h,'papertype','<custom>')
set(h,'PaperUnits','inches');
set(h,'PaperOrientation','portrait');
set(h,'PaperSize',[H,W]);
set(h,'PaperPosition',[0,0,W,H]);
set(h,'defaultaxesposition', [0.15, 0.15, 0.75, 0.75])

% Set Fonts
set(0,'defaultaxesfontsize', 14)
%FN = findall(h,'-property','FontName');
%set(FN,'FontName','/usr/share/fonts/dejavu/DejaVuSerifCondensed.ttf');
%FS = findall(h,'-property','FontSize');
%set(FS,'FontSize',8);

% Save Figure
print(h,'-dpng','-color','model_accuracy.png');
