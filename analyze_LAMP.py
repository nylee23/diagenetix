#! /usr/bin/env python
# 
# Program: ANALYZE_LAMP
#
# Author: Nick Lee
#
# Usage: ./analyze_LAMP.py 
#
# Description: Perform machine learning algorithms on full powdery mildew dataset. Can use logistic regression or linear regression,
#              depending on what data is available. Can be run on our real datasets or modeled datasets. This set of functions will
#              choose the regularization and threshold values to optimize the algorithm and then test the precision, recall, and F1 values
#              to determine how well the algorithm is performing. 
#
# Comments:
#    Always include a non-zero regularization parameter. For some reason, if no regularization is included, fmin_bfgs can fail to converge.
#   
#
# Revision History:
#    Date        Vers.    Author      Description
#    1/2/15    1.0a0    Nick Lee    First checked in
#    2/2/15    2.0a0    Nick Lee    Added logistic regression
#
# To Do:
#    
#

# Import Libraries
import numpy as np
import pylab as pl
import scipy.optimize as op
import math
import pdb
import os
import sys
import getopt
import string

# Constants - all prefixed with a lower-case 'k'
k_n_iter = 40    # Number of iterations to randomly determine training, cv, and test set
k_n_grid_napa = 90 # Number of cells in Napa data
#k_n_days_napa = 32 # Number of days in Napa Data

# Globals - all prefixed with a lower-case 'g'
g_samplenames = ['frei','two_rock','laguna','napa']
#g_regularization = np.linspace(0.1,10,21)
g_regularization = np.logspace(-2,1,num=20)
g_threshold = np.linspace(0.1,0.4,21)
g_wind = False
g_windtype = 'uv'
g_frac_data = 0.6   # Fraction of October data to use when training algorithm
g_recheck = 5
g_year =  2015  # 2014

####################
#### Functions #####
####################
'''
Function: MAIN
Interpret any command line arguments
Inputs:
   ARGV - Vector of command line arguments from sys. Generally this should just be sys.argv[1:] (ignoring first term that is name of program)
Outputs:
   None
Effects:
   Will set global variables G_WIND and G_WINDTYPE to determine if we are using a wind model, and what type of wind model we're using (uv or polar)
'''

def main(argv):
    # Try to interpret command line arguments
    try:
        opts, args = getopt.getopt(argv, 'hw:t:f:r:', ['help','wind=','type=','frac=','recheck='])
    except getopt.GetoptError:
        print 'analyze_LAMP.py -w <WIND> -t <WINDTYPE> -f <FRACTION> -r <RECHECK>'
        print 'WIND variable is Boolean'
        print 'WINDTYPE variable can be uv or polar'
        print 'FRACTION variable is float between 0 and 1, and designates what fraction of data to train algorithm on'
        print 'RECHECK variable is integer between 1 and 99 and sets how many days between successive measurements of concentration'
        sys.exit(2)

    # Loop through arguments
    for opt, arg in opts:
        if opt in ('-h','--help'):
            print 'analyze_LAMP.py -w <WIND> -t <WINDTYPE> -f <FRACTION> -r <RECHECK>'
            print 'WIND variable is Boolean'
            print 'WINDTYPE variable can be uv or polar'
            print 'FRACTION variable is float between 0 and 1, and designates what fraction of data to train algorithm on'
            print 'RECHECK variable is integer between 1 and 99 and sets how many days between successive measurements of concentration'
            sys.exit()
        elif opt in ('-w','--wind'):
            global g_wind
            if arg == 'False': g_wind = False
        elif opt in ('-t','--type'):
            global g_windtype
            g_windtype = arg
        elif opt in ('-f','--frac'):
            global g_frac_data
            g_frac_data = np.float(arg)
        elif opt in ('-r','--recheck'):
            global g_recheck
            g_recheck = np.int(arg)

#####################
'''
Function: GET_FEAT
Read the features and answers from a text table. 
Inputs:
   SAMPLE_NAME - Name of sample to read in. Text table should have filename SAMPLE_NAME_FEAT.TXT.
                 Last column of table will be the answers, while all other columns are input features
   WIND - Set keyword to "yes" to use 2-D grid wind models. Default is set to "no".
Outputs:
   X - [M x N] matrix of features 
   y - [M x 1] vector of answers corresponding to each training set
'''
def get_feat(sample_name,month='Oct',wind_cat=True,year=2014):

    # Check for filename in catalogs directory first. If it doesn't exist, specify the full path of the tables
    if g_wind==False or wind==False:
        if year==2015:
            fname = 'catalogs/'+sample_name+'_feat2015.txt'
        else:
            fname = 'catalogs/'+sample_name+'_feat.txt'
    else: fname = 'catalogs/'+sample_name+'_feat_wind_'+month+'.txt'

    tab = np.genfromtxt(fname)
    X = tab[:,0:-1]
    y = tab[:,-1]

    return X, y


#####################
## Wrapper to combine the features and answers from all samples and put them together
def get_all_feat(year=2014):
    
    # Loop over all sample names (except Napa!)
    for ii, name in enumerate(g_samplenames[:-1]):
        X, y = get_feat(name,year=year)
        if ii == 0:
            X_all = X
            y_all = y
        else:
            X_all = np.concatenate((X_all,X))
            y_all = np.concatenate((y_all,y))
            
    return X_all, y_all


#####################
'''
Function: featureNormalize
Normalize each set of features, such that the mean of each feature is 0 and the standard deviation is 1.
Also outputs the original mean and standard deviation to be able to transform between normalized and un-normalized features.

Inputs:
   X - [M x N] matrix of unnormalized features
   MU - (Optional) Length [N] vector of means to force the features to normalize to (RECALCULATE must be set to False)
   SIGMA - (Optional) Length [N] vector of standard deviations to normalize to (RECALCULATE must be set to False)
   RECALCULATE - Set to TRUE if you want to recalculate the mean and sigma based on the input X
Outputs:
   NORM_X - [M x N] matrix of normalized features
   MU - [1 x N] vector of mean values of each feature
   SIGMA - [1 x N] vector of standard deviation values of each feature
'''
def featureNormalize(X,mu=0,sigma=0,recalculate=True):

    # Initialize arrays
    norm_x = X * 0.
    
    # Calculate Mean & Standard deviation
    if recalculate == True:
        mu = np.mean(X,axis=0)
        sigma = np.std(X,axis=0)

    # Only normalize features that aren't all 0s 
    ind_nonzero = np.all([mu != 0, sigma !=0],axis=0)
    norm_x[:,ind_nonzero] = (X[:,ind_nonzero]-mu[ind_nonzero])/sigma[ind_nonzero]
    
    return norm_x, mu, sigma

#####################
## Sigmoid Function
def sigmoid(z):
    g = 1./(1+np.exp(-z))
    return g

#####################
'''
Function: logCost
Calculate the logistic regression cost function given a set of parameters (theta), features, answers, and a regularization parameter. 
Inputs:
   THETA - Set of parameters (includes X0 parameters). Vector of size n
   X - Set of features (includes X0 features). Matrix of size [m x n]
   Y - Set of answers. Vector of size [m]
   REG - Regularization parameter, single value
Outputs:
   COST - Cost function from logistic regression
'''
def logCost(theta,X,y,reg):

    ## Make sure theta, X, and y are matrices of correct dimensions:
    m, n = np.shape(X)
    y = y.reshape((m,1))
    theta = theta.reshape((n,1))

    # Calculate Hypothesis & Cost Function
    hyp = sigmoid(np.dot(X,theta))
    J = 1./m * np.sum(np.multiply(-y,np.log(hyp)) - np.multiply(1-y,np.log(1-hyp)))

    ## Add regularization
    reg_var = np.array(theta)
    reg_var[0] = 0 # Don't regularize Theta_0
    J+= reg/(2.*m)*np.sum(np.square(reg_var))
    
    return J

#####################
'''
Function: logGrad
Calculate the logistic regression gradients given a set of parameters (theta), features, answers, and a regularization parameter. 
Inputs:
   THETA - Set of parameters (includes X0 parameters). Vector of size n
   X - Set of features (includes X0 features). Matrix of size [m x n]
   Y - Set of answers. Vector of size [m]
   REG - Regularization parameter, single value
Outputs:
   GRAD - Vector of partial derivatives of J with respect to theta_i. Vector of length [n]
'''
def logGrad(theta,X,y,reg):
    
    ## Make sure theta, X, and y are matrices of correct dimensions:
    m, n = np.shape(X)
    y = y.reshape((m,1))
    theta = theta.reshape((n,1))

    # Calculate Hypothesis & gradient
    hyp = sigmoid(np.dot(X,theta))
    grad = np.dot(X.T,hyp-y)/float(m)

    ## Add regularization
    reg_var = np.array(theta)
    reg_var[0] = 0 # Don't regularize Theta_0
    grad+= float(reg)/m*reg_var
    
    return grad.flatten()

#####################
'''
Function: linCost
Calculate the linear regression cost function given a set of parameters (theta), features, answers, and a regularization parameter. 
Inputs:
   THETA - Set of parameters (includes X0 parameters). Vector of size n
   X - Set of features (includes X0 features). Matrix of size [m x n]
   Y - Set of answers. Vector of size [m]
   REG - Regularization parameter, single value
Outputs:
   COST - Cost function from linear regression
'''
def linCost(theta,X,y,reg):

    ## Make sure theta, X, and y are matrices of correct dimensions:
    m, n = np.shape(X)
    y = y.reshape((m,1))
    theta = theta.reshape((n,1))

    # Calculate Hypothesis & Cost Function
    hyp = np.dot(X,theta)
    J = 1./(2*m) * np.sum(np.square(hyp-y))

    ## Add regularization
    reg_var = np.array(theta)
    reg_var[0] = 0 # Don't regularize Theta_0
    J+= reg/(2.*m)*np.sum(np.square(reg_var))
    
    return J

#####################
'''
Function: linGrad
Calculate the linear regression gradients given a set of parameters (theta), features, answers, and a regularization parameter. 
Inputs:
   THETA - Set of parameters (includes X0 parameters). Vector of size n
   X - Set of features (includes X0 features). Matrix of size [m x n]
   Y - Set of answers. Vector of size [m]
   REG - Regularization parameter, single value
Outputs:
   GRAD - Vector of partial derivatives of J with respect to theta_i. Vector of length [n]
'''
def linGrad(theta,X,y,reg):
    
    ## Make sure theta, X, and y are matrices of correct dimensions:
    m, n = np.shape(X)
    y = y.reshape((m,1))
    theta = theta.reshape((n,1))

    # Calculate Hypothesis & gradient
    hyp = np.dot(X,theta)
    grad = np.dot(X.T,hyp-y)/float(m)

    ## Add regularization
    reg_var = np.array(theta)
    reg_var[0] = 0 # Don't regularize Theta_0
    grad+= float(reg)/m*reg_var
    
    return grad.flatten()

#####################
'''
Function: fit_theta
Determine the set of parameters that best fit the given training set + answers

Inputs:
    X - Array of features for each training set (including X0 column), of size [m x n]
    y - 1-D vector of answers corresponding to each training example, of size [m]
    REG - regularization parameter, single value
Outputs:
    Theta - Vector of parameters that provides best fit
    Cost - Cost produced by chosen Theta.
'''
def fit_theta(X, y, reg):
    
    ## Make sure theta, X, and y are matrices of correct dimensions:
    m, n = np.shape(X)
    
    # Initialize Theta
    theta0 = np.zeros(n)

    # Optimize using Broyden, Fletcher, Goldfarb, and Shanno (BFGS) method
    if g_wind==False: opt = op.fmin_bfgs(logCost,theta0,logGrad,args=(X,y,reg),full_output=True,disp=False)
    else: opt = op.fmin_bfgs(linCost,theta0,linGrad,args=(X,y,reg),full_output=True,disp=False)
    theta = opt[0]
    cost = opt[1]

    return theta, cost

#####################
'''
Function: split_data
Randomly split the full set of training examples and answers into separate training set (60% of data), Cross-validation set (20%), and test set (20%)

Inputs:
    X - Full array of training examples, including X0 feature. Of size [m x n]
    y - 1-D vector of answers corresponding to each training example, of size [m]
Outputs:
    X_train - Array of training examples, of size [m_train x n]
    y_train - Vector of answers, of size [m_train]
    X_cv - Array of cross-validation examples, of size [m_cv x n]
    y_cv - Vector of cross-validation answers, of size [m_cv]
    x_test - Array of test examples, of size [m_test x n]
    y_test - Vector of test answers, of size [m_test]
'''
def split_data(X, y):

    ## Find size of full array
    m, n = np.shape(X)
    ind60 = round(g_frac_data*m)
    ind80 = round((1+g_frac_data)/2.*m)

    ## Randomly shuffle the m indices, then split into training, cv, and test sets
    ind_shuf = np.random.permutation(m)
    ind_train = ind_shuf[0:ind60]
    ind_cv = ind_shuf[ind60:ind80]
    ind_test = ind_shuf[ind80:]

    X_train = X[ind_train,:]
    y_train = y[ind_train]
    X_cv = X[ind_cv,:]
    y_cv = y[ind_cv]
    X_test = X[ind_test,:]
    y_test = y[ind_test]

    return X_train, y_train, X_cv, y_cv, X_test, y_test

#####################
'''
Function: validationCurve
Find the best regularization parameter that lowers the Cost Function

Inputs:
    X - Array of features for the training set (including X0 column), of size [m x n]
    Y - 1-D vector of answers corresponding to each training example, of size [m]
    X_CV - Array of features for the cross-validation set (including X0 column), of size [m_cv x n]
    Y_CV - 1-D vector of answers corresponding to the CV set, of size [m_cv]
    REG_ARR - Optional input that specifies the regularization parameters to try. Default is set to the global g_regularization
Outputs:
    ERROR_TRAIN - Vector of costs from the training set, corresponding to each regularization parameter.
    ERROR_CV - Vector of costs from the cross-validation set, corresponding to each regularization parameter
    REG_BEST - Regularization parameter that minimizes the CV error
'''
def validationCurve(X,y,X_cv,y_cv,reg_arr=g_regularization,wind=True):

    ## Empty arrays
    error_train = np.zeros(len(reg_arr))
    error_cv = np.zeros(len(reg_arr))

    # Loop over all regularization parameters
    for ii, reg in enumerate(reg_arr):

        # Find optimum theta using this regularization parameter
        theta, J = fit_theta(X,y,reg)

        # Calculate cost from using this theta with no regulalrization on CV and training sets:
        if g_wind==False:
            error_train[ii] = logCost(theta,X,y,0)
            error_cv[ii] = logCost(theta,X_cv,y_cv,0)
        else:
            error_train[ii] = linCost(theta,X,y,0)
            error_cv[ii] = linCost(theta,X_cv,y_cv,0)

    # Find regularization parameter that produces lowest CV error
    reg_best = reg_arr[np.argmin(error_cv)]

    return error_train, error_cv, reg_best

#####################
'''
Function: calc_f1
Find the precision and recall arrays as a function of threshold values, and calculate which threshold results in the best F1 value

Inputs:
    X - Array of features for the training set (including X0 column), of size [m x n]
    Y - 1-D vector of answers corresponding to each training example, of size [m]
    X_CV - Array of features for the cross-validation set (including X0 column), of size [m_cv x n]
    Y_CV - 1-D vector of answers corresponding to the CV set, of size [m_cv]
    REG - Regularization Parameter to use
    THRESHOLD_ARR - Vector of possible threshold values to try. Default is set to global g_threshold
Outputs:
    PRECISION_ARR - Vector of precision values. Precision = True positives/(True positives + False positives)
    RECALL_ARR - Vector of recall values. Recall = True positives/(True positives + False negatives)
    F1_ARR - Vector of F1 values. F1 = 2*(P*R)/(P+R)
    BEST_THRESHOLD - Threshold value that corresponds to the lowest F1 value
    ERR_BEST - 3-element array containing precision, recall, and F1 value at BEST_THRESHOLD
'''
def calc_f1(X,y,X_cv,y_cv,reg,threshold_arr=g_threshold):

    # Empty arrays
    prec_arr = np.zeros(len(threshold_arr))
    recall_arr = np.zeros(len(threshold_arr))

    # Find best parameters
    theta, J_train = fit_theta(X,y,reg)
    # Find hypothesis by applying parameters (theta) to CV set
    if g_wind==False: hyp = sigmoid(np.dot(X_cv,theta))
    else: hyp = np.dot(X_cv,theta)

    # Loop over values of threshold
    for ii, threshold in enumerate(threshold_arr):

        # Make prediction array based on hypothesis and threshold
        pred = np.zeros(len(y_cv))
        pred[hyp >= threshold] = 1.

        # Find true positive, false positives, false negatives
        true_pos = np.sum(np.all([pred==1,y_cv==1],axis=0)).astype(float)
        false_pos = np.sum(np.all([pred==1,y_cv==0],axis=0)).astype(float)
        false_neg = np.sum(np.all([pred==0,y_cv==1],axis=0)).astype(float)

        # Calculate precision & recall
        prec_arr[ii] = true_pos/(true_pos+false_pos)
        recall_arr[ii] = true_pos/(true_pos+false_neg)

    # Calculate F1
    f1_arr = 2 * (prec_arr * recall_arr)/(prec_arr + recall_arr)

    # Find value of threshold that gives best F1 value
    try:
        best_ind = np.nanargmin(f1_arr)
        best_threshold = threshold_arr[best_ind]
        err_best = [prec_arr[best_ind],recall_arr[best_ind],f1_arr[best_ind]]

    except:
        ## If F1 is all NaNs
        best_threshold = np.nan
        err_best = [np.nan,np.nan,np.nan]
        
    return prec_arr, recall_arr, f1_arr, best_threshold, err_best

#####################
'''
Function: predict
   Generate the hypothesis of a set of data given the chosen theta matrix.
Inputs:
   THETA - Set of parameters (includes X0 parameters). Vector of size n
   X - Array of features for the test set (including X0 column), of size [m x n]
   Y - Set of answers. Vector of size [m]
Keywords:
   THRES - Threshold parameter (above which is considered a positive detection). Default = 0.5
   REG_TYPE - Type of regression being used, options are 'log' (logistic) or 'lin' (linear). Default is 'log' 
Outputs:
'''
def predict(theta,X,y,thres=0.5,reg_type='log'):

    ## Make sure theta, X, and y are matrices of correct dimensions:
    m, n = np.shape(X)
    y = y.reshape((m,1))
    theta = theta.reshape((n,1))

    # Calculate Hypothesis 
    if reg_type=='lin':
        hyp = np.dot(X,theta)
        pred_y = hyp
    else:
        hyp = sigmoid(np.dot(X,theta))
        pred_y = np.where(hyp.reshape((1,m))>thres,np.zeros(m)+1,np.zeros(m))

    return pred_y


#####################
'''
Function: MOCK_CATALOG
Create a mock catalog of predicted concentration values using the derived best Theta

Inputs:
   THETA - Vector of parameter values. Length (n)
   MU - Array of mean values used for normalizing, length [n-1] (because x0 has no normalization)
   SIGMA - Array of standard deviation values used for normalizing, length [n-1] (because x0 has no normalization)
   MONTH - Month of data to read - Can be Aug or Oct
   RECHECK - Number of days to re-evaluate the machine learning algorithm with the real previous concentration
Outputs:
   HYP - Hypothesis from using machine learning using full set of real concentrations to measure "previous concentration"
   HYP_ML - Hypothesis from using machine learning using real concentrations only on "recheck" days
Effects:
   Writes a text table identical to that given by MONTH, but replace last column with column of predictions from machine learning
'''
def mock_catalog(theta,mu,sigma,recheck=5,month='Oct'):

    # Define filename
    in_fname = 'catalogs/napa_model_wind_'+month+'.txt'
    out_fname = 'catalogs/napa_model_'+month+'_full_ml{:2.0f}.txt'.format(g_frac_data*100)
    out_fname_ml = 'catalogs/napa_model_'+month+'_recheck'+np.str(recheck)+'_ml{:2.0f}.txt'.format(g_frac_data*100)
      
    # Read in data from table
    dat = np.genfromtxt(in_fname,dtype=np.str)
    #keep = dat[:,0:-1]

    ## Get X and Y features from catalog, normalize X
    X_all, y_all = get_feat('napa',month=month)
    X_norm, m, s = featureNormalize(X_all,mu=mu,sigma=sigma,recalculate=False)
    # Add column of 1s
    X = np.insert(X_norm,0,1,axis=1)
    m, n = np.shape(X)

    ## Figure out number of days
    n_days = m/k_n_grid_napa
    
    ## Apply machine learning to full catalog:
    # Calculate hypothesis from machine learning
    hyp = np.dot(X,theta)

    ## Use machine learning results to build through each day
    ## Loop over data, running machine learning on each day at a time
    hyp_ml = np.empty(len(hyp))
    prev_conc = np.zeros(k_n_grid_napa)+1.  # Initialize as 1 
    for ind in range(n_days):
        # Grab X features from full array
        X_bin = X[ind*k_n_grid_napa:(ind+1)*k_n_grid_napa,:]
      
        # Replace previous concentration column with what was calculated from yesterday's ML (and normalize)
        # Only do this if it doesn't fall on a recheck day
        if np.mod(ind,recheck)!=0:
            X_bin[:,1] = (prev_conc-mu[0])/sigma[0]

        # Calculate new concentration using ML
        hyp_bin = np.dot(X_bin,theta)
        hyp_ml[ind*k_n_grid_napa:(ind+1)*k_n_grid_napa] = hyp_bin
       
        # Save results as new previous concentration array
        prev_conc = np.array(hyp_bin)
       
    ## Write new tables
    tab = open(out_fname,'w')
    tab_ml = open(out_fname_ml,'w')
    # Loop over every entry
    for ii, pred in enumerate(hyp):
        # Full array at once
        tab_str = string.join(dat[ii,0:-1],' ')
        # Add hypothesis
        tab_str += '  {:8.3f}\n'.format(pred)
        tab.write(tab_str)
        
        # Looped array
        tab_ml_str = string.join(dat[ii,0:-1],' ') + '  {:8.3f}\n'.format(hyp_ml[ii])
        tab_ml.write(tab_ml_str)
        
    tab.close()
    tab_ml.close()      

    return hyp, hyp_ml
    

#####################
###### Run Code #####
#####################
if __name__ == '__main__':

    # Process command line arguments
    main(sys.argv[1:])
    
    # Get all of the features & answers
    if g_wind: allX, y = get_feat('napa')
    else: allX, y = get_all_feat(year=g_year)
    
    # Normalize features
    normX, mu, sigma = featureNormalize(allX)
    
    # Add column of 1's for parameter x0
    X = np.insert(normX,0,1,axis=1)
    m, n = np.shape(X)

    # Empty arrays to hold interesting values through loops
    all_train = np.zeros([k_n_iter,len(g_regularization)])
    all_cv = np.zeros([k_n_iter,len(g_regularization)])
    all_prec = np.zeros([k_n_iter,len(g_threshold)])
    all_recall = np.zeros([k_n_iter,len(g_threshold)])
    all_f1 = np.zeros([k_n_iter,len(g_threshold)])
    test_err = np.empty(k_n_iter)
    pred_err = np.empty((k_n_iter,4))
    opt_reg = np.empty(k_n_iter)
    all_theta = np.empty([k_n_iter,n])
    all_farm_test = np.empty([k_n_iter,6])
    all_farm_pred = np.empty([k_n_iter,6])
    
    ## Loop over the desired number of iterations
    for ii in np.arange(k_n_iter): 

        # Split data into Training, CV, and Test
        X_train, y_train, X_cv, y_cv, X_test, y_test = split_data(X,y)

        # Create validation curve
        err_train, err_cv, best_reg = validationCurve(X_train,y_train,X_cv,y_cv)
        all_train[ii,:] = err_train
        all_cv[ii,:] = err_cv

        # Calculate error on test set
        theta_iter, cost_iter = fit_theta(X, y, best_reg)
        all_theta[ii,:] = theta_iter
        opt_reg[ii] = best_reg
        if g_wind==False:
            # Find best threshold using ideal lambda
            prec, recall, f1, best_thres, err_best = calc_f1(X_train,y_train,X_cv,y_cv,best_reg)
            all_prec[ii,:] = prec
            all_recall[ii,:] = recall
            all_f1[ii,:] = f1
            test_err[ii] = logCost(theta_iter,X_test,y_test,0)

            # Find how accurate the test results are using fraction of correct results
            pred_y = predict(theta_iter,X_test,y_test,thres=best_thres)
            pred_err[ii,:] = np.hstack([np.sum(pred_y != y_test)/np.float(len(y_test)),err_best])

            ## Test 2015 farm data
            if g_year==2015:
                farm_name15 = np.array(['A01','A08','A11','A17','A20','A40'])
                #farm_test = np.empty(len(farm_name15))
                #farm_pred = np.empty(len(farm_name15))
                for jj, name15 in enumerate(farm_name15):
                    # Read in 2015 features
                    Xfarm, yfarm = get_feat(name15,wind_cat=False,year=2015)
                    # Normalize features
                    normX_farm, mu_foo, sig_foo = featureNormalize(Xfarm,mu=mu,sigma=sigma,recalculate=False)
                    X_farm = np.insert(normX_farm,0,1,axis=1)
                    # Calculate test error
                    all_farm_test[ii,jj] = logCost(theta_iter,X_farm,yfarm,0)
                    # Calculate prediction error
                    pred_y = predict(theta_iter,X_farm,yfarm,thres=best_thres)
                    all_farm_pred[ii,jj] = np.sum(pred_y != yfarm)/np.float(len(yfarm))

        else:
            test_err[ii] = linCost(theta_iter,X_test,y_test,0)

        #print 'iteration {:3.0f} done'.format(ii)
        
    # Average all arrays
    avg_train = np.nanmean(all_train,axis=0)
    avg_cv = np.nanmean(all_cv,axis=0)
    avg_prec = np.nanmean(all_prec,axis=0)
    avg_recall = np.nanmean(all_recall,axis=0)
    avg_f1 = np.nanmean(all_f1,axis=0)
    avg_theta = np.nanmean(all_theta,axis=0)

    ## Typical errors
    typ_pred = np.mean(pred_err[:,0])
    typ_prec = np.nanmean(pred_err[:,1])
    typ_recall = np.nanmean(pred_err[:,2])
    typ_f1 = np.nanmean(pred_err[:,3])

    print 'Prediction accuracy: {:5.3f}; Precision: {:5.3f}; Recall:  {:5.3f}; F1: {:5.3f}'.format(typ_pred,typ_prec,typ_recall,typ_f1)
    
    ## See how well the derived model works with the data from 2015
    # Loop over each 2015 farm
    farm_name15 = np.array(['A01','A08','A11','A17','A20','A40'])
    farm_test = np.empty(len(farm_name15))
    farm_pred = np.empty(len(farm_name15))
    for ii, name15 in enumerate(farm_name15):
        # Read in 2015 features
        Xfarm, yfarm = get_feat(name15,wind_cat=False,year=2015)
        # Normalize features
        normX_farm, mu_foo, sig_foo = featureNormalize(Xfarm,mu=mu,sigma=sigma,recalculate=False)
        X_farm = np.insert(normX_farm,0,1,axis=1)
        # Calculate test error
        farm_test[ii] = logCost(avg_theta,X_farm,yfarm,0)
        # Calculate prediction error
        pred_y = predict(avg_theta,X_farm,yfarm)
        farm_pred[ii] = np.sum(pred_y != yfarm)/np.float(len(yfarm))
        
    
    # Write catalogs
    if g_wind==True:
        mock_catalog(avg_theta,mu,sigma,recheck=g_recheck)
        mock_catalog(avg_theta,mu,sigma,month='Aug',recheck=g_recheck)
        
    ## Plot precision, recall, and F1
    #pl.plot(g_threshold,avg_prec,'r-')
    #pl.plot(g_threshold,avg_recall,'b-')
    #pl.plot(g_threshold,avg_f1,'k-')
    #pl.show()
    
    ## Plot regularization values
    pl.plot(g_regularization,err_train,'b-',label='Training Error')
    pl.plot(g_regularization,err_cv,'r-',label='CV Error')
    pl.xscale('log')
    pl.xlabel('Regularization')
    pl.ylabel('Cost')
    pl.legend()
    pl.show()
    
