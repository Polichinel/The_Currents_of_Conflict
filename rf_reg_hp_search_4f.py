print('importing libs...', end = '\r')

import os
import numpy as np
import pandas as pd

import pickle
import time

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm
import seaborn as sns

from utils_ens import get_Xy_tt
from utils_ens import my_sigmoid

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

print('Getting data......', end = '\r')
#pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/rf_selected_features.pkl', 'rb')
pkl_file = open('/home/projects/ku_00017/data/generated/currents/rf_selected_features.pkl', 'rb')
selected_features = pickle.load(pkl_file)
pkl_file.close()

X_train, y_train, X_test, y_test = get_Xy_tt(local = False, binary_y= False)
n_rounds = 50

best_features = selected_features['features'][:4].values # should have specific feature selection for reg... also try with 4..

# hp lists:
max_depth_list = [] 
n_estimators_list = []
min_samples_split_list = []
criterion_list = []
class_weight_list = []
max_features_list = []
min_samples_leaf_list = []

# metric lists:
# train
train_preds_con = []
test_preds_con = []
train_preds_prob = []
test_preds_prob = []

MSE_train_list = []
MAE_train_list = []

AUC_train_list = []
AP_train_list = []
BS_train_list = []

pr_train_list = []
roc_train_list = []

# test
MSE_test_list = []
MAE_test_list = []

AUC_test_list = [] 
AP_test_list = []
BS_test_list = []

pr_test_list = []
roc_test_list = []

print('Beginning loop')
for i in range(n_rounds):

    # Variable hyper parameters
    n_estimators = np.random.randint(100,150) # just to check if it is running at all...
    min_samples_split = np.random.randint(2,15) # could go higher..
    max_depth = np.random.randint(5,11)
    min_samples_leaf = np.random.randint(1,200)
    
    #criterion = ['mse', 'mae'][np.random.randint(0,2)] # mae still not run now with log(y)
    criterion ='mse'
    max_features = ['auto', 'sqrt', 'log2'][np.random.randint(0,3)]


    print(f'defining model {i+1}/{n_rounds}...  \nn_estimators: {n_estimators}, \nmin_samples_split: {min_samples_split}, \nmax_depth : {max_depth}, \nmin_sample_leaf : {min_samples_leaf}, \ncriterion : {criterion}, \nmax_features : {max_features} ')


    model = RandomForestRegressor( n_estimators=n_estimators, criterion = criterion, max_depth = max_depth, 
                                   min_samples_split= min_samples_split, min_samples_leaf= min_samples_leaf,
                                   random_state=i, n_jobs= 18)
    
    print(f'fitting model {i+1}/{n_rounds}...', end = '\r')
    model.fit(X_train[best_features], y_train)

    # metrics
    print(f'getting metrics from model {i+1}/{n_rounds}...', end = '\r')
    y_train_pred_con = model.predict(X_train[best_features])
    y_test_pred_con = model.predict(X_test[best_features])

    y_train_pred_prob = my_sigmoid(y_train_pred_con) # maybe do a logistic regression instead...
    y_test_pred_prob = my_sigmoid(y_test_pred_con) # maybe do a logistic regression instead...

    y_train_bi = (y_train > 0)*1
    y_test_bi = (y_test > 0)*1

    # train:
    MSE_train_list.append(metrics.mean_squared_error(y_train, y_train_pred_con))
    MAE_train_list.append(metrics.mean_absolute_error(y_train, y_train_pred_con))
    
    AUC_train_list.append(metrics.roc_auc_score(y_train_bi, y_train_pred_prob))
    AP_train_list.append(metrics.average_precision_score(y_train_bi, y_train_pred_prob))
    BS_train_list.append(metrics.brier_score_loss(y_train_bi, y_train_pred_prob))

    precision_train, recall_train, _ = metrics.precision_recall_curve(y_train_bi, y_train_pred_prob)
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train_bi, y_train_pred_prob)

    pr_train_list.append((precision_train, recall_train))
    roc_train_list.append((fpr_train, tpr_train))

    # test:
    MSE_test_list.append(metrics.mean_squared_error(y_test, y_test_pred_con))
    MAE_test_list.append(metrics.mean_absolute_error(y_test, y_test_pred_con))
    
    AUC_test_list.append(metrics.roc_auc_score(y_test_bi, y_test_pred_prob))
    AP_test_list.append(metrics.average_precision_score(y_test_bi, y_test_pred_prob))
    BS_test_list.append(metrics.brier_score_loss(y_test_bi, y_test_pred_prob))

    precision_test, recall_test, _ = metrics.precision_recall_curve(y_test_bi, y_test_pred_prob)
    fpr_test, tpr_test, _ = metrics.roc_curve(y_test_bi, y_test_pred_prob)    

    pr_test_list.append((precision_test, recall_test))
    roc_test_list.append((fpr_test, tpr_test))
   
    train_preds_con.append(y_train_pred_con)
    test_preds_con.append(y_test_pred_con)
    train_preds_prob.append(y_train_pred_prob)
    test_preds_prob.append(y_test_pred_prob)

    # hps:
    n_estimators_list.append(n_estimators)
    max_depth_list.append(max_depth)
    min_samples_split_list.append(min_samples_split)
    criterion_list.append(criterion)
    max_features_list.append(max_features)
    min_samples_leaf_list.append(min_samples_leaf)

    loop_string = f'{i+1}/{n_rounds} \n done. MSE test: {MSE_test_list[i]} MSE train: {MSE_train_list[i]} \n MAE test: {MAE_test_list[i]} MAE train: {MAE_train_list[i]} \n AP test: {AP_test_list[i]}, AP train: {AP_train_list[i]} \n'
    print(loop_string , end='\r')


print('Making datafame', end = '\r')
hp_df = pd.DataFrame({'n_estimators' : n_estimators_list, 'max_depth' : max_depth_list, 'min_samples_split' : min_samples_split_list,
                      'min_samples_leaf' : min_samples_leaf_list ,'criterion' : criterion_list, 'max_features' : max_features_list,  
                      'test_preds_con' : test_preds_con, 'test_preds_prob' : test_preds_prob,
                      'MSE' : MSE_test_list, 'MAE' : MAE_test_list, 
                      'AP' : AP_test_list, 'PR' : pr_test_list, 'ROC' : roc_test_list})


print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/rf_reg_hp_4f_df.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/rf_reg_hp_4f_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(hp_df, output)
output.close()