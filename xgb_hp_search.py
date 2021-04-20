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

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics


# get df:
#pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/xgb_selected_features.pkl', 'rb')
pkl_file = open('/home/projects/ku_00017/data/generated/currents/xgb_selected_features.pkl', 'rb')
selected_features = pickle.load(pkl_file)
pkl_file.close()

X_train, y_train, X_test, y_test = get_Xy_tt(local = False)
n_rounds = 500

best_features = selected_features['features'][:8].values # four first chosen features from forward featurte selection.

# hp lists 
learning_rate_list = []
booster_list = []
importance_type_list = []
gamma_list = []
max_depth_list = []
max_delta_step_list = []
colsample_bytree_list = []
reg_alpha_list = []
reg_lambda_list = []
min_child_weight_list = []
scale_pos_weight_list = []
base_score_list = []
n_estimators_list = []

# metric lists
train_preds = []
test_preds = []

AUC_train_list = []
AP_train_list = []
BS_train_list = []

pr_train_list = []
roc_train_list = []

AUC_test_list = [] 
AP_test_list = []
BS_test_list = []

pr_test_list = []
roc_test_list = []

print('Beginning loop')
for i in range(n_rounds):

    # Variable hyper parameters
    learning_rate = np.random.uniform(0.001, 0.1)
    booster = ["gbtree", "gblinear", "dart"][np.random.randint(0,3)]
    importance_type = ["gain", "weight", "cover", "total_gain", "total_cover"][np.random.randint(0,5)]
    gamma = np.random.uniform(0.01, 2)
    max_depth = np.random.randint(2, 11)
    max_delta_step = np.random.randint(2, 11)
    colsample_bytree = np.random.uniform(0.3, 1.0)
    #subsample = np.random.uniform(0.2, 0.9) # might not work due to high imbalance. you do it manually if need be.
    reg_alpha = np.random.uniform(0, 1)
    reg_lambda = np.random.uniform(0.01, 0.9)
    min_child_weight = np.random.randint(1, 9)
    scale_pos_weight = np.random.uniform(0,1)
    base_score = np.random.uniform(0,1)
    n_estimators = np.random.randint(100, 150)
    
    # model
    model = XGBClassifier(learning_rate = learning_rate, booster = booster, importance_type = importance_type, 
                          gamma = gamma, max_depth = max_depth, max_delta_step = max_delta_step, colsample_bytree = colsample_bytree,
                          reg_alpha = reg_alpha,  reg_lambda = reg_lambda, min_child_weight = min_child_weight,
                          scale_pos_weight = scale_pos_weight, base_score = base_score, n_estimators=n_estimators,  
                          random_state=i, n_jobs= 20, objective='binary:logistic') #, use_label_encoder=False)
    
    model.fit(X_train[best_features], y_train)

    # save metrics
    y_train_pred = model.predict_proba(X_train[best_features])[:,1]
    y_test_pred = model.predict_proba(X_test[best_features])[:,1]

    AUC_train_list.append(metrics.roc_auc_score(y_train, y_train_pred))
    AP_train_list.append(metrics.average_precision_score(y_train, y_train_pred))
    BS_train_list.append(metrics.brier_score_loss(y_train, y_train_pred))

    precision_train, recall_train, _ = metrics.precision_recall_curve(y_train, y_train_pred)
    fpr_train, tpr_train, _ = metrics.roc_curve(y_train, y_train_pred)

    pr_train_list.append((precision_train, recall_train))
    roc_train_list.append((fpr_train, tpr_train))

    AUC_test_list.append(metrics.roc_auc_score(y_test, y_test_pred))
    AP_test_list.append(metrics.average_precision_score(y_test, y_test_pred))
    BS_test_list.append(metrics.brier_score_loss(y_test, y_test_pred))

    precision_test, recall_test, _ = metrics.precision_recall_curve(y_test, y_test_pred)
    fpr_test, tpr_test, _ = metrics.roc_curve(y_test, y_test_pred)    

    pr_test_list.append((precision_test, recall_test))
    roc_test_list.append((fpr_test, tpr_test))
       
    train_preds.append(y_train_pred)
    test_preds.append(y_test_pred)

    # save hps:
    learning_rate_list.append(learning_rate)
    booster_list.append(booster)
    importance_type_list.append(importance_type)
    gamma_list.append(gamma)
    max_depth_list.append(max_depth)
    max_delta_step_list.append(max_delta_step)
    colsample_bytree_list.append(colsample_bytree)
    reg_alpha_list.append(reg_alpha)
    reg_lambda_list.append(reg_lambda)
    min_child_weight_list.append(min_child_weight)
    scale_pos_weight_list.append(scale_pos_weight)
    base_score_list.append(base_score)
    n_estimators_list.append(n_estimators)


    print(f'{i+1}/{n_rounds} done', end='\r')


# creating df:
print('Creating DF..')
hp_df = pd.DataFrame({'learning_rate' : learning_rate_list, 'booster' : booster_list, 'importance_type' : importance_type_list,                                                
                      'gamma' : gamma_list, 'max_depth' : max_depth_list, 'max_delta_step' : max_delta_step_list, 
                      'colsample_bytree' : colsample_bytree_list, 'reg_alpha' : reg_alpha_list, 'reg_lambda' : reg_lambda_list, 
                      'min_child_weight' : min_child_weight_list, 'scale_pos_weight' : scale_pos_weight_list, 'base_score' : base_score_list, 
                      'n_estimators' : n_estimators_list, 'test_preds' : test_preds, 'AP' : AP_test_list, 'PR' : pr_test_list, 'ROC' : roc_test_list})


print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/xgb_hp_df.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/xgb_hp_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(hp_df, output)
output.close()

