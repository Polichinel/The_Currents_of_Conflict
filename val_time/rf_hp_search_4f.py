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

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# get df:
#pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/rf_selected_features.pkl', 'rb')
pkl_file = open('/home/projects/ku_00017/data/generated/currents/rf_selected_features.pkl', 'rb')
selected_features = pickle.load(pkl_file)
pkl_file.close()

X_train, y_train, X_test, y_test = get_Xy_tt(local = False)
n_rounds = 1000

best_features = selected_features['features'][:4].values # four first chosen features from forward featurte selection.

n_estimators_list = []
min_samples_split_list = []
max_depth_list = []
criterion_list = []
max_features_list = []
min_samples_leaf_list = []


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
    n_estimators = np.random.randint(120,150)
    min_samples_split = np.random.randint(4,7)
    max_depth = np.random.randint(5,7)
    min_samples_leaf = np.random.randint(1,100)
    
    criterion = ['gini', 'entropy'][np.random.randint(0,2)] 
    max_features = ['auto', 'sqrt', 'log2'][np.random.randint(0,3)]
    
    model = RandomForestClassifier( n_estimators=n_estimators, criterion = criterion, max_depth = max_depth, 
                                    min_samples_split= min_samples_split, min_samples_leaf= min_samples_leaf,
                                    random_state=i, n_jobs= 18)
    
    model.fit(X_train[best_features], y_train)

    # metrics
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

    # hps:
    n_estimators_list.append(n_estimators)
    max_depth_list.append(max_depth)
    min_samples_split_list.append(min_samples_split)
    criterion_list.append(criterion)
    max_features_list.append(max_features)
    min_samples_leaf_list.append(min_samples_leaf)
    
    print(f'{i+1}/{n_rounds} done. AP test: {AP_test_list[i]}, AP train: {AP_train_list[i]}', end='\r')

hp_df = pd.DataFrame({'n_estimators' : n_estimators_list, 'max_depth' : max_depth_list, 'min_samples_split' : min_samples_split_list,
                      'min_samples_leaf' : min_samples_leaf_list ,'criterion' : criterion_list, 'max_features' : max_features_list,  
                      'test_preds' : test_preds, 'AP' : AP_test_list, 'PR' : pr_test_list, 'ROC' : roc_test_list})


print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/rf_hp_4f_df.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/rf_hp_4f_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(hp_df, output)
output.close()
