import os
import numpy as np
import pandas as pd

import pickle
import time

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics

from utils_ens import get_Xy_tt

X_train, y_train, X_test, y_test = get_Xy_tt(local = False)


chosen_features = []
chosen_features_ap = []

n_feat = X_train.columns.shape[0]

for j in range(n_feat): # set 4 for test

    x_list = []
    AP_val_list = []
    AP_train_list = []

    mask = np.isin(X_train.columns, chosen_features, invert = True)
    n_i = mask.sum()
    for i, x in enumerate(X_train.columns[mask]): # is the index of the (new) feature tested

        #x = X_train.columns[i]
        X_temp = chosen_features + [x]

        print(f'{i+1}/{n_i}, Running with: {X_temp}')

        model_tmp = XGBClassifier(n_estimators=128, max_depth = 5, gamma = 0.7, reg_lambda = 0.25, random_state=42, n_jobs= 4, objective='binary:logistic') #, use_label_encoder=False)# HP from quick naive search
        #  max_depth = 2. du bliver nød til også at køre med andre max depth for at se om det ændre noget..

        model_tmp.fit(X_train[X_temp], y_train)

        y_train_pred = model_tmp.predict_proba(X_train[X_temp])[:,1]
        y_val_pred = model_tmp.predict_proba(X_test[X_temp])[:,1]

        x_list.append(x)
        AP_train_list.append(metrics.average_precision_score(y_train, y_train_pred))
        AP_val_list.append(metrics.average_precision_score(y_test, y_val_pred))

        print(f'train: {AP_train_list[i]}, test: {AP_val_list[i]}\n')

    df_temp = pd.DataFrame({'x': x_list, 'AP': AP_val_list})
    chosen_features.append(df_temp.sort_values('AP', ascending= False).iloc[0]['x'])
    chosen_features_ap.append(df_temp.sort_values('AP', ascending= False).iloc[0]['AP'])

    print(f'round {j+1}/{n_feat}. choosenfeatures: {chosen_features} w/ AP: {chosen_features_ap[j]}\n\n')

    # the break sould be here somewhere...

selected_features = pd.DataFrame({'features' : chosen_features, 'AP' : chosen_features_ap})

print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/xgb_selected_features.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/xgb_selected_features.pkl'
output = open(new_file_name, 'wb')
pickle.dump(selected_features, output)
output.close()


