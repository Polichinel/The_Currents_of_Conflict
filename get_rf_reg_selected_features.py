import os
import numpy as np
import pandas as pd

import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from utils_ens import get_Xy_tt

X_train, y_train, X_test, y_test = get_Xy_tt(local = False, binary_y=False)


chosen_features = []
chosen_features_MSE = []

n_feat = X_train.columns.shape[0]

for j in range(n_feat): # set 4 for test

    x_list = []
    MSE_val_list = []
    MSE_train_list = []

    mask = np.isin(X_train.columns, chosen_features, invert = True)
    n_i = mask.sum()
    for i, x in enumerate(X_train.columns[mask]): # is the index of the (new) feature tested

        #x = X_train.columns[i]
        X_temp = chosen_features + [x]

        print(f'{i+1}/{n_i}, Running with: {X_temp}')

        model_tmp = RandomForestRegressor(n_estimators=100, criterion = 'mse', max_depth = 9, min_samples_split = 10, min_samples_leaf= 100, random_state=42, n_jobs= 18) # HP from quick naive search

        model_tmp.fit(X_train[X_temp], y_train)

        y_train_pred = model_tmp.predict(X_train[X_temp])
        y_val_pred = model_tmp.predict(X_test[X_temp])

        x_list.append(x)
        MSE_train_list.append(metrics.mean_squared_error(y_train, y_train_pred))
        MSE_val_list.append(metrics.mean_squared_error(y_test, y_val_pred))

        print(f'train: {MSE_train_list[i]}, test: {MSE_val_list[i]}\n')

    df_temp = pd.DataFrame({'x': x_list, 'MSE': MSE_val_list})
    chosen_features.append(df_temp.sort_values('MSE', ascending= True).iloc[0]['x'])
    chosen_features_MSE.append(df_temp.sort_values('MSE', ascending= True).iloc[0]['MSE'])

    print(f'round {j+1}/{n_feat}. choosenfeatures: {chosen_features} w/ MSE: {chosen_features_MSE[j]}\n\n')

    # the break sould be here somewhere...

selected_features = pd.DataFrame({'features' : chosen_features, 'MSE' : chosen_features_MSE})

print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/rf_reg_selected_features.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/rf_reg_selected_features.pkl'
output = open(new_file_name, 'wb')
pickle.dump(selected_features, output)
output.close()


