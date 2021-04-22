import os
import numpy as np
import pandas as pd

import pickle


def get_Xy_tt(local = False, binary_y = True):
    
    # get df:
    
    if local == True:
        pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/preds_df.pkl', 'rb')
    
    else:
        pkl_file = open('/home/projects/ku_00017/data/generated/currents/preds_df.pkl', 'rb')
    
    df_merged = pickle.load(pkl_file)
    pkl_file.close()

    feature_set = [ 'dce_mu', 'dce_mu_slope', 'dce_mu_acc', 'dce_mu_mass',
                    'dce_mu_s', 'dce_mu_s_slope','dce_mu_s_acc', 'dce_mu_s_mass',
                    'dce_mu_l', 'dce_mu_l_slope', 'dce_mu_l_acc', 'dce_mu_l_mass',
                    'dce_var', 'dce_var_s', 'dce_var_l', 
                    'cm_mu', 'cm_mu_slope', 'cm_mu_acc', 'cm_mu_mass',
                    'cm_mu_s', 'cm_mu_s_slope', 'cm_mu_s_acc', 'cm_mu_s_mass',
                    'cm_mu_l', 'cm_mu_l_slope', 'cm_mu_l_acc', 'cm_mu_l_mass',
                    'cm_var', 'cm_var_s', 'cm_var_l']


    X_train = df_merged[df_merged['train'] == 1][feature_set] 
    X_test = df_merged[df_merged['train'] == 0][feature_set] # val, not test       
 
    if binary_y == True:

        y_train = (df_merged[df_merged['train'] == 1]['ged_best_sb'] > 0) * 1
        y_test = (df_merged[df_merged['train'] == 0]['ged_best_sb'] > 0) * 1 # val, not test

    elif binary_y == False:

        y_train = np.log(df_merged[df_merged['train'] == 1]['ged_best_sb']) # try with log you tool
        y_test = np.log(df_merged[df_merged['train'] == 0]['ged_best_sb']) # try with log you tool.. and its val, not test

       

    print(f'X_train: {X_train.shape}')
    print(f'y_train: {y_train.shape}')
    print(f'X_test: {X_test.shape}')
    print(f'y_test: {y_test.shape}')

    return(X_train, y_train, X_test, y_test)


def my_sigmoid(x):
    s = 1/(1+np.e**(-x))
    return(s)


