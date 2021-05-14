import numpy as np
import pandas as pd
import pickle

from utils_ens import get_Xy_tt
from sklearn import metrics

# get df:
pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/preds_df_tt.pkl', 'rb')
#pkl_file = open('/home/projects/ku_00017/data/generated/currents/preds_df_tt.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close()


# get df:
pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/rf_class_4f_df_tt.pkl', 'rb')
#pkl_file = open('/home/projects/ku_00017/data/generated/currents/rf_class_4f_df_tt.pkl', 'rb')
df_tt = pickle.load(pkl_file)
pkl_file.close()


def get_monthly_AP_df(df = df, df_tt = df_tt):

    X_train, y_train, X_test, y_test = get_Xy_tt(local = True)

    rf_pred_mean = np.array(df_tt['test_preds']).mean(axis = 0)
    rf_pred_std = np.array(df_tt['test_preds']).std(axis = 0) # you don't use this...

    df_test = df[df['train'] == 0].copy()
    df_test['pred_mean'] = rf_pred_mean
    df_test['pred_std'] = rf_pred_std
    df_test['y_binary'] = y_test.values

    rf_pred_indv = pd.DataFrame.from_dict(dict(zip(df_tt['test_preds'].index, df_tt['test_preds'].values)))
    rf_pred_indv['month'] = df[df['train'] == 0]['X'].values
    rf_pred_indv['y_binary'] = y_test.values

    months = rf_pred_indv['month'].unique()
    indv_preds = rf_pred_indv.columns[:-2]

    indv_AP_dict = {}

    for i in indv_preds:
        monthly_AP_list = []

        for m in months:

            monthly_indv_preds = rf_pred_indv[rf_pred_indv['month'] == m][i].values
            monthly_y = rf_pred_indv[rf_pred_indv['month'] == m]['y_binary'].values
            monthly_indv_AP = metrics.average_precision_score(monthly_y, monthly_indv_preds)
            
            monthly_AP_list.append(monthly_indv_AP)

        indv_AP_dict[str(i)] = monthly_AP_list
        print(f'{i+1}/{indv_preds.shape[0]} indvs done...', end = '\r')

    monthly_AP_df = pd.DataFrame(indv_AP_dict)

    monthly_AP_mean_list = []
    monthly_AP_persitance_list = []

    print('Getting mean preds and persitance model preds')
    for m in months:

        monthly_subset = df_test[ df_test['X'] == m]
        monthly_AP = metrics.average_precision_score(monthly_subset['y_binary'], monthly_subset['pred_mean'])
        monthly_AP_mean_list.append(monthly_AP)

        last_obs_y_binary = (df[df['X'] == df[df['train'] == 1]['X'].max()]['y'] > 0) *1
        monthly_AP_persitance = metrics.average_precision_score(monthly_subset['y_binary'], last_obs_y_binary)
        monthly_AP_persitance_list.append(monthly_AP_persitance)

    monthly_AP_df['mean_preds'] = monthly_AP_mean_list
    monthly_AP_df['persitance_preds'] = monthly_AP_persitance_list
    print('Done...')

    return(monthly_AP_df)

monthly_AP_df = get_monthly_AP_df()

print('Pickling.............')
#new_file_name = '/home/projects/ku_00017/data/generated/currents/monthly_AP_df.pkl'
new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/monthly_AP_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(monthly_AP_df, output)
output.close()