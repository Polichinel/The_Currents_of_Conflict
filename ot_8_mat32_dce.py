import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import time

from IPython.display import clear_output

# some prob need fitting now... 
from utils_dce import get_views_coord
from utils_dce import test_val_train
from utils_dce import sample_conflict_timeline
from utils_dce import get_hyper_priors
from utils_dce import predict_ot
from utils_dce import get_mse_ot
from utils_dce import get_metrics_ot

import pymc3 as pm
import theano

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn import metrics

import warnings
warnings.simplefilter("ignore", UserWarning)

# get df:
#pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/sce_pred_df.pkl', 'rb')
pkl_file = open('/home/projects/ku_00017/data/generated/currents/sce_pred_df.pkl' , 'rb')
sce_pred_df = pickle.load(pkl_file)
pkl_file.close()

# get df:
#path = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS' 
path = '/home/projects/ku_00017/data/generated/currents' 
file_name = 'ViEWS_coord.pkl'
df_views_coord = get_views_coord(path = path, file_name = file_name)

# Notice: so you get all obs now but naturally mu etc. is only there for the set used in get_sce_pred_df
df = pd.merge(sce_pred_df, df_views_coord[['id', 'pg_id', 'month_id', 'month', 'year','gwcode', 'xcoord', 'ycoord','ged_best_sb', 'ged_dummy_sb']],how = 'outer', on = ['month_id', 'xcoord', 'ycoord'])
df.rename(columns =  {'mu' : 'sce_mu', 'var': 'sce_var'}, inplace=True)

# ---------------------------------------------------------------------------------------
# as a start we just still assume two trends.
# you need to take do the whole full experimt thing agina..
# in the end, you need to import an use cm_pred in the predicitons to get anything acurrate: yoh want complimentary signals!
# ---------------------------------------------------------------------------------------

# as a start we just still assume two trends.
# dict for the dfs/dicts holding the results
out_dict = {}

# minimum number of conf in timeslines predicted. C = 0 for full run
C_pred = 1#0

# minimum number of conf in one year in timeslines used to est hyper parameters
C_est = 8 # 8

# conflict type. Som might need lower c_est than 100 to work
# so now this should be sce_mu - so you do not need to take log.
conf_type = 'sce_mu' #['ged_best_sb', 'ged_best_ns', 'ged_best_os', 'ged_best']

# short term kernel
#s_kernel = 'Matern32' #['ExpQuad', 'RatQuad', 'Matern32'] #, 'Matern52']

# Start timer
start_time = time.time()

# So, here is the thing. you only have trian in df_merge right now....
# get train and validation id:
train_id, val_id = test_val_train(df, test_time= False)

print(f"{C_est}_{C_pred}_{conf_type}_'Mat32'\n")


# Priors
η_beta = 2
ℓ_beta = 0.4
ℓ_alpha = 8
σ_beta = 5

with pm.Model() as model:

    # trend
    ℓ = pm.Gamma("ℓ", alpha=ℓ_alpha , beta=ℓ_beta)
    η = pm.HalfCauchy("η", beta=η_beta)
    cov = η **2 * pm.gp.cov.Matern32(1, ℓ) # Cov func.

    # noise model
    σ = pm.HalfCauchy("σ", beta=σ_beta)

    # mean func. (constant) 
    mean =  pm.gp.mean.Zero()# placeholder if you do individuel

    # sample and split X,y ---------------------------------------------  
    sample_pr_id = sample_conflict_timeline(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, C = C_est)

    # GP
    gp = pm.gp.Marginal(mean_func = mean, cov_func=cov)

    df_sorted = df.sort_values(['pg_id', 'month_id'])

    # shared:
    pg_len = df_sorted[df_sorted['id'].isin(train_id)]['month_id'].unique().shape[0]
    X = theano.shared(np.zeros(pg_len)[:,None], 'X')
    y = theano.shared(np.zeros(pg_len), 'y')

    # sample:
    for i, j in enumerate(sample_pr_id):

        print(f'Time-line {i+1}/{sample_pr_id.shape[0]} in the works (estimation)...', end= '\r') 

        X.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None])
        y.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type].values)

        y_ = gp.marginal_likelihood(f'y_{i}', X=X, y=y, noise= σ)
    
    mp = pm.find_MAP()

map_df = pd.DataFrame({"Parameter": ["ℓ", "η", "σ"],"Value at MAP": [float(mp["ℓ"]), float(mp["η"]), float(mp["σ"])]}) 
# Getting the predictions and merging with original df:
# might be a problem here.. both with log and conf_type. Check utils

df_new = predict_ot(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, mp = mp, gp = gp, σ=σ, C=C_pred)

df_merged = pd.merge(df_new, df[['id', 'pg_id','year','gwcode', 'xcoord', 'ycoord','ged_best_sb']], how = 'left', on = ['id', 'pg_id'])

df_merged.rename(columns =  {'mu' : 'dce_mu', 'var': 'dce_var', 
                             'mu_s' : 'dce_mu_s', 'var_s': 'dce_var_s', 
                             'mu_l' : 'dce_mu_l', 'var_l': 'dce_var_l'}, inplace=True)

# getting mse results:
# need to fix stuff here prob...
print('Getting MSE')
mse_resutls_df = get_mse_ot(df_merged = df_merged, train_id = train_id, test_id = val_id)

# Creating devrivatives:
print('Creating devrivatives')
df_merged.sort_values(['pg_id', 'X'], inplace= True)
df_merged['dce_mu_slope'] = df_merged.groupby('pg_id')['dce_mu'].transform(np.gradient)
df_merged['dce_mu_acc'] = df_merged.groupby('pg_id')['dce_mu_slope'].transform(np.gradient)
df_merged['dce_mu_mass'] = df_merged.groupby('pg_id')['dce_mu'].transform(np.cumsum)

df_merged['dce_mu_s_slope'] = df_merged.groupby('pg_id')['dce_mu_s'].transform(np.gradient)
df_merged['dce_mu_s_acc'] = df_merged.groupby('pg_id')['dce_mu_s_slope'].transform(np.gradient)
df_merged['dce_mu_s_mass'] = df_merged.groupby('pg_id')['dce_mu_s'].transform(np.cumsum)

df_merged['dce_mu_l_slope'] = df_merged.groupby('pg_id')['dce_mu_l'].transform(np.gradient)
df_merged['dce_mu_l_acc'] = df_merged.groupby('pg_id')['dce_mu_l_slope'].transform(np.gradient)
df_merged['dce_mu_l_mass'] = df_merged.groupby('pg_id')['dce_mu_l'].transform(np.cumsum)


# Get classification results
# also need new one here.
# variables have new names and you must have cm on here: what complimentary signals not the same.
print('Getting classifcation results')
df_results = get_metrics_ot(df_merged = df_merged, train_id = train_id, test_id = val_id)

# "filing" names
print('Saving..')
pre_script_map_df = f'{C_est}_{C_pred}_{conf_type}_Mat32_map_df'
pre_script_mse_resutls_df = f'{C_est}_{C_pred}_{conf_type}_Mat32_mse_results_df'
pre_script_df_results = f'{C_est}_{C_pred}_{conf_type}_Mat32_df_results'
pre_script_df = f'{C_est}_{C_pred}_{conf_type}_Mat32_df_merged'

# Save in the eksperiments_dict
out_dict[pre_script_map_df] = map_df
out_dict[pre_script_mse_resutls_df] = mse_resutls_df
out_dict[pre_script_df_results] = df_results
out_dict[pre_script_df] = df_merged
            
new_file_name = '/home/projects/ku_00017/data/generated/currents/ot_8_mat32_dce_dict.pkl'
output = open(new_file_name, 'wb')
pickle.dump(out_dict, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)







