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
from utils_dce import predict
from utils_dce import get_mse
from utils_dce import get_metrics

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
C_pred = 0

# minimum number of conf in one year in timeslines used to est hyper parameters
C_est = 8 # 8

# conflict type. Som might need lower c_est than 100 to work
# so now this should be sce_mu - so you do not need to take log.
conf_type = 'sce_mu' #['ged_best_sb', 'ged_best_ns', 'ged_best_os', 'ged_best']

# short term kernel
s_kernel = 'Matern32' #['ExpQuad', 'RatQuad', 'Matern32'] #, 'Matern52']

# Start timer
start_time = time.time()

# So, here is the thing. you only have trian in df_merge right now....
# get train and validation id:
train_id, val_id = test_val_train(df, test_time= False)

print(f"{C_est}_{C_pred}_{conf_type}_{s_kernel}\n")

# get pkl mp
path = open('/home/projects/ku_00017/data/generated/currents/dce_mp.pkl', 'rb')
dce_mp = pickle.load(path)
path.close()
print(f"got mp: ℓ_l:{dce_mp['ℓ_l']}, η_l:{dce_mp['η_l']}, ℓ_s:{dce_mp['ℓ_s']}, η_s:{dce_mp['η_s']}, σ:{dce_mp['σ']}")

# Constuction the gps and getting the map
hps = get_hyper_priors() # these might need some changing... 

with pm.Model() as model:

# short term trend/irregularities ---------------------------------

    ℓ_s = pm.Gamma("ℓ_s", alpha=hps['ℓ_alpha_s'] , beta=hps['ℓ_beta_s'])
    η_s = pm.HalfCauchy("η_s", beta=hps['η_beta_s'])

    # mean func for short term trend
    mean_s =  pm.gp.mean.Zero()

    # cov function for short term trend
    cov_s = η_s ** 2 * pm.gp.cov.Matern32(1, ℓ_s) 

    # GP short term trend 
    gp_s = pm.gp.Marginal(mean_func = mean_s, cov_func=cov_s)


    # long term trend -------------------------------------------------
    ℓ_l = pm.Gamma("ℓ_l", alpha=hps['ℓ_alpha_l'] , beta=hps['ℓ_beta_l'])
    η_l = pm.HalfCauchy("η_l", beta=hps['η_beta_l'])
                
    # mean and kernal for long term trend
    mean_l =  pm.gp.mean.Zero()
    cov_l = η_l **2 * pm.gp.cov.ExpQuad(1, ℓ_l) # Cov func.
                
    # GP short term trend 
    gp_l = pm.gp.Marginal(mean_func = mean_l, cov_func=cov_l)

    # noise (constant "white noise") -----------------------------------
    σ = pm.HalfCauchy("σ", beta=hps['σ_beta'])

    # sample and split X,y ---------------------------------------------  
    sample_pr_id = sample_conflict_timeline(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, C = C_est)

    # Full GP ----------------------------------------------------------
    gp = gp_s + gp_l

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
    

# Getting the predictions and merging with original df:
# might be a problem here.. both with log and conf_type. Check utils
df_new = predict(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, mp = dce_mp, gp = gp, gp_s = gp_s, gp_l = gp_l, σ=σ, C=C_pred)

dce_pred_df = pd.merge(df_new, df[['id', 'pg_id','year','gwcode', 'xcoord', 'ycoord','ged_best_sb']], how = 'left', on = ['id', 'pg_id'])

dce_pred_df.rename(columns =  {'mu' : 'dce_mu', 'var': 'dce_var', 
                             'mu_s' : 'dce_mu_s', 'var_s': 'dce_var_s', 
                             'mu_l' : 'dce_mu_l', 'var_l': 'dce_var_l'}, inplace=True)


print('Pickling...')
new_file_name = '/home/projects/ku_00017/data/generated/currents/dce_pred_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(dce_pred_df, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)





