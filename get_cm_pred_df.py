
import os
import numpy as np
import pandas as pd

import pickle
import time

from IPython.display import clear_output

from utils import get_views_coord
from utils import test_val_train
from utils import sample_conflict_timeline
from utils import get_hyper_priors
from utils import predict

import pymc3 as pm
import theano

import warnings
warnings.simplefilter("ignore", UserWarning)

# Start timer
start_time = time.time()

# minimum number of conf in timeslines predicted. C = 0 for full run
C_pred = 0 # 100

# minimum number of conf in one year in timeslines used to est hyper parameters
C_est = 8

# conflict type. Som might need lower c_est than 100 to work
conf_type = 'ged_best_sb' #['ged_best_sb', 'ged_best_ns', 'ged_best_os', 'ged_best']

# get df:
path = '/home/projects/ku_00017/data/generated/currents' 
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)
print('Got df')

# get train and validation id:
train_id, val_id = test_val_train(df, test_time= True)
print('Got train/val ids')

# get pkl mp
path = open('/home/projects/ku_00017/data/generated/currents/cm_mp.pkl', 'rb')
cm_mp = pickle.load(path)
path.close()
print(f"got mp: ℓ_l:{cm_mp['ℓ_l']}, η_l:{cm_mp['η_l']}, ℓ_s:{cm_mp['ℓ_s']}, η_s:{cm_mp['η_s']}, σ:{cm_mp['σ']}")

# Getting hps
hps = get_hyper_priors(plot = False)

print(f"{C_est}_{C_pred}_{conf_type}_Matern32\n")

with pm.Model() as model:

# short term trend/irregularities ---------------------------------

    ℓ_s = pm.Gamma("ℓ_s", alpha=hps['ℓ_alpha_s'] , beta=hps['ℓ_beta_s'])
    η_s = pm.HalfCauchy("η_s", beta=hps['η_beta_s'])

    # cov function for short term trend
    cov_s = η_s ** 2 * pm.gp.cov.Matern32(1, ℓ_s) 

    # mean func for short term trend
    mean_s =  pm.gp.mean.Zero()

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

        print(f'Time-line {i+1}/{sample_pr_id.shape[0]} in the works (building the model)...', end = '\r') 

        X.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None])
        y.set_value(np.log(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values)

        y_ = gp.marginal_likelihood(f'y_{i}', X=X, y=y, noise= σ)
    
# Getting the predictions and merging with original df:
print('Begins prediction')
cm_pred_df = predict(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, mp = cm_mp, gp = gp, gp_s = gp_s, gp_l = gp_l, σ=σ, C=C_pred)

print('Pickling...')
new_file_name = '/home/projects/ku_00017/data/generated/currents/cm_pred_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(cm_pred_df, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)