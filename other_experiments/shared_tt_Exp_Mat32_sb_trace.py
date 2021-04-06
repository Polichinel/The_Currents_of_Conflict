
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import time

from IPython.display import clear_output

from utils import get_views_coord
from utils import test_val_train
from utils import sample_conflict_timeline
from utils import get_hyper_priors
from utils import predict
from utils import plot_predictions
from utils import get_mse
from utils import get_metrics

import pymc3 as pm
import theano

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics

import warnings
warnings.simplefilter("ignore", UserWarning)

# dict for the dfs/dicts holding the results
two_trend_Matern32_dict = {}

# minimum number of conf in timeslines used to est hyper parameters
C_est = 100 #64 #34 #100

# conflict type. Som might need lower c_est than 100 to work
conf_type = 'ged_best_sb' #['ged_best_sb', 'ged_best_ns', 'ged_best_os', 'ged_best']

# short term kernel
s_kernel = 'Matern32' #['ExpQuad', 'RatQuad', 'Matern32'] #, 'Matern52']

# Start timer
start_time = time.time()


# get df:
#path = '/home/polichinel/Documents/Articles/conflict_prediction/data/ViEWS/'
path = '/home/projects/ku_00017/data/generated/currents' 
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)

print('got df')


# get train and validation id:
train_id, val_id = test_val_train(df)

print(f"{C_est}_{conf_type}_ExoQuad_{s_kernel}\n")


# Constuction the gps and getting the map
hps = get_hyper_priors(plot = False)

with pm.Model() as model:

# short term trend/irregularities ---------------------------------

    ℓ_s = pm.Gamma("ℓ_s", alpha=hps['ℓ_alpha_s'] , beta=hps['ℓ_beta_s'])
    η_s = pm.HalfCauchy("η_s", beta=hps['η_beta_s'])

    # mean func for short term trend
    mean_s =  pm.gp.mean.Zero()

    # cov function for short term trend
    if s_kernel == 'ExpQuad': 
        cov_s = η_s ** 2 * pm.gp.cov.ExpQuad(1, ℓ_s) 

    elif s_kernel == 'Matern32': 
        cov_s = η_s ** 2 * pm.gp.cov.Matern32(1, ℓ_s) 

    elif s_kernel == 'Matern52': 
        cov_s = η_s ** 2 * pm.gp.cov.Matern32(1, ℓ_s) 

    elif s_kernel == 'RatQuad': 

        α_s = pm.Gamma("α_s", alpha=hps['α_alpha_s'], beta=hps['α_beta_s']) 
        cov_s = η_s ** 2 * pm.gp.cov.RatQuad(1, ℓ_s, α_s) # this seems to help alot when you split the trends below

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

        print(f'Time-line {i+1}/{sample_pr_id.shape[0]} in the works (estimation)...') 
        clear_output(wait=True)

        X.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None])
        y.set_value(np.log(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values)

        y_ = gp.marginal_likelihood(f'y_{i}', X=X, y=y, noise= σ)
    


with model:
    print('Starting trace:\n')
    trace = pm.sample(1000, chains=4, cores=4)


mp = {'ℓ' : np.array([pm.summary(trace)['mean'].iloc[0]]),  
      'η': np.array([pm.summary(trace)['mean'].iloc[1]]), 
      'σ' : np.array([pm.summary(trace)['mean'].iloc[2]]),
      'ℓ_sd' : np.array([pm.summary(trace)['sd'].iloc[0]]),  
      'η_sd': np.array([pm.summary(trace)['sd'].iloc[1]]), 
      'σ_sd' : np.array([pm.summary(trace)['sd'].iloc[2]]),
      }


# this have worked before...
file_name = "/home/projects/ku_00017/data/generated/currents/shared_tt_Exp_Mar32_sb_trace.pkl"
output = open(file_name, 'wb') 
pickle.dump(trace, output)
output.close()

file_name_mp = "/home/projects/ku_00017/data/generated/currents/shared_tt_Exp_Mat32_sb_mp.pkl"
output = open(file_name_mp, 'wb') 
pickle.dump(mp, output)
output.close()

# correct trace pickle: https://stackoverflow.com/questions/44764932/can-a-pymc3-trace-be-loaded-and-values-accessed-without-the-original-model-in-me#44768217
#file_name_model = "/home/projects/ku_00017/data/generated/currents/tt_Exp_Mat32_sb_model.pkl"
#with open(file_name_model, 'wb') as buff:
#    pickle.dump({'model': model, 'trace': trace}, buff)
