# %%
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

theano.gof.cc.get_module_cache().clear()

# %%
# dict for the dfs/dicts holding the results
two_trend_ExpQuad_dict = {}

# minimum number of conf in timeslines predicted. C = 0 for full run
#C_pred = 32 
C_pred = 0

N=None # if you want to sample a subset of the time lines drawn given c_est/C_pred
seed = 42 # the random seed used if you set N != None
dem = False # not sure this is in use..

# minimum number of conf in timeslines used to est hyper parameters
C_est = 32

# conflict type. Som might need lower c_est than 100 to work
conf_type = 'ged_best_sb' #['ged_best_sb', 'ged_best_ns', 'ged_best_os', 'ged_best']

# short term kernel
s_kernel = 'ExpQuad' #['ExpQuad', 'RatQuad', 'Matern32'] #, 'Matern52']


# %%

print(f"{C_est}_{conf_type}_{s_kernel}\n")

# Start timer
start_time = time.time()

# get df:
# path = '/home/polichinel/Documents/Articles/conflict_prediction/data/ViEWS/'
path = '/home/projects/ku_00017/data/generated/currents' 
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)

# get train and validation id:
train_id, val_id = test_val_train(df)

# get train (train + val) and test id:
# train_id, test_id = test_val_train(df, test_time = True)

# if you want to plot the hyper priors
# hyper_priors_dict = get_hyper_priors(plot = True)

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
    X, y, _, _ = sample_conflict_timeline(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, C = C_est, N = None)

    # Full GP ----------------------------------------------------------
    gp = gp_s + gp_l

    # sample:
    for i in range(y.shape[1]):

        # Theano shared variables: maybe helps with the memory leak
        X_shared = theano.shared(X[:,i][:,None]) #, borrow=True)
        y_shared = theano.shared(y[:,i]) #, borrow=True)

        print(f'Time-line {i+1}/{y.shape[1]} in the works (estimation)...') 
        clear_output(wait=True)

        y_ = gp.marginal_likelihood(f'y_{i}', X=X_shared, y=y_shared, noise= σ)

        theano.gof.cc.get_module_cache().clear() # it is a test...
    
        # with shared you might give it the same y?            
    mp = pm.find_MAP()

# get alpha if you used the Rational Quadratic kernel for short term
if s_kernel == 'RatQuad':

    map_df = pd.DataFrame({
        "Parameter": ["ℓ_s", "η_s", "α_s", "ℓ_l", "η_l", "σ"],
        "Value at MAP": [float(mp["ℓ_s"]), float(mp["η_s"]), float(mp["α_s"]), float(mp["ℓ_l"]), float(mp["η_l"]), float(mp["σ"])]
        })

# if Exponentiated Quadratic og Matern kernel were used ignore alpha
else:

    map_df = pd.DataFrame({
        "Parameter": ["ℓ_s", "η_s", "ℓ_l", "η_l", "σ"],
        "Value at MAP": [float(mp["ℓ_s"]), float(mp["η_s"]), float(mp["ℓ_l"]), float(mp["η_l"]), float(mp["σ"])]
        }) 

# Getting the predictions and merging with original df:
df_new = predict(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, mp = mp, gp = gp, gp_s = gp_s, gp_l = gp_l, σ=σ, C=C_pred)

df_merged = pd.merge(df_new, df[['id', 'pg_id','year','gwcode', 'xcoord', 'ycoord','ged_best_sb','ged_best_ns', 'ged_best_os', 'ged_best']], how = 'left', on = 'id') 

# plot sanity check, but not if you have many time lines
# plot_predictions(df_merged = df_merged)

# getting mse results:
mse_resutls_df, mse_dict = get_mse(df_merged = df_merged, train_id = train_id, test_id = val_id)

# Creating devrivatives:
df_merged.sort_values(['pg_id', 'X'], inplace= True)
df_merged['mu_l_slope'] = df_merged.groupby('pg_id')['mu_l'].transform(np.gradient)
df_merged['mu_l_acc'] = df_merged.groupby('pg_id')['mu_l_slope'].transform(np.gradient)
df_merged['mu_l_mass'] = df_merged.groupby('pg_id')['mu_l'].transform(np.cumsum)
            
# Get classification results
df_results = get_metrics(df_merged = df_merged, train_id = train_id, test_id = val_id)


# end timer
final_time = time.time()
final_run_time = final_time - start_time

string = f'Run for {final_run_time/60:.3} minutes'
print(string)

# "filing" names
pre_script_map_df = f'{C_est}_{conf_type}_{s_kernel}_map_df'
pre_script_mse_resutls_df = f'{C_est}_{conf_type}_{s_kernel}_mse_results_df'
pre_script_mse_dict = f'{C_est}_{conf_type}_{s_kernel}_mse_dict'
pre_script_df_results = f'{C_est}_{conf_type}_{s_kernel}_df_results'

# Save in the eksperiments_dict
two_trend_ExpQuad_dict[pre_script_map_df] = map_df
two_trend_ExpQuad_dict[pre_script_mse_resutls_df] = mse_resutls_df
two_trend_ExpQuad_dict[pre_script_mse_dict] = mse_dict
two_trend_ExpQuad_dict[pre_script_df_results] = df_results
            

# %%
#new_file_name = "/home/polichinel/Documents/Articles/conflict_prediction/data/ViEWS/two_trend_ExpQuad_dict.pkl"
new_file_name = '/home/projects/ku_00017/data/generated/currents/two_trend_ExpQuad_dict.pkl'
output = open(new_file_name, 'wb')
pickle.dump(two_trend_ExpQuad_dict, output)
output.close()
