
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
from utils import predict_ot
from utils import plot_predictions
from utils import get_mse_ot
from utils import get_metrics_ot

import pymc3 as pm
import theano

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics

import warnings
warnings.simplefilter("ignore", UserWarning)

#import theano
#import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

# dict for the dfs/dicts holding the results
out_dict = {}

# minimum number of conf in timeslines predicted. C = 0 for full run
C_pred = 100#1 # 100

# minimum number of conf in timeslines used to est hyper parameters
C_est = 100#32 #100

# conflict type. Som might need lower c_est than 100 to work
conf_type = 'ged_best_sb' #['ged_best_sb', 'ged_best_ns', 'ged_best_os', 'ged_best']

# Start timer
start_time = time.time()

# get df:
path = '/home/projects/ku_00017/data/generated/currents' 
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)

# get train and validation id:
train_id, val_id = test_val_train(df)

print(f"{C_est}_{C_pred}_{conf_type}_{s_kernel}\n")

# Priors
η_beta = 2
ℓ_beta = 0.2
ℓ_alpha = 8
σ_beta = 5

with pm.Model() as model:

    # trend
    ℓ = pm.Gamma("ℓ", alpha=ℓ_alpha , beta=ℓ_beta)
    η = pm.HalfCauchy("η", beta=η_beta)
    cov = η **2 * pm.gp.cov.ExpQuad(1, ℓ) # Cov func.

    # noise model
    σ = pm.HalfCauchy("σ", beta=σ_beta)

    # mean func. (constant) 
    mean =  pm.gp.mean.Zero()# placeholder if you do individuel

    # GP
    gp = pm.gp.Marginal(mean_func = mean, cov_func=cov)

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
    
    mp = pm.find_MAP()

map_df = pd.DataFrame({"Parameter": ["ℓ", "η", "σ"],"Value at MAP": [float(mp["ℓ"]), float(mp["η"]), float(mp["σ"])]}) 


# Getting the predictions and merging with original df:
df_new = predict_ot(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, mp = mp, gp = gp, σ=σ, C=C_pred)

df_merged = pd.merge(df_new, df[['id', 'pg_id','year','gwcode', 'xcoord', 'ycoord','ged_best_sb','ged_best_ns', 'ged_best_os', 'ged_best']], how = 'left', on = ['id', 'pg_id'])


# getting mse results:
print('Getting MSE')
mse_resutls_df = get_mse_ot(df_merged = df_merged, train_id = train_id, test_id = val_id)

# Creating devrivatives:
print('Creating devrivatives')
df_merged.sort_values(['pg_id', 'X'], inplace= True)
df_merged['mu_slope'] = df_merged.groupby('pg_id')['mu'].transform(np.gradient)
df_merged['mu_acc'] = df_merged.groupby('pg_id')['mu_slope'].transform(np.gradient)
df_merged['mu_mass'] = df_merged.groupby('pg_id')['mu'].transform(np.cumsum)


# Get classification results
print('Getting classifcation results')
df_results = get_metrics_ot(df_merged = df_merged, train_id = train_id, test_id = val_id)

# "filing" names
print('Saving..')
pre_script_map_df = f'{C_est}_{C_pred}_{conf_type}_{s_kernel}_map_df'
pre_script_mse_resutls_df = f'{C_est}_{C_pred}_{conf_type}_{s_kernel}_mse_results_df'
pre_script_df_results = f'{C_est}_{C_pred}_{conf_type}_{s_kernel}_df_results'
pre_script_df = f'{C_est}_{C_pred}_{conf_type}_{s_kernel}_df_merged'

# Save in the eksperiments_dict
out_dict[pre_script_map_df] = map_df
out_dict[pre_script_mse_resutls_df] = mse_resutls_df
out_dict[pre_script_df_results] = df_results
out_dict[pre_script_df] = df_merged

new_file_name = '/home/projects/ku_00017/data/generated/currents/shared_ot_Exp_sb_dict.pkl'
output = open(new_file_name, 'wb')
pickle.dump(out_dict, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)
