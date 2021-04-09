# libs:
import os
import numpy as np
import pandas as pd

import pickle
import time

from utils_sce import get_views_coord
from utils_sce import test_val_train
from utils_sce import sample_conflict_timeline
from utils_sce import get_spatial_hps

import pymc3 as pm
import theano

import warnings
warnings.simplefilter("ignore", UserWarning)

# Start timer
start_time = time.time()

# get df:
#path = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/'
path = '/home/projects/ku_00017/data/generated/currents' 
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)
print('Got df')

# get train and validation id:
train_id, val_id = test_val_train(df, test_time= False)
print("Got train/val index")

# Constuction the gps and getting the map
η_beta, ℓ_beta, ℓ_alpha, σ_beta = get_spatial_hps(plot = False)
print("Got hyper priors")

# -------------------------------------------------------------------------------------------------------------

# minimum number of conf in one year for timeslines to be used to est hyper parameters
#C_est = 8

# the n most conflict experiencing cells that month.
nnz = 65 # 100% arbitrary

# conflict type.
conf_type = 'ged_best_sb'

print(f'{nnz}_{conf_type}')

# given new results it might actually be two trend...
with pm.Model() as model:

    # Hyper priors
    ℓ = pm.Gamma("ℓ", alpha=ℓ_alpha , beta=ℓ_beta)
    η = pm.HalfCauchy("η", beta=η_beta)
    cov = η **2 * pm.gp.cov.Matern32(2, ℓ) # Cov func.

    # noise model
    σ = pm.HalfCauchy("σ", beta=σ_beta)

    # mean func. (constant) 
    mean =  pm.gp.mean.Zero()

    # sample and split X,y ---------------------------------------------  
    #sample_pr_id = sample_conflict_timeline(conf_type = conf_type, df = df, train_id = train_id, test_id = val_id, C = C_est)

    # GP
    gp = pm.gp.MarginalSparse(mean_func=mean ,cov_func=cov)
 
    # always prudent
    #df_sorted = df.sort_values(['pg_id', 'month_id'])

    #shared
    coord_len = df.groupby(['xcoord', 'ycoord']).sum().shape[0]
    y = theano.shared(np.zeros(coord_len), 'y')
    X = theano.shared(np.zeros([coord_len, 2]), 'X')

    # this does not vary here:
    #Xu = theano.shared(df[(df['id'].isin(sample_pr_id))][['xcoord','ycoord']].values, 'Xu')
    Xu = theano.shared(np.zeros([nnz, 2]), 'Xu')

    # loop
    month_ids = df[df['id'].isin(train_id)]['month_id'].unique()
    n = month_ids.shape[0]

    for i, j in enumerate(month_ids):
        print(f'{i+1}/{n} (estimation)', end='\r')       

        y.set_value(np.log(df[(df['id'].isin(train_id)) & (df['month_id'] == j)]['ged_best_sb'].values + 1))
        X.set_value(df[(df['id'].isin(train_id))  & (df['month_id'] == j)][['xcoord','ycoord']].values)
        Xu.set_value(df[df['month_id'] == j].sort_values('ged_best_sb', ascending = False)[:nnz][['xcoord','ycoord']].values) 

        #Xu.set_value(df[(df['pg_id'].isin(sample_pr_id)) & (df['month_id'] == j)][['xcoord','ycoord']].values)
 
        y_ = gp.marginal_likelihood(f"y_{i}", X=X, Xu = Xu, y=y, noise= σ)

    mp = pm.find_MAP()
print('Got mp')


print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/sce_mp_nnz65.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/sce_mp.pkl'
output = open(new_file_name, 'wb')
pickle.dump(mp, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)