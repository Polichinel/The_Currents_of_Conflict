# libs:
import os
import numpy as np
import pandas as pd

import pickle
import time

from utils import get_views_coord
from utils import test_val_train
from utils import sample_conflict_timeline
from utils import get_spatial_hps

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
#η_beta, ℓ_beta, ℓ_alpha, σ_beta = get_spatial_hps(plot = False)

η_l_beta = 2
ℓ_l_beta = 1
ℓ_l_alpha = 5

η_s_beta = 6
ℓ_s_beta = 2
ℓ_s_alpha = 2

σ_beta = 1

print("Got hyper priors")

# -------------------------------------------------------------------------------------------------------------

# the n most conflict experiencing cells that month.
nnz = 60 # 100% arbitrary

# conflict type.
conf_type = 'ged_best_sb'

print(f'{nnz}_{conf_type}')

# given new results it might actually be two trend...
with pm.Model() as model:

    # short Hyper priors
    ℓ_s = pm.Gamma("ℓ_s", alpha=ℓ_s_alpha , beta=ℓ_s_beta)
    η_s = pm.HalfCauchy("η_s", beta=η_s_beta)
    cov_s = η_s **2 * pm.gp.cov.ExpQuad(2, ℓ_s) # Cov func.

    # short GP
    gp_s = pm.gp.MarginalSparse(cov_func=cov_s)
    
    # long Hyper priors
    ℓ_l = pm.Gamma("ℓ_l", alpha=ℓ_l_alpha , beta=ℓ_l_beta)
    η_l = pm.HalfCauchy("η_l", beta=η_l_beta)
    cov_l = η_l **2 * pm.gp.cov.Matern32(2, ℓ_l) # Cov func.

    # long GP
    gp_l = pm.gp.MarginalSparse(cov_func=cov_l)
 
    #full GP
    gp = gp_s + gp_l

    # noise model
    σ = pm.HalfCauchy("σ", beta=σ_beta)

    #shared
    coord_len = df.groupby(['xcoord', 'ycoord']).sum().shape[0]
    y = theano.shared(np.zeros(coord_len), 'y')
    X = theano.shared(np.zeros([coord_len, 2]), 'X')

    # this does not vary here:
    Xu = theano.shared(np.zeros([nnz, 2]), 'Xu')

    # loop
    month_ids = df[df['id'].isin(train_id)]['month_id'].unique()
    n = month_ids.shape[0]

    for i, j in enumerate(month_ids):
        print(f'{i+1}/{n} (estimation)', end='\r')       

        y.set_value(np.log(df[(df['id'].isin(train_id)) & (df['month_id'] == j)]['ged_best_sb'].values + 1))
        X.set_value(df[(df['id'].isin(train_id))  & (df['month_id'] == j)][['xcoord','ycoord']].values)
        Xu.set_value(df[df['month_id'] == j].sort_values('ged_best_sb', ascending = False)[:nnz][['xcoord','ycoord']].values) 

        y_ = gp.marginal_likelihood(f"y_{i}", X=X, Xu = Xu, y=y, noise= σ)

    mp = pm.find_MAP()
print('Got mp')


print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/tt_sce_mp.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/tt_sce_mp.pkl'
output = open(new_file_name, 'wb')
pickle.dump(mp, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)