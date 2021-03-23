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
C_est = 8

# conflict type.
conf_type = 'ged_best_sb'

print(f'{C_est}_{conf_type}')

with pm.Model() as model:

    # Hyper priors
    ℓ = pm.Gamma("ℓ", alpha=ℓ_alpha , beta=ℓ_beta)
    η = pm.HalfCauchy("η", beta=η_beta)
    cov = η **2 * pm.gp.cov.ExpQuad(2, ℓ) # Cov func.

    # noise model
    σ = pm.HalfCauchy("σ", beta=σ_beta)

    # mean func. (constant) 
    mean =  pm.gp.mean.Zero()

    # GP
    gp = pm.gp.Marginal(mean_func=mean ,cov_func=cov)
 
    # always prudent:
    df_sorted = df.sort_values(['pg_id', 'month_id'])

    #shared
    coord_len = df.groupby(['xcoord', 'ycoord']).sum().shape[0]
    y = theano.shared(np.zeros(coord_len), 'y')
    X = theano.shared(np.zeros([coord_len, 2]), 'X')
    
    # loop
    month_ids = df[df['id'].isin(train_id)]['month_id'].unique()[:5] # note!!!
    n = len(month_ids)

    for i, j in enumerate(month_ids):
        print(f'{i}/{n} (estimation)', end='\r')       

        y.set_value(np.log(df[(df['id'].isin(train_id)) & (df['month_id'] == j)]['ged_best_sb'].values + 1))
        X.set_value(df[(df['id'].isin(train_id)) & (df['month_id'] == j)][['xcoord','ycoord']].values)

        y_ = gp.marginal_likelihood(f"y_{i}", X=X, y=y, noise= σ)

    mp = pm.find_MAP()
print('Got mp')

# put it into a df for order - can always go back
sce_mp_df = pd.DataFrame({"Parameter": ["ℓ", "η", "σ"],
                       "Value at MAP": [float(mp["ℓ"]), float(mp["η"]), float(mp["σ"])]}) 

print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/sce_mp_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(sce_mp_df, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)