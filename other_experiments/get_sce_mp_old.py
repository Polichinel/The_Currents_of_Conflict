import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import time

from utils import get_views_coord
from utils import test_val_train
from utils import get_spatial_hps

import pymc3 as pm
import theano
import theano.tensor as tt

import warnings
warnings.simplefilter("ignore", UserWarning)

# Start timer
start_time = time.time()

# the model
def get_spatial_mp(month_id, nnz, η_beta, ℓ_beta, ℓ_alpha, σ_beta):
    
    try:

        with pm.Model() as model:

            # trend
            ℓ = pm.Gamma("ℓ", alpha=ℓ_alpha , beta=ℓ_beta)
            η = pm.HalfCauchy("η", beta=η_beta)
            cov = η **2 * pm.gp.cov.ExpQuad(2, ℓ) # Cov func.

            # noise model
            σ = pm.HalfCauchy("σ", beta=σ_beta)

            # mean func. (constant) 
            mean =  pm.gp.mean.Zero()

            # sample and split X,y ---------------------------------------------  
            non_zeroes = df.groupby('pg_id').sum().sort_values('ged_best_sb', ascending = False).index[:nnz]

            y = np.log(df[(df['id'].isin(train_id)) & (df['month_id'] == month_id)]['ged_best_sb'].values + 1)
            X = df[(df['id'].isin(train_id))  & (df['month_id'] == month_id)][['xcoord','ycoord']].values
            Xu = df[(df['id'].isin(train_id)) & (df['pg_id'].isin(non_zeroes)) & (df['month_id'] == month_id)][['xcoord','ycoord']].values

            # The GP
            gp = pm.gp.MarginalSparse(cov_func=cov) #, approx="DTC") # ved ikke om det gør en forskel til FICT
            y = gp.marginal_likelihood("y", X=X, Xu = Xu, y=y, noise= σ)

            mp = pm.find_MAP()

            return(mp)
    
    except:
        pass


# get df:
path = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/'
#path = '/home/projects/ku_00017/data/generated/currents' 
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)
print('got df \n')

# get trian/test id
train_id, val_id = test_val_train(df)

# the loop
η_beta, ℓ_beta, ℓ_alpha, σ_beta = get_spatial_hps(plot = False)
nnz = 60# 100% arbitrary
month_ids = df[df['id'].isin(train_id)]['month_id'].unique()
n = len(month_ids)

ℓ_list = []
η_list = []
σ_list = []
m_list  = []

for i, j in enumerate(month_ids):
    
    print(f'{i}/{n}')
    mp = get_spatial_mp(j, nnz, η_beta, ℓ_beta, ℓ_alpha, σ_beta)

    if mp != None:
    
        ℓ_list.append(mp['ℓ'])
        η_list.append(mp['η'])
        σ_list.append(mp['σ'])
        m_list.append(j)
    
    else:
        pass

mp = {'ℓ_log__': np.log(np.mean(ℓ_list)),
 'η_log__': np.log(np.mean(η_list)),
 'σ_log__': np.log(np.mean(σ_list)),
 'ℓ': np.mean(ℓ_list),
 'η': np.mean(η_list),
 'σ': np.mean(σ_list)}

mp_df = pd.DataFrame({'month_id': m_list ,'ℓ':  ℓ_list, 'η': η_list, 'σ': σ_list})

mp_dict = {'mp': mp, 'mp_df' : mp_df}

# Save .pkl
new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/sce_mp_dict.pkl'
#new_file_name = '/home/projects/ku_00017/data/generated/currents/sce_mp_dict.pkl'
output = open(new_file_name, 'wb')
pickle.dump(mp_dict, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string) 