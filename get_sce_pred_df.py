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
from utils import sample_conflict_timeline

import pymc3 as pm
import theano
import theano.tensor as tt

import warnings
warnings.simplefilter("ignore", UserWarning)

# Start timer
start_time = time.time()

# get df:
#path = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/'
path = '/home/projects/ku_00017/data/generated/currents/'
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)
print('got df')

# get trian/test id
train_id, val_id = test_val_train(df)
print('got split')

# get pkl mp
#pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/sce_mp.pkl', 'rb')
path = '/home/projects/ku_00017/data/generated/currents/sce_mp.pkl'
sce_mp = pickle.load(pkl_file)
pkl_file.close()
sce_mp

print(f"got mp: ℓ:{sce_mp['ℓ']}, η:{sce_mp['η']}, σ:{sce_mp['σ']}")

# get hps and run gp
η_beta, ℓ_beta, ℓ_alpha, σ_beta = get_spatial_hps(plot = False)

# Containers
mu_list = []
var_list = []
month_list = []
y_list = []
X_list = []
xcoord_list = []
ycoord_list= []
log_best_list= []

# minimum number of conf in one year for timeslines to be used to est hyper parameters
nnz = 60

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

    # GP
    gp = pm.gp.MarginalSparse(mean_func=mean ,cov_func=cov)
 
    # always prudent
    #df_sorted = df.sort_values(['pg_id', 'month_id'])

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
        print(f'{i+1}/{n} (predicting..)', end='\r')       

        y.set_value(np.log(df[(df['id'].isin(train_id)) & (df['month_id'] == j)]['ged_best_sb'].values + 1))
        X.set_value(df[(df['id'].isin(train_id))  & (df['month_id'] == j)][['xcoord','ycoord']].values)
        Xu.set_value(df[df['month_id'] == j].sort_values('ged_best_sb', ascending = False)[:nnz][['xcoord','ycoord']].values) # could also just be all non-zeroes that month
 
        y_ = gp.marginal_likelihood(f"y_{i}", X=X, Xu = Xu, y=y, noise= σ)

        mu, var = gp.predict(X, point=sce_mp, given = {'gp' : gp, 'X' : X, 'y' : y, 'noise' : σ }, diag=True)
    
        mu_list.append(mu)
        var_list.append(var)
        month_list.append([j] * mu.shape[0])
        xcoord_list.append(X.get_value()[:,0])
        ycoord_list.append(X.get_value()[:,1])
        log_best_list.append(y.get_value())

print('Done predicting \nCreating sce_pred_df')
mu_col = np.array(mu_list).reshape(-1,)
var_col = np.array(var_list).reshape(-1,)
month_col = np.array(month_list).reshape(-1,)
xcoord_col = np.array(xcoord_list).reshape(-1,)
ycoord_col = np.array(ycoord_list).reshape(-1,)
log_best_col = np.array(log_best_list).reshape(-1,)

sce_pred_df = pd.DataFrame({'mu': mu_col, 'var':  var_col, 'month_id': month_col, 'xcoord': xcoord_col, 'ycoord': ycoord_col, 'log_best': log_best_col})

print('Pickling..')
new_file_name = '/home/projects/ku_00017/data/generated/currents/sce_pred_df.pkl'
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/sce_pred_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(sce_pred_df, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string)