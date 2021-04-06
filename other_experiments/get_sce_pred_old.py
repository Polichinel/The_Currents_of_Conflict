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

def get_spatial_gp(ℓ_alpha, ℓ_beta, η_beta, σ_beta):

    #month_id = 121#384 # last month in train doing val - should not be that important here.
    #nnz = 100 #60 # should be better!

    with pm.Model() as model:

        # trend
        ℓ = pm.Gamma("ℓ", alpha=ℓ_alpha , beta=ℓ_beta)
        η = pm.HalfCauchy("η", beta=η_beta)
        cov = η **2 * pm.gp.cov.Matern32(2, ℓ) # Cov func.

        # noise model
        σ = pm.HalfCauchy("σ", beta=σ_beta)

        # mean func. (constant) 
        mean =  pm.gp.mean.Zero()
        
        # The GP
        # gp = pm.gp.MarginalSparse(mean_func = mean ,cov_func=cov) #, approx="DTC") # ved ikke om det gør en forskel til FICT
        gp = pm.gp.Marginal(mean_func = mean ,cov_func=cov) #, approx="DTC") # ved ikke om det gør en forskel til FICT

        y_len = df.groupby(['xcoord', 'ycoord']).sum().shape[0]
        X_len = df.groupby(['xcoord', 'ycoord']).sum().shape[0]
 
        y = theano.shared(np.zeros(y_len), 'y')
        X = theano.shared(np.zeros([X_len, 2]), 'X')
        #Xu = theano.shared(np.zeros([nnz, 2]), 'X')

        # loop
        month_ids = df[df['id'].isin(train_id)]['month_id'].unique()[:5] # note!!!
        n = len(month_ids)

        for i, j in enumerate(month_ids):
            print(f'{i}/{n} (estimation)', end='\r')       


            y.set_value(np.log(df[(df['id'].isin(train_id)) & (df['month_id'] == j)]['ged_best_sb'].values + 1))
            X.set_value(df[(df['id'].isin(train_id)) & (df['month_id'] == j)][['xcoord','ycoord']].values)

            # Non zeroes ---------------------------------------------  
            #non_zeroes = df[(df['id'].isin(train_id)) & (df['month_id'] == j)].sort_values('ged_best_sb', ascending = False).index[:nnz]
            #Xu.set_value(df[(df['id'].isin(train_id)) & (df['pg_id'].isin(non_zeroes)) & (df['month_id'] == j)][['xcoord','ycoord']].values)
            # The non_zero index should be wbough now...

            #print(f'\n{Xu.get_value().shape}')

            #y_ = gp.marginal_likelihood(f"y_{i}", X=X, Xu = Xu, y=y, noise= σ)
            y_ = gp.marginal_likelihood(f"y_{i}", X=X, y=y, noise= σ)
     
    return(gp, σ)# right now y and X is just for size later on.. 


# get df:
path = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/'
#path = '/home/projects/ku_00017/data/generated/currents/'
file_name = 'ViEWS_coord.pkl'
df = get_views_coord(path = path, file_name = file_name)
print('got df')

# get trian/test id
train_id, val_id = test_val_train(df)
print('got split')

# get pkl mp
pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/sce_mp_dict.pkl', 'rb')
#pkl_file = open('/home/projects/ku_00017/data/generated/currents/sce_mp_dict.pkl', 'rb')
mp_dict = pickle.load(pkl_file)
pkl_file.close()
print('got mp')

mp = mp_dict['mp']
mp_df = mp_dict['mp_df']

# geet hps and run gp
η_beta, ℓ_beta, ℓ_alpha, σ_beta = get_spatial_hps(plot = False)
gp, X, y, σ = get_spatial_gp(ℓ_alpha, ℓ_beta, η_beta, σ_beta) # X and y is just to get the shape.
print('ran gp - starts prediction')

# prediction
month_ids = df[df['id'].isin(train_id)]['month_id'].unique()[:5] #note!!!
n = len(month_ids)

y_len = df.groupby(['xcoord', 'ycoord']).sum().shape[0]
X_len = df.groupby(['xcoord', 'ycoord']).sum().shape[0]
 
y = theano.shared(np.zeros(y_len), 'y')
X = theano.shared(np.zeros([X_len, 2]), 'X')

mu_list = []
var_list = []
month_list = []
y_list = []
X_list = []
xcoord_list = []
ycoord_list= []
log_best_list= []

for i, j in enumerate(month_ids):
    print(f'{i}/{n} (prediction)', end='\r')

    y.set_value(np.log(df[(df['id'].isin(train_id)) & (df['month_id'] == j)]['ged_best_sb'].values + 1))
    X.set_value(df[(df['id'].isin(train_id)) & (df['month_id'] == j)][['xcoord','ycoord']].values)

    mu, var = gp.predict(X, point=mp, given = {'gp' : gp, 'X' : X, 'y' : y, 'noise' : σ }, diag=True)
    
    mu_list.append(mu)
    var_list.append(var)
    month_list.append([j] * mu.shape[0])
    xcoord_list.append(X.get_value()[:,0])
    ycoord_list.append(X.get_value()[:,1])
    log_best_list.append(y.get_value())

mu_col = np.array(mu_list).reshape(-1,)
var_col = np.array(var_list).reshape(-1,)
month_col = np.array(month_list).reshape(-1,)
xcoord_col = np.array(xcoord_list).reshape(-1,)
ycoord_col = np.array(ycoord_list).reshape(-1,)
log_best_col = np.array(log_best_list).reshape(-1,)


sce_df = pd.DataFrame({'mu': mu_col, 'var':  var_col, 'month_id': month_col, 'xcoord': xcoord_col, 'ycoord': ycoord_col, 'log_best': log_best_col})

# Save .pkl
new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/ViEWS/sce_pred_df.pkl'
#new_file_name = '/home/projects/ku_00017/data/generated/currents/sce_pred_df.pkl'
output = open(new_file_name, 'wb')
pickle.dump(sce_df, output)
output.close()

# end timer
final_time = time.time()
final_run_time = final_time - start_time
string = f'Run for {final_run_time/60:.3} minutes'
print(string) 