import os
import numpy as np
import pandas as pd
import pymc3 as pm
import theano

import pickle

import warnings
warnings.simplefilter("ignore", UserWarning)


def get_views_coord(path, file_name):

    """Get views data with coords and return it as a pandas dataframe"""

    file_path = os.path.join(path, file_name)
    views_coord = pd.read_pickle(file_path)

    return(views_coord)


def test_val_train(df, info = True, test_time = False):

    """For train, validation and test. In accordance with Hegre et al. 2019 p. 163"""

    df_sorted = df.sort_values('month_id') 

    if test_time == False:

        train_id = df_sorted[(df_sorted['year'] > 1989) & (df_sorted['year'] <= 2011) ]['id'].values
        val_id = df_sorted[(df_sorted['year'] > 2011) & (df_sorted['year'] <= 2014) ]['id'].values
        test_id = df_sorted[(df_sorted['year'] > 2014) & (df_sorted['year'] <= 2017) ]['id'].values
        
        if info == True:

            train_start_year = df_sorted[df_sorted['id'].isin(train_id)]['year'].min()
            train_end_year = df_sorted[df_sorted['id'].isin(train_id)]['year'].max()
            
            train_start_month = df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['year'] == train_start_year)]['month'].min()
            train_end_month = df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['year'] == train_end_year)]['month'].max()
            n_train = df_sorted[df_sorted['id'].isin(train_id)]['month_id'].unique().shape[0]

            val_start_year = df_sorted[df_sorted['id'].isin(val_id)]['year'].min()
            val_end_year = df_sorted[df_sorted['id'].isin(val_id)]['year'].max()
            
            val_start_month = df_sorted[(df_sorted['id'].isin(val_id)) & (df_sorted['year'] == val_start_year)]['month'].min()
            val_end_month = df_sorted[(df_sorted['id'].isin(val_id)) & (df_sorted['year'] == val_end_year)]['month'].max()
            n_val = df_sorted[df_sorted['id'].isin(val_id)]['month_id'].unique().shape[0]

            test_start_year = df_sorted[df_sorted['id'].isin(test_id)]['year'].min()
            test_end_year = df_sorted[df_sorted['id'].isin(test_id)]['year'].max()
            
            test_start_month = df_sorted[(df_sorted['id'].isin(test_id)) & (df_sorted['year'] == test_start_year)]['month'].min()
            test_end_month = df_sorted[(df_sorted['id'].isin(test_id)) & (df_sorted['year'] == test_end_year)]['month'].max()
            n_test = df_sorted[df_sorted['id'].isin(test_id)]['month_id'].unique().shape[0]


            string1 = f'Train from {train_start_month}/{train_start_year} trough {train_end_month}/{train_end_year} ({n_train})\n'
            string2 = f'Val from {val_start_month}/{val_start_year} trough {val_end_month}/{val_end_year} ({n_val})\n'
            string3 = f'Test time from {test_start_month}/{test_start_year} trough {test_end_month}/{test_end_year} ({n_test})\n'
            string4 = f'(Test=False, so test set not outputted)\n'
            print(string1 + string2 + string3 + string4)
        
        return(train_id, val_id)


    if test_time == True:

        train_id = df_sorted[(df_sorted['year'] > 1989) & (df_sorted['year'] <= 2014) ]['id'].values
        test_id = df_sorted[(df_sorted['year'] > 2014) & (df_sorted['year'] <= 2017) ]['id'].values 
        
        if info == True:

            train_start_year = df_sorted[df_sorted['id'].isin(train_id)]['year'].min()
            train_end_year = df_sorted[df_sorted['id'].isin(train_id)]['year'].max()
            
            train_start_month = df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['year'] == train_start_year)]['month'].min()
            train_end_month = df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['year'] == train_end_year)]['month'].max()
            n_train = df_sorted[df_sorted['id'].isin(train_id)]['month_id'].unique().shape[0]

            test_start_year = df_sorted[df_sorted['id'].isin(test_id)]['year'].min()
            test_end_year = df_sorted[df_sorted['id'].isin(test_id)]['year'].max()
            
            test_start_month = df_sorted[(df_sorted['id'].isin(test_id)) & (df_sorted['year'] == test_start_year)]['month'].min()
            test_end_month = df_sorted[(df_sorted['id'].isin(test_id)) & (df_sorted['year'] == test_end_year)]['month'].max()
            n_test = df_sorted[df_sorted['id'].isin(test_id)]['month_id'].unique().shape[0]

            string1 = f'Train from {train_start_month}/{train_start_year} trough {train_end_month}/{train_end_year} ({n_train})\n'
            string2 = f'Test time from {test_start_month}/{test_start_year} trough {test_end_month}/{test_end_year} ({n_test})\n'
            string3 = f'(Test=True, so val set neither genereted or outputted)\n'
            print(string1 + string2 + string3)
        
        return(train_id, test_id)


def sample_conflict_timeline(conf_type, df, train_id, test_id, C=12):

    """This function samples N time-lines contining C >= conflicts in at least one year.
    Default C = 12, so that is one year with a conflict each day."""

    #Set the dummy corrospoding to the conflcit type
    conf_type = 'ged_best_sb'
    dummy = 'ged_dummy_sb'

    # sort the df - just in case
    df_sorted = df.sort_values(['pg_id', 'month_id'])
   
    df_sum = df_sorted.groupby(['pg_id', 'year']).sum()[[dummy]].reset_index()
    sample_pr_id = df_sum[df_sum[dummy] >= C]['pg_id'].unique()

    return(sample_pr_id)



def get_hyper_priors(plot = True, η_beta_s = 0.5, ℓ_beta_s = 0.8, ℓ_alpha_s = 2, α_alpha_s = 5, α_beta_s = 1, η_beta_l = 4, ℓ_beta_l = 1, ℓ_alpha_l = 36, σ_beta = 5):

    """Get hyper prior dict, an potntially plot"""


    #hyper_priors_dict
    hps = {}

    # short term priors
    hps['η_beta_s'] =  η_beta_s
    hps['ℓ_beta_s'] = ℓ_beta_s
    hps['ℓ_alpha_s'] = ℓ_alpha_s

    hps['α_alpha_s'] = α_alpha_s #  for Rational Quadratic Kernel. Ignore for Quad or Matern
    hps['α_beta_s'] = α_beta_s # for Rational Quadratic Kernel. Ignore for Quad or Matern

    # long term priors
    hps['η_beta_l'] = η_beta_l
    hps['ℓ_beta_l'] = ℓ_beta_l
    hps['ℓ_alpha_l'] = ℓ_alpha_l

    # noise prior
    hps['σ_beta'] = σ_beta

    return(hps)


def predict(conf_type, df, train_id, test_id, mp, gp, gp_s, gp_l, σ, C, indv_mean = False):

    """This function takes the mp, gps and σ for a two-trend implimentation.
    it also needs the df, the train ids and the val/test ids.
    It outpust a pandas daframe with X, y (train/test) along w/ mu and var.
    We get mu and var over all X and for both full gp, long trend gp and short trand gp.
    C denotes the number of minimum conlflict in timelines and is just for testing.
    I a full run set C = 0."""

    new_id = np.append(train_id, test_id)
    df_sorted = df.sort_values(['pg_id', 'month_id'])
    X_new = df_sorted[df_sorted['id'].isin(new_id) ]['month_id'].unique()[:,None] # all X

    sample_pg_id = sample_conflict_timeline(conf_type = conf_type, df = df, train_id = train_id, test_id = test_id, C = C)

    train_len = df_sorted[df_sorted['id'].isin(train_id)]['month_id'].unique().shape[0]#test
    test_len = df_sorted[df_sorted['id'].isin(test_id)]['month_id'].unique().shape[0]#test
    X = theano.shared(np.zeros(train_len)[:,None], 'X')#test
    y = theano.shared(np.zeros(train_len), 'y')#test

    # make lists
    mu_list = []
    mu_s_list = []
    mu_l_list = []
    var_list = []
    var_s_list = []
    var_l_list = []
    X_new_list = []
    #y_new_list = []
    idx_list = []
    pg_idx_list = []
    train_list = []

    # Loop gp predict over time lines
    for i, j in enumerate(sample_pg_id):

        print(f'Time-line {i+1}/{sample_pg_id.shape[0]} in the works (prediction)...', end = '\r')

        idx = df_sorted[(df_sorted['id'].isin(new_id)) & (df_sorted['pg_id'] == j)]['id'].values
        #y_new = df_sorted[(df_sorted['id'].isin(new_id)) & (df_sorted['pg_id'] == j)][conf_type].values



        X.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None])
        y.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type].values)

        mu, var = gp.predict(X_new, point=mp, given = {'gp' : gp, 'X' : X, 'y' : y, 'noise' : σ}, diag=True)
        mu_s, var_s = gp_s.predict(X_new, point=mp, given = {'gp' : gp, 'X' : X, 'y' : y, 'noise' : σ}, diag=True)
        mu_l, var_l = gp_l.predict(X_new, point=mp, given = {'gp' : gp, 'X' : X, 'y' : y, 'noise' : σ}, diag=True)

        mu_list.append(mu)
        mu_s_list.append(mu_s)
        mu_l_list.append(mu_l)
        var_list.append(var)
        var_s_list.append(var_s)
        var_l_list.append(var_l)
        X_new_list.append(X_new)
        #y_new_list.append(y_new)
        idx_list.append(idx)
        pg_idx_list.append([j] * mu.shape[0])
        train_list.append(np.array([1] * train_len + [0] * test_len)) # dummy for training...

    mu_col = np.array(mu_list).reshape(-1,) 
    mu_s_col = np.array(mu_s_list).reshape(-1,) 
    mu_l_col = np.array(mu_l_list).reshape(-1,)
    var_col = np.array(var_list).reshape(-1,) 
    var_s_col = np.array(var_s_list).reshape(-1,) 
    var_l_col = np.array(var_l_list).reshape(-1,) 
    X_new_col = np.array(X_new_list).reshape(-1,) 
    #y_new_col = np.array(y_new_list).reshape(-1,)     
    idx_col = np.array(idx_list).reshape(-1,)    
    pg_idx_col = np.array(pg_idx_list).reshape(-1,)
    train_col =  np.array(train_list).reshape(-1,)

    df_new = pd.DataFrame({
                           'mu': mu_col, 'mu_s' : mu_s_col, 'mu_l' : mu_l_col,
                            'var' : var_col, 'var_s' : var_s_col, 'var_l' : var_l_col, 
                            'X' : X_new_col, 
                            'id' : idx_col, 'pg_id' : pg_idx_col, 'train' : train_col
                            }) #  'y' : y_new_col ,

    return(df_new)
