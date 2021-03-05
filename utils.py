import os
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

def get_views_coord():

    """Get views data with coords and return it as a pandas dataframe"""

    path = '/home/polichinel/Documents/Articles/conflict_prediction/data/ViEWS/'
    file_name = 'ViEWS_coord.pkl'

    file_path = os.path.join(path, file_name)
    views_coord = pd.read_pickle(file_path)

    return(views_coord)



def test_val_train(df, info = True, test_time = False):

    """For train, validation and test. In accordance with Hegre et al. 2019 p. 163"""

    #Train: jan 1990 = month_id 121 (12) - dec 2011 = month_id 384 (275)
    #Val: jan 2012 = month_id 385 (276)- dec 2014 = month_id 420 (311) # hvorfor kun 35? 
    #Test: jan 2015 = month_id 421 (312) - dec 2017 = month_id 456 (347) # hvorfor kun 35?

    df_sorted = df.sort_values('month_id') # actually might be better to just sort after id.

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



def sample_conflict_timeline(df, train_id, test_id, C=5, N=3, demean = False, seed = 42, get_index = False):

    """ This function samples N time-lines contining c>=C conflicts. 
    If N == None, it gets all time-lines with c>C conflicts.
    As default it will try to get the val_id. Error will come if it does not exits"""

    # sort the df - just in case
    df_sorted = df.sort_values('month_id')

    # groupby gids and get total events
    df_sb_total_events = df.groupby(['pg_id']).sum()['ged_dummy_sb'].reset_index().rename(columns = {'ged_dummy_sb':'ged_total_events_sb'})
    
    if N == None:
        sample_gids = df_sb_total_events[df_sb_total_events['ged_total_events_sb'] >= C]['pg_id'].values
        N = len(sample_gids)

    else:
        # sample one gid from all gids from timeline with c>C events
        sample_gids = df_sb_total_events[df_sb_total_events['ged_total_events_sb'] > C]['pg_id'].sample(N, random_state = seed).values

    train_mask = df_sorted['pg_id'].isin(sample_gids) & df_sorted['id'].isin(train_id) 
    test_mask = df_sorted['pg_id'].isin(sample_gids) & df_sorted['id'].isin(test_id)
    
    y = np.log(df_sorted[train_mask]['ged_best_sb'] +1).values.reshape(-1,N)
    X = df_sorted[train_mask]['month_id'].values.reshape(-1,N)

    y_test = np.log(df_sorted[test_mask]['ged_best_sb'] +1).values.reshape(-1,N)
    X_test = df_sorted[test_mask]['month_id'].values.reshape(-1,N)

    if demean == True:

        y = y - y.mean(axis = 0) # you can't demean testset right?

    print(f'\nX: {X.shape}, y: {y.shape} \nX (val/test): {X_test.shape}, y (val/test): {y_test.shape} \n')

    if get_index == False:
        return(X, y, X_test, y_test)


    if get_index == True:

        idx = df_sorted[train_mask]['id'].values.reshape(-1,N)
        idx_test = df_sorted[test_mask]['id'].values.reshape(-1,N)

        return(X, y, X_test, y_test, idx, idx_test)



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

    if plot == True:

        # plot:
        grid = np.linspace(0,64,1000)
            
        priors = [
            ('η_prior_s', pm.HalfCauchy.dist(beta=hps['η_beta_s'])),
            ('ℓ_prior_s', pm.Gamma.dist(alpha=hps['ℓ_alpha_s'] , beta=hps['ℓ_beta_s'])),
            ('α_prior_s', pm.Gamma.dist(alpha=hps['α_alpha_s'], beta= hps['α_beta_s'])),
            ('η_prior_l', pm.HalfCauchy.dist(beta=hps['η_beta_l'])),
            ('ℓ_prior_l', pm.Gamma.dist(alpha=hps['ℓ_alpha_l'] , beta=hps['ℓ_beta_l'])),
            ('σ', pm.HalfCauchy.dist(beta=hps['σ_beta']))]

        plt.figure(figsize= [15,5])
        plt.title('hyper-priors')

        for i, prior in enumerate(priors):
            plt.plot(grid, np.exp(prior[1].logp(grid).eval()), label = prior[0])

        plt.legend()
        plt.show()

    return(hps)