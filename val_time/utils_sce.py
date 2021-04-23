import os
import numpy as np
import pandas as pd
import pymc3 as pm
import theano

import warnings
warnings.simplefilter("ignore", UserWarning)


def get_views_coord(path, file_name):

    """Get views data with coords and return it as a pandas dataframe"""

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


def sample_conflict_timeline(conf_type, df, train_id, test_id, C=12):

    """This function samples N time-lines contining C >= conflicts in at least one year.
    Default C = 12, so that is one year with a conflict each day."""

    #Set the dummy corrospoding to the conflcit type
    if conf_type == 'ged_best_sb':
        dummy = 'ged_dummy_sb'

    elif conf_type == 'ged_best_ns':
        dummy = 'ged_dummy_ns'

    elif conf_type == 'ged_best_os':
        dummy = 'ged_dummy_os'

    elif conf_type == 'ged_best':
        dummy = 'ged_dummy'

    # sort the df - just in case
    df_sorted = df.sort_values(['pg_id', 'month_id'])

    # groupby gids and get total events
    #df_sb_total_events = df.groupby(['pg_id']).sum()[dummy].reset_index().rename(columns = {dummy:'ged_total_events'})
     #sample_pr_id = df_sb_total_events[df_sb_total_events['ged_total_events'] >= C]['pg_id'].unique()
   
    df_sum = df.groupby(['pg_id', 'year']).sum()[[dummy]].reset_index()
    sample_pr_id = df_sum[df_sum[dummy] >= C]['pg_id'].unique()

    return(sample_pr_id)


def get_spatial_hps(plot = False):

    """Get the one trend spetial prior"""

    η_beta = 5
    ℓ_beta = 1
    ℓ_alpha = 4
    σ_beta = 1

    #grid = np.linspace(0,15,1000)
    #priors = [
    #    ('$\eta$_prior', pm.HalfCauchy.dist(beta=η_beta)),
    #    ('$\ell$_prior', pm.Gamma.dist(alpha=ℓ_alpha , beta=ℓ_beta )),
    #    ('$\sigma$', pm.HalfCauchy.dist(beta=σ_beta))]


    return(η_beta, ℓ_beta, ℓ_alpha, σ_beta)

