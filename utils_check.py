import os
import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import theano
from IPython.display import clear_output

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


def sample_conflict_timeline_old(conf_type, df, train_id, test_id, C=5):

    """ This function samples N time-lines contining c>=C conflicts. 
    As default it will try to get the val_id. Error will come if it does not exits"""

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
    df_sb_total_events = df.groupby(['pg_id']).sum()[dummy].reset_index().rename(columns = {dummy:'ged_total_events'})
    
    sample_pr_id = df_sb_total_events[df_sb_total_events['ged_total_events'] >= C]['pg_id'].unique()

    return(sample_pr_id)


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


def get_spatial_hps(plot = False):

    """Get the one trend spetial prior"""

    η_beta = 5
    ℓ_beta = 1
    ℓ_alpha = 4
    σ_beta = 1

    grid = np.linspace(0,15,1000)
    priors = [
        ('$\eta$_prior', pm.HalfCauchy.dist(beta=η_beta)),
        ('$\ell$_prior', pm.Gamma.dist(alpha=ℓ_alpha , beta=ℓ_beta )),
        ('$\sigma$', pm.HalfCauchy.dist(beta=σ_beta))]

    if plot == True:

        plt.figure(figsize= [15,5])
        plt.title('Hyper-priors')

        for i, prior in enumerate(priors):
            plt.plot(grid, np.exp(prior[1].logp(grid).eval()), label = prior[0])

        plt.xticks(np.arange(0,16,1))
        plt.xlim(0,15)
        plt.legend()
        plt.show()

    return(η_beta, ℓ_beta, ℓ_alpha, σ_beta)



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
    y_new_list = []
    idx_list = []
    pg_idx_list = []
    train_list = []

    # Loop gp predict over time lines
    for i, j in enumerate(sample_pg_id):

        print(f'Time-line {i+1}/{sample_pg_id.shape[0]} in the works (prediction)...', end = '\r')

        idx = df_sorted[(df_sorted['id'].isin(new_id)) & (df_sorted['pg_id'] == j)]['id'].values
        y_new = np.log(df_sorted[(df_sorted['id'].isin(new_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values

        #X = df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None]
        #y = np.log(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values
        X.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None])
        y.set_value(np.log(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values)

        #if indv_mean == True:
            #gp.mean_func = pm.gp.mean.Constant(y.mean()) # individual mean_func

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
        y_new_list.append(y_new)
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
    y_new_col = np.array(y_new_list).reshape(-1,)     
    idx_col = np.array(idx_list).reshape(-1,)    
    pg_idx_col = np.array(pg_idx_list).reshape(-1,)
    train_col =  np.array(train_list).reshape(-1,)

    df_new = pd.DataFrame({
                            'mu': mu_col, 'mu_s' : mu_s_col, 'mu_l' : mu_l_col,
                            'var' : var_col, 'var_s' : var_s_col, 'var_l' : var_l_col, 
                            'X' : X_new_col, 'y' : y_new_col , 
                            'id' : idx_col, 'pg_id' : pg_idx_col, 'train' : train_col
                            })

    return(df_new)

def predict_ot(conf_type, df, train_id, test_id, mp, gp, σ, C):

    """same as normal, just lazy implimentation if we only have one trend"""

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
    var_list = []
    X_new_list = []
    y_new_list = []
    idx_list = []
    pg_idx_list = []
    train_list = []

    # Loop gp predict over time lines
    for i, j in enumerate(sample_pg_id):

        print(f'Time-line {i+1}/{sample_pg_id.shape[0]} in the works (prediction)...')
        clear_output(wait=True)        

        idx = df_sorted[(df_sorted['id'].isin(new_id)) & (df_sorted['pg_id'] == j)]['id'].values
        y_new = np.log(df_sorted[(df_sorted['id'].isin(new_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values

        #X = df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None]
        #y = np.log(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values
        X.set_value(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)]['month_id'].values[:,None])
        y.set_value(np.log(df_sorted[(df_sorted['id'].isin(train_id)) & (df_sorted['pg_id'] == j)][conf_type] + 1).values)

        #gp.mean_func = pm.gp.mean.Constant(y.mean()) # individual mean_func

        mu, var = gp.predict(X_new, point=mp, given = {'gp' : gp, 'X' : X, 'y' : y, 'noise' : σ}, diag=True)

        mu_list.append(mu)
        var_list.append(var)
        X_new_list.append(X_new)
        y_new_list.append(y_new)
        idx_list.append(idx)
        pg_idx_list.append([j] * mu.shape[0])
        train_list.append(np.array([1] * train_len + [0] * test_len)) # dummy for training...
        #train_list.append(np.array([1] * y.shape[0] + [0] * (mu.shape[0] - y.shape[0]))) # dummy for training...

    mu_col = np.array(mu_list).reshape(-1,) 
    var_col = np.array(var_list).reshape(-1,) 
    X_new_col = np.array(X_new_list).reshape(-1,) 
    y_new_col = np.array(y_new_list).reshape(-1,)     
    idx_col = np.array(idx_list).reshape(-1,)    
    pg_idx_col = np.array(pg_idx_list).reshape(-1,)
    train_col =  np.array(train_list).reshape(-1,)

    df_new = pd.DataFrame({
                            'mu': mu_col, 'var' : var_col, 'X' : X_new_col, 'y' : y_new_col , 
                            'id' : idx_col, 'pg_id' : pg_idx_col, 'train' : train_col
                            })

    return(df_new)

def plot_predictions(df_merged):

    """This funcitons takes df containint the original data and the predictied data.
    Specfically, the df should be a merger between the original df and the new_df outputted by 'predict' """

    time_lines = df_merged['pg_id'].unique()
    
    fig = plt.figure(figsize=(20, 8))
    colors = sns.color_palette("hls", len(time_lines))

    print(f'Number of time lines plotted: {len(time_lines)}')

    for i,j in enumerate(time_lines):

        X = df_merged[df_merged['pg_id'] == j]['X']
        y = df_merged[df_merged['pg_id'] == j]['y']

        X_train = df_merged[(df_merged['pg_id'] == j) & (df_merged['train'] == 1)]['X']
        X_test = df_merged[(df_merged['pg_id'] == j) & (df_merged['train'] == 0)]['X']
        y_train = df_merged[(df_merged['pg_id'] == j) & (df_merged['train'] == 1)]['y']
        y_test = df_merged[(df_merged['pg_id'] == j) & (df_merged['train'] == 0)]['y']

        mu = df_merged[df_merged['pg_id'] == j]['mu']
        mu_s = df_merged[df_merged['pg_id'] == j]['mu_s']
        mu_l = df_merged[df_merged['pg_id'] == j]['mu_l']
        sd = np.sqrt(df_merged[df_merged['pg_id'] == j]['var'])
        sd_s = np.sqrt(df_merged[df_merged['pg_id'] == j]['var_s'])
        sd_l = np.sqrt(df_merged[df_merged['pg_id'] == j]['var_l'])

        plt.plot(X, mu,'-', color = colors[i])
        plt.plot(X, mu_s,'-', color = colors[i])
        plt.plot(X, mu_l,'-', color = colors[i])

        plt.plot(X_train, y_train,'o', color = colors[i])
        plt.plot(X_test, y_test,'x', color = colors[i])

        plt.plot(X, mu + 2 * sd, "-", lw=1, color=colors[i], alpha=0.5)
        plt.plot(X, mu - 2 * sd, "-", lw=1, color=colors[i], alpha=0.5)
        plt.fill_between(X, mu - 2 * sd, mu + 2 * sd, color=colors[i], alpha=0.2)

        plt.plot(X, mu_s + 2 * sd_s, "-", lw=1, color=colors[i], alpha=0.5)
        plt.plot(X, mu_s - 2 * sd_s, "-", lw=1, color=colors[i], alpha=0.5)
        plt.fill_between(X, mu_s - 2 * sd_s, mu_s + 2 * sd_s, color=colors[i], alpha=0.2)

        plt.plot(X, mu_l + 2 * sd_l, "-", lw=1, color=colors[i], alpha=0.5)
        plt.plot(X, mu_l - 2 * sd_l, "-", lw=1, color=colors[i], alpha=0.5)
        plt.fill_between(X, mu_l - 2 * sd_l, mu_l + 2 * sd_l, color=colors[i], alpha=0.2)


    plt.vlines(df_merged['X'].max()-36, -1, 8, linestyles='dashed', color = 'red', alpha = 0.5)

    plt.show()


# This here just vectorize it!
def get_mse(df_merged, train_id, test_id):

    """This funciton takes a merged df, the train ids and val/test ids. 
    The df should be a merger between the original df and the new-df from 'predict'.
    The funciton outputs a df containing the in/out mse for the gp, gp_s and gp_l."""

    # iterate over time lines - we would like a distribution of mse

    y_true_train = df_merged[df_merged['id'].isin(train_id)]['y']
    pred_train = df_merged[df_merged['id'].isin(train_id)]['mu']
    pred_s_train = df_merged[df_merged['id'].isin(train_id)]['mu_s']
    pred_l_train = df_merged[df_merged['id'].isin(train_id)]['mu_l']

    mse_train = mean_squared_error(y_true_train, pred_train)
    mse_s_train = mean_squared_error(y_true_train, pred_s_train)
    mse_l_train = mean_squared_error(y_true_train, pred_l_train)

    y_true_test = df_merged[df_merged['id'].isin(test_id)]['y']
    pred_test = df_merged[df_merged['id'].isin(test_id)]['mu']
    pred_s_test = df_merged[df_merged['id'].isin(test_id)]['mu_s']
    pred_l_test = df_merged[df_merged['id'].isin(test_id)]['mu_l']

    mse_test = mean_squared_error(y_true_test, pred_test)
    mse_s_test = mean_squared_error(y_true_test, pred_s_test)
    mse_l_test = mean_squared_error(y_true_test, pred_l_test)

    mse_resutls_df = pd.DataFrame({
            "Gps": ["Full", "Short", "long"],
            "MSE insample (mean)": [mse_train, mse_s_train, mse_l_train],
            "MSE outsample (mean)": [mse_test, mse_s_test, mse_l_test],
            })

    return(mse_resutls_df)


# This here just vectorize it!
def get_mse_ot(df_merged, train_id, test_id):

    """Same as normal just for one-trend"""

    # iterate over time lines - we would like a distribution of mse

    y_true_train = df_merged[df_merged['id'].isin(train_id)]['y']
    pred_train = df_merged[df_merged['id'].isin(train_id)]['mu']

    mse_train = mean_squared_error(y_true_train, pred_train)

    y_true_test = df_merged[df_merged['id'].isin(test_id)]['y']
    pred_test = df_merged[df_merged['id'].isin(test_id)]['mu']

    mse_test = mean_squared_error(y_true_test, pred_test)

    mse_resutls_df = pd.DataFrame({
            "Gps": ["Full"],
            "MSE insample (mean)": [mse_train],
            "MSE outsample (mean)": [mse_test],
            })

    return(mse_resutls_df)


# make sure you can use -1 cores..
def get_metrics(df_merged, train_id, test_id):

    """A function that takes the merged df.
    The df must now include both data from 'predict',
    and the devrived slope, acc ans mass.
    The function uses a simple rf classifier to test the temporal features.
    Very simple classifier so results are only indicative"""

#    X_train = df_merged[df_merged['id'].isin(train_id)][['mu_l', 'mu_l_slope', 'mu_l_acc', 'mu_l_mass']] 
#    y_train = (df_merged[df_merged['id'].isin(train_id)]['y'] > 0) * 1

#    X_test = df_merged[df_merged['id'].isin(test_id)][['mu_l', 'mu_l_slope', 'mu_l_acc', 'mu_l_mass']] 
#    y_test = (df_merged[df_merged['id'].isin(test_id)]['y'] > 0) * 1

    X_train = df_merged[df_merged['id'].isin(train_id)][['mu', 'mu_slope', 'mu_acc', 
                                                         'mu_mass','mu_s', 'mu_s_slope', 
                                                         'mu_s_acc', 'mu_s_mass','mu_l', 
                                                         'mu_l_slope', 'mu_l_acc', 'mu_l_mass',
                                                         'var', 'var_s', 'var_l']] 
    
    y_train = (df_merged[df_merged['id'].isin(train_id)]['y'] > 0) * 1

    X_test = df_merged[df_merged['id'].isin(test_id)][['mu', 'mu_slope', 'mu_acc', 
                                                         'mu_mass','mu_s', 'mu_s_slope', 
                                                         'mu_s_acc', 'mu_s_mass','mu_l', 
                                                         'mu_l_slope', 'mu_l_acc', 'mu_l_mass',
                                                         'var', 'var_s', 'var_l']]

    y_test = (df_merged[df_merged['id'].isin(test_id)]['y'] > 0) * 1



    # totally vanilla - just indicative
    model = RandomForestClassifier(n_estimators=64, max_depth=6, min_samples_split=8, random_state=42, n_jobs= -1)
    #model = AdaBoostClassifier(n_estimators=100, random_state=42)
    #model = LogisticRegression()

    model.fit(X_train, y_train)

    y_train_pred = model.predict_proba(X_train)[:,1]
    y_test_pred = model.predict_proba(X_test)[:,1]

    AUC_train = metrics.roc_auc_score(y_train, y_train_pred)
    AP_train = metrics.average_precision_score(y_train, y_train_pred)
    BS_train = metrics.brier_score_loss(y_train, y_train_pred)

    AUC_test = metrics.roc_auc_score(y_test, y_test_pred)
    AP_test = metrics.average_precision_score(y_test, y_test_pred)
    BS_test = metrics.brier_score_loss(y_test, y_test_pred)

    df_results =  pd.DataFrame({
            "Metrics": ["AUC", "AP", "BS"],
            "Train": [AUC_train, AP_train, BS_train],
            "Test": [AUC_test, AP_test, BS_test]
        })

    return(df_results)



# make sure you can use -1 cores..
def get_metrics_ot(df_merged, train_id, test_id):

    """Same as normal, just for only one-trend"""

    X_train = df_merged[df_merged['id'].isin(train_id)][['mu', 'mu_slope', 'mu_acc', 'mu_mass', 'var']] 
    
    y_train = (df_merged[df_merged['id'].isin(train_id)]['y'] > 0) * 1

    X_test = df_merged[df_merged['id'].isin(test_id)][['mu', 'mu_slope', 'mu_acc','mu_mass','var']]

    y_test = (df_merged[df_merged['id'].isin(test_id)]['y'] > 0) * 1

    # totally vanilla - just indicative
    model = RandomForestClassifier(n_estimators=64, max_depth=6, min_samples_split=8, random_state=42, n_jobs= -1)

    model.fit(X_train, y_train)

    y_train_pred = model.predict_proba(X_train)[:,1]
    y_test_pred = model.predict_proba(X_test)[:,1]

    AUC_train = metrics.roc_auc_score(y_train, y_train_pred)
    AP_train = metrics.average_precision_score(y_train, y_train_pred)
    BS_train = metrics.brier_score_loss(y_train, y_train_pred)

    AUC_test = metrics.roc_auc_score(y_test, y_test_pred)
    AP_test = metrics.average_precision_score(y_test, y_test_pred)
    BS_test = metrics.brier_score_loss(y_test, y_test_pred)

    df_results =  pd.DataFrame({
            "Metrics": ["AUC", "AP", "BS"],
            "Train": [AUC_train, AP_train, BS_train],
            "Test": [AUC_test, AP_test, BS_test]
        })

    return(df_results)