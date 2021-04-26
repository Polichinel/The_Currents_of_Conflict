import numpy as np
import pandas as pd

import pickle

# why not also get the coords from ViEWS here?

# get dce_pred_df:
#pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/dce_pred_df.pkl', 'rb')
pkl_file = open('/home/projects/ku_00017/data/generated/currents/dce_pred_df_tt.pkl', 'rb')
dce_pred_df = pickle.load(pkl_file)
pkl_file.close()

print('Creating devrivatives')
dce_pred_df.sort_values(['pg_id', 'X'], inplace= True)
dce_pred_df['dce_mu_slope'] = dce_pred_df.groupby('pg_id')['dce_mu'].transform(np.gradient)
dce_pred_df['dce_mu_acc'] = dce_pred_df.groupby('pg_id')['dce_mu_slope'].transform(np.gradient)
dce_pred_df['dce_mu_mass'] = dce_pred_df.groupby('pg_id')['dce_mu'].transform(np.cumsum)

dce_pred_df['dce_mu_s_slope'] = dce_pred_df.groupby('pg_id')['dce_mu_s'].transform(np.gradient)
dce_pred_df['dce_mu_s_acc'] = dce_pred_df.groupby('pg_id')['dce_mu_s_slope'].transform(np.gradient)
dce_pred_df['dce_mu_s_mass'] = dce_pred_df.groupby('pg_id')['dce_mu_s'].transform(np.cumsum)

dce_pred_df['dce_mu_l_slope'] = dce_pred_df.groupby('pg_id')['dce_mu_l'].transform(np.gradient)
dce_pred_df['dce_mu_l_acc'] = dce_pred_df.groupby('pg_id')['dce_mu_l_slope'].transform(np.gradient)
dce_pred_df['dce_mu_l_mass'] = dce_pred_df.groupby('pg_id')['dce_mu_l'].transform(np.cumsum)

# get cm_pred_df:
#pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/cm_pred_df.pkl', 'rb')
pkl_file = open('/home/projects/ku_00017/data/generated/currents/cm_pred_df_tt.pkl', 'rb')
cm_pred_df = pickle.load(pkl_file)
pkl_file.close()

cm_pred_df.rename(columns =  {'mu' : 'cm_mu', 'var': 'cm_var', 
                              'mu_s' : 'cm_mu_s', 'var_s': 'cm_var_s', 
                              'mu_l' : 'cm_mu_l', 'var_l': 'cm_var_l'}, inplace=True)


cm_pred_df.sort_values(['pg_id', 'X'], inplace= True)
cm_pred_df['cm_mu_slope'] = cm_pred_df.groupby('pg_id')['cm_mu'].transform(np.gradient)
cm_pred_df['cm_mu_acc'] = cm_pred_df.groupby('pg_id')['cm_mu_slope'].transform(np.gradient)
cm_pred_df['cm_mu_mass'] = cm_pred_df.groupby('pg_id')['cm_mu'].transform(np.cumsum)

cm_pred_df['cm_mu_s_slope'] = cm_pred_df.groupby('pg_id')['cm_mu_s'].transform(np.gradient)
cm_pred_df['cm_mu_s_acc'] = cm_pred_df.groupby('pg_id')['cm_mu_s_slope'].transform(np.gradient)
cm_pred_df['cm_mu_s_mass'] = cm_pred_df.groupby('pg_id')['cm_mu_s'].transform(np.cumsum)

cm_pred_df['cm_mu_l_slope'] = cm_pred_df.groupby('pg_id')['cm_mu_l'].transform(np.gradient)
cm_pred_df['cm_mu_l_acc'] = cm_pred_df.groupby('pg_id')['cm_mu_l_slope'].transform(np.gradient)
cm_pred_df['cm_mu_l_mass'] = cm_pred_df.groupby('pg_id')['cm_mu_l'].transform(np.cumsum)

df_merged = pd.merge(dce_pred_df, cm_pred_df, how = 'left', on = ['id', 'pg_id', 'train', 'X'])


print('Pickling..')
#new_file_name = '/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/preds_df.pkl'
new_file_name = '/home/projects/ku_00017/data/generated/currents/preds_df_tt.pkl'
output = open(new_file_name, 'wb')
pickle.dump(df_merged, output)
output.close()