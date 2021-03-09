# create test folder to store the data in 
# Use urllib.request to get the files to thr test folder
# get snippets from other files 
# merge ViEWS with prio and pickle merged file in test folder
# Use a temp notebook to test the data
# Cange dirs to fit computerome.

import urllib.request
import pandas as pd
import geopandas as gpd
import pickle

# Getting and loading views data
print('Beginning file download ViEWS...')

url_views = 'https://views.pcr.uu.se/download/datasets/ucdp_views_priogrid_month.csv.zip'
path_views = '/home/polichinel/Documents/Articles/conflict_prediction/data/computerome_test/ucdp_views_priogrid_month.csv.zip'
urllib.request.urlretrieve(url_views, path_views)
df_views = pd.read_csv(path_views)

# Getting and loading prio data
print('Beginning file download PRIO...')

url_prio = 'http://file.prio.no/ReplicationData/PRIO-GRID/priogrid_shapefiles.zip'

path_prio = '/home/polichinel/Documents/Articles/conflict_prediction/data/computerome_test/priogrid_shapefiles.zip'
urllib.request.urlretrieve(url_prio, path_prio)
df_prio = gpd.read_file('zip://' + path_prio)
prio_coord = pd.DataFrame(df_prio[['gid', 'xcoord', 'ycoord']].rename(columns={'gid': 'pg_id'}))

# mergning to get coords
new_df = pd.merge(df_views, prio_coord, how = 'left', on = 'pg_id')

# (re) creating the "best" feature
new_df['ged_best'] = new_df['ged_best_sb'] + new_df['ged_best_ns'] + new_df['ged_best_os']
new_df['ged_dummy'] = ((new_df['ged_dummy_sb'] + new_df['ged_dummy_ns'] + new_df['ged_dummy_os']) > 0) *1

# Save pickle
file_name = "/home/polichinel/Documents/Articles/conflict_prediction/data/computerome_test/ViEWS_coord.pkl"
output = open(file_name, 'wb')
pickle.dump(new_df, output)
output.close()


