
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# get df:
pkl_file = open('/home/simon/Documents/Articles/conflict_prediction/data/computerome/currents/preds_df_tt.pkl', 'rb')
#pkl_file = open('/home/projects/ku_00017/data/generated/currents/rf_selected_features.pkl', 'rb')
df = pickle.load(pkl_file)
pkl_file.close()


cmap = 'turbo' 

for i, j in enumerate(df['X'].unique()):

    plt.figure(figsize = [15,4])
    plt.suptitle(f'Month: {j}', fontsize = 18, y = 1)
    
    plt.subplot(1,3,1)
    plt.title('$cm = log(best_{sb})$')

    plt.scatter(
        df[df['X'] == j]['xcoord'], 
        df[df['X'] == j]['ycoord'], 
        c = df[df['X'] == j]['y'], s=2, marker='s', cmap= cmap, vmin = 0, vmax  = df['y'].max())

    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.title('$\mu_{dce}$')

    plt.scatter(
        df[df['X'] == j]['xcoord'], 
        df[df['X'] == j]['ycoord'], 
        c = df[df['X'] == j]['dce_mu'], s=2, marker='s', cmap= cmap, vmin = 0, vmax  = df['dce_mu'].max() )

    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title('$\mu_{cm}$')

    plt.scatter(
        df[df['X'] == j]['xcoord'], 
        df[df['X'] == j]['ycoord'], 
        c = np.maximum(0,df[df['X'] == j]['cm_mu']), s=2, marker='s', cmap= cmap, vmin = 0, vmax  = df['cm_mu'].max() ) # np.max just for cosmetics - maybe remove
    
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    # save
    no = f'0000{i}'[-3:] # hack
    fig_path = f'/home/simon/Documents/Articles/conflict_prediction/JPR_RNR/the_currents_of_conflict/plots/timelapse_plots/plot_{no}.jpg'
    plt.savefig(fig_path, bbox_inches = "tight")
    
    plt.close(fig=None)
    #plt.cla()
    #plt.clf()

    print(f"{i+1}/{df['X'].unique().shape[0]} done....", end='\r')

print('DONE')
# to generate timelapse run in terminal:
# ffmpeg -framerate 3 -pattern_type glob -i "/home/simon/Documents/Articles/conflict_prediction/JPR_RNR/the_currents_of_conflict/plots/timelapse_plots/*.jpg" -s:v 1440x1080 -c:v libx264 -crf 17 -pix_fmt yuv420p my-timelapse.mp4')


