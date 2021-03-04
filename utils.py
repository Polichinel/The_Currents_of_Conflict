import os
import numpy as np
import pandas as pd

def get_views_coord():

    """Get views data with coords and return it as a pandas dataframe"""

    path = '/home/polichinel/Documents/Articles/conflict_prediction/data/ViEWS/'
    file_name = 'ViEWS_coord.pkl'

    file_path = os.path.join(path, file_name)
    views_coord = pd.read_pickle(file_path)

    return(views_coord)


def test_val_train(df, info = True):

    """For train, validation and test. In accordance with Hegre et al. 2019 p. 163.
    This function returns two lists, each with tre enties. 
    The 'start' list holds the start idx for train[0], val[1] and test[2].
    The 'end' lisr holds the end ind for train[0], val[1] and test[2]."""

    #Train: jan 1990 = month_id 121 (12) - dec 2011 = month_id 384 (275)
    #Val: jan 2012 = month_id 385 (276)- dec 2014 = month_id 420 (311) # !!! er det her ikke kun 35?
    #Test: jan 2015 = month_id 421 (312) - dec 2017 = month_id 456 (347) # tjek at det her er 36...

    set_ = ['Train', 'Validation', 'Test (not used here)']
    start = [12, 276, 312]
    end = [275, 311, 347]

    for i in range(3):

        jan = df['month_id'].unique()[start[i]]
        dec = df['month_id'].unique()[end[i]]

        month1 = df[df['month_id'] == jan][['year', 'month']]['month'].unique()
        year1 = df[df['month_id'] == jan][['year', 'month']]['year'].unique()

        month2 = df[df['month_id'] == dec][['year', 'month']]['month'].unique()
        year2 = df[df['month_id'] == dec][['year', 'month']]['year'].unique()

        if info == True:
            string = f'{set_[i]} set from: {month1[0]}/{year1[0]} to {month2[0]}/{year2[0]}\n'
            print(string)

    return(start, end)

start, end = test_val_train(df)