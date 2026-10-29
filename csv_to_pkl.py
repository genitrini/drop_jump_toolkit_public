import pandas as pd
import os
from utilities import *
import numpy as np

subj = -1 # start from -1 cause in  the first iteration (all files in root dir) I have no files and no subjects
for root, dirs, files in os.walk("../data_csv"):
    subj += 1
    trial = 0
    h_prev = None
    for f in files:
        if f != 'info.csv':
            height = int(f.split('JUMP', 1)[1][1:3])
            data = pd.read_csv(os.path.join(root, f), skiprows = 3)
            keep = list(set(data.columns) & set(column_rename().keys()))
            data = data[keep]
            data = data.rename(columns = column_rename(), errors="ignore")
            data['time'] = data['time'] - data['time'][0]
            data = data.melt(id_vars="time", var_name="quantity", value_name="value")

            data.insert(2, 'side', np.where(data["quantity"].str.endswith("r"), 'R',  np.where(data["quantity"].str.endswith("l"), 'L', 'X')))
            data.loc[(data["side"] == "R") |
                     (data["side"] == "L"), "quantity"] = data.loc[(data["side"] == "R") |
                                                                   (data["side"] == "L"), "quantity"].str[:-2]
            trial = trial + 1 if height == h_prev else 1
            data.insert(0, 'subject', subj)
            data.insert(1, 'trial', trial)
            data.insert(2, 'height', height)
            data.insert(5, 'muscle', np.where(data["quantity"].isin(['bf', 'st','gm','gl','so','rf','vl','vm','ta',]), 'Y',  'N'))
            data = data[["subject", "trial", "height", "quantity", "muscle", "side", "time", "value"]]
            file_name = 's' + str(subj) + '_' + str(height) + '_' + str(trial) + '.pkl'
            data.to_pickle('../data_pkl_all/' + file_name)
            h_prev = height


        else:
            continue
