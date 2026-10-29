import pandas as pd
import numpy as np
from utilities import *
import matplotlib
import seaborn as sns

from matplotlib import pyplot as plt
import os
os.chdir("your/project/directory")
fs = 2000

################################################################ DROP JUMP METRICS #########################################################################


data = pd.read_pickle("data_clean.pkl")
data_fz = data[data['quantity'] == 'fz']
df = pd.DataFrame(columns = ['subject', 'height', 'trial', 'rsi', 'jump_height',
                             'ttto', 'com_disp_min', 'pass_ecc_dur', 'act_ecc_dur', 'conc_dur',
                             'k_avg', 'avg_act_ecc_force', 'avg_con_force', 'com_vel_pass', 'com_vel_att',
                             'avg_pos_power', 'peak_power', 'peak_power_instant'])



# iterating over subjects, box heights and trials
for subj in data_fz.subject.unique():
    for height in data_fz[data_fz.subject == subj].height.unique():
        for trial in data_fz[(data_fz.subject == subj) & (data_fz.height == height)].trial.unique():
            fz = data_fz['value'][(data_fz.subject == subj) & (data_fz.height == height) & (data_fz.trial == trial)].to_numpy() # take force data for this jump from data frame
            body_mass = data_fz['body_mass'][(data_fz.subject == subj) & (data_fz.height == height)].mean() # take body mass
            datatrial = data[(data.subject == subj) & (data.height == height) & (data.trial == trial)] # take data for that given jump
            quantities_list = dj(datatrial, fz, fs, body_mass)
            id = list([int(subj), int(height), int(trial)])
            row = id.extend(quantities_list)
            df.loc[len(df)] = row # adds iteratively a row with the data of a given jump instance to the previously created data frame


df.to_pickle('jump_data_computed.pkl') # saves data





