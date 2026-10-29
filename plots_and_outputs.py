import pandas as pd
import numpy as np
np.set_printoptions(legacy = '1.25')
from utilities import *
from matplotlib.colors import ListedColormap
import matplotlib
from matplotlib import pyplot as plt
#from sklearn.linear_model import LinearRegression
import matplotlib.cm as cmap

df = pd.read_pickle("jump_data.pkl") # upload the data frame produced by the "variables_computation" script
df["subject"] = df["subject"].astype("category")
df["height"] = df["height"].astype("category")
df['com_disp_min'] *= -1 # setting downwards displecement and velocities to positive (i.e. I take their absolute value)
df['com_vel_att'] *= -1
df['com_vel_pass'] *= -1

##########################################################################################################################
##########################################################################################################################

# JUMP COMPARISON AND DESCRIPTION
# this functions corresponds to the first objective of the article:
# providing practiotioners with a clear tool to compare jump instances - both within and between subjects

j1 = [9, 20, 1] # first number = participant; second number = box height; third number = trial
j2 = [8, 40, 2]
data = jump_comparison(df, j1, j2)

##########################################################################################################################
##########################################################################################################################
# Creating, for each athlete his/her personal table

cols_idx = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14] # selecting the quantities (data frame cols) to be included in the output tables
matrices = []

for subj in df['subject'].unique():

    dfsubj = df[df['subject'] == subj].reset_index(drop=True) # selecting data relative to a given subject only
    arr = dfsubj.to_numpy()
    rsi_col = dfsubj.columns.get_loc('rsi')
    n = len(arr)
    n_pairs = n * (n - 1) // 2 # number of possible pairwise comparisons
    n_feat = len(cols_idx) # number of features
    matrix = np.empty((n_pairs, n_feat), dtype=np.uint8)
    idx = 0

    for i in range(n - 1):
        rsi_i = arr[i, rsi_col]
        for j in range(i + 1, n): # based on RSI, it creates a matrix of 0 and 1

            if rsi_i > arr[j, rsi_col]:
                matrix[idx] = arr[i, cols_idx] > arr[j, cols_idx]
            else:
                matrix[idx] = arr[j, cols_idx] > arr[i, cols_idx]

            idx += 1

    matrices.append(matrix)

##########################################################################################################################
# creates matrix with white and gray cells

cols = ['JH', r"$\mathbf{CT}_{\mathbf{BW}}$", 'COM disp', 'I-ecc dur', 'B-ecc dur', 'Conc dur', 'Stiffness', 'Ecc force', 'Conc force', 'COM V init']#, 'Pos power']

matplotlib.use('TkAgg')
cmap = ListedColormap(["white", "grey"])
for mat in matrices:
    col_pct = np.round(mat.sum(axis=0) / np.size(mat, 0) * 100, 0)

    col_pct = [x if x > 50 else 100 - x for x in col_pct]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)

    ax.grid(which="minor", color="black", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    for j, col in enumerate(cols):  # column labels
        ax.text(j, -1.3, col, ha='center', va='center', fontsize=15, fontweight='bold')

    for j in range(np.size(mat, 1)):
        ax.text(j, np.size(mat, 0) + 0.3, f"{col_pct[j]:.0f}%",
                ha='center', va='center', fontsize=15, fontweight='bold')

        ax.set_ylim(np.size(mat, 0)+ 1, -0.5)


plt.show()


print('done')
######################################################################################################################################
######################################################################################################################################
# Running statistics on individual tables. The result will be a table with each subject representing a single row
# so to compare different individuals

nperm = 1000 # number of permutations
njumps = 9 # number of jumps per person

######################################################################################################################################
# Binomial Wilson test
p = 0.5
combs = sum(np.arange(njumps)) # possible pairwise combinations
mu = p * combs
variance = combs * p * (1-p)
sigma = np.sqrt(variance)
z_sig = 2.33 # 2.33 for 1% ; 1.96 for 5% significance threshold

df_ci = pd.DataFrame(columns = ['subject', 'quantity', 'frequency',  'ci_low', 'ci_high', 'significant'])
quantities = ['JH', 'Contact time', 'COM disp', 'I-ecc dur', 'B-ecc dur', 'Conc dur', 'stiffness', 'Ecc force', 'Conc force', 'COM v ecc']


for subj in df.subject.unique(): # computing stats
    matsubj = matrices[int(subj)-1]
    for i in np.arange(matsubj.shape[1]):
        pi = np.sum(matsubj[:, i]) / matsubj.shape[0]
        x = np.sum(matsubj[:, i])
        p_hat = (pi + (z_sig**2 / (2*combs))) / (1 + (z_sig**2 / combs))
        semi_range = (z_sig * np.sqrt((pi * (1 - pi) / combs) + (z_sig**2 / (4 * combs **2)))) / (1 + z_sig**2 / combs)
        ci_low = p_hat - semi_range
        ci_high = p_hat + semi_range
        significant = 1 if 0.5 < ci_low or 0.5 > ci_high else 0
        frequency = pi * 100

        row = [subj, quantities[i], frequency,  ci_low, ci_high, significant]
        df_ci.loc[len(df_ci)] = row

mat = np.empty([10, 10])
for i in np.arange(mat.shape[0]):
    mat[i, : ] = df_ci.significant[df_ci.subject == i+1].values

######################################################################################################################################
# plotting after Wilson binomial test

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(14, 6))

# draw patterned matrix manually
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):

        if mat[i, j] == 1:
            # white + grey pattern
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor="white", edgecolor="black", hatch="xx", linewidth=2.5)
        else:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor="white", edgecolor="white", linewidth=0)
        ax.add_patch(rect)

ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
ax.grid(which="minor", color="black", linewidth=1)
ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

# top header boxes
for j in range(mat.shape[1]):
    ax.add_patch(plt.Rectangle((j - 0.5, -1.5), 1, 1, facecolor="lightgrey", edgecolor="none", clip_on=False))

# left header boxes
for i in range(mat.shape[0]):
    ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 1, 1, facecolor="lightgrey", edgecolor="none", clip_on=False))

# column labels
for j, col in enumerate(cols):
    ax.text(j, -1, col, ha='center', va='center', fontsize=16, fontweight='bold')

# row labels
for i in range(mat.shape[0]):
    ax.text(-1, i, f" S {i+1}",  ha='right', va='center', fontsize=22, fontweight='bold')

ax.set_xlim(-1.5, mat.shape[1] - 0.5)
ax.set_ylim(mat.shape[0] - 0.5, -1.5)
ax.set_title('After Wilson binomial confidence interval', fontsize=40, pad=30)
fig.subplots_adjust(left=0.055)
plt.show()

######################################################################################################################################
######## Leave 2 out stats : I repeat as above leaving itarartively 2/9 jumps out of the computation
# and re-calculating significance

from itertools import combinations
import numpy as np
import pandas as pd

p = 0.5
z_sig = 2.33
cols_idx = [4,5,6,7,8,9,10,11,12,14]
quantities = df.columns[cols_idx]
rsi_col = df.columns.get_loc('rsi')

subjects_data = {s: df[df.subject == s].reset_index(drop=True).to_numpy() for s in df.subject.unique()}
subj_list = []
quant_list = []
outcome_list = []

for subject, arr_full in subjects_data.items():
    n_jumps = arr_full.shape[0]

    # all leave-2-out combinations
    leave2out = list(combinations(range(n_jumps), 2))
    for excluded in leave2out:

        mask = np.ones(n_jumps, dtype=bool)
        mask[list(excluded)] = False
        arr = arr_full[mask]
        njumpsperm = arr.shape[0]
        n_pairs = njumpsperm * (njumpsperm - 1) // 2
        combs = n_pairs
        matrix = np.empty((n_pairs, len(cols_idx)), dtype=np.uint8)
        idx = 0

        for j in range(njumpsperm - 1):
            rsi_j = arr[j, rsi_col]

            for k in range(j + 1, njumpsperm):
                if rsi_j > arr[k, rsi_col]:
                    matrix[idx] = ( arr[j, cols_idx] > arr[k, cols_idx])

                else:
                    matrix[idx] = (arr[k, cols_idx] > arr[j, cols_idx])

                idx += 1

        col_sums = matrix.sum(axis=0)
        pi = col_sums / n_pairs
        p_hat = (pi + (z_sig**2 / (2 * combs))) / (1 + (z_sig**2 / combs))
        semi_range = (z_sig * np.sqrt((pi * (1 - pi) / combs) + (z_sig**2 / (4 * combs**2)))) / (1 + z_sig**2 / combs)
        ci_low = p_hat - semi_range
        ci_high = p_hat + semi_range

        outcomes = ((0.5 < ci_low) | (0.5 > ci_high)).astype(np.uint8)
        subj_list.extend([int(subject)] * len(cols_idx))
        quant_list.extend(quantities)
        outcome_list.extend(outcomes)

df_perm = pd.DataFrame({'subject': subj_list, 'quantity': quant_list, 'outcome': outcome_list })
dffinal = (df_perm.groupby(['subject', 'quantity'], as_index=False)['outcome'].sum())

dffinal['significant'] = 0
for subj in df.subject.unique():
    dfsubj = df[df.subject == subj]
    njumps = dfsubj.shape[0]
    combs = sum(np.arange(njumps - 2))
    mu = p * combs
    sigma = np.sqrt(combs * p * (1 - p))
    x = z_sig * sigma + mu + 0.5
    dffinal.loc[(dffinal["subject"] == subj) & (dffinal["outcome"] > x), "significant"] = 1

df_ci['quantity'] = df_ci['quantity'].replace({'JH' : 'jump_height'})


#########################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# pivot to matrix
quantities = df.columns[[4, 5, 6, 7, 8, 9, 10, 11, 12, 14]]

mat = dffinal.assign(val=(dffinal['significant'] == 1).astype(int)) .pivot(index='subject',
                                                                           columns='quantity', values='val') .fillna(0)

mat = mat.reindex(columns=quantities)
fig, ax = plt.subplots(figsize=(14, 6))


for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if mat.iloc[i, j] == 1:
            rect = plt.Rectangle ((j - 0.5, i - 0.5), 1, 1, facecolor='white',
                                  edgecolor='black', hatch='xx',  linewidth=2.5)
        else:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor='white', edgecolor='white', linewidth=0)

        ax.add_patch(rect)


ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
ax.grid(which="minor", color="black", linewidth=1)

ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

for j in range(mat.shape[1]):
    ax.add_patch(plt.Rectangle( (j - 0.5, -1.5), 1, 1, facecolor='lightgrey',
                                alpha=0.5, edgecolor='none', clip_on=False))

for i in range(mat.shape[0]):
    ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 1, 1,
                               facecolor='lightgrey', alpha=0.5, edgecolor='none', clip_on=False))

for j, col in enumerate(cols):
    ax.text( j, -1,  col, ha='center', va='center', fontsize=16,  fontweight='bold' )

for i in range(mat.shape[0]):
    ax.text(-1, i, f"S {i+1}", ha='right', va='center', fontsize=22, fontweight='bold' )


ax.set_xlim(-1.5, mat.shape[1] - 0.5)
ax.set_ylim(mat.shape[0] - 0.5, -1.5)

ax.set_title( 'After leave-two-out', fontsize=40, pad=30)
fig.subplots_adjust(left=0.055)
plt.show()

##################################################################################################
######## step 3 permutation with respect to RSI
feature_cols = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14]

pvalues_table = np.empty((len(df.subject.unique()), len(feature_cols)))
for subj in df.subject.unique():
    dfsubj = df[df.subject == subj].reset_index(drop = True)
    pvalues_table[int(subj - 1), :] = permutation_test_fast(dfsubj, rsi_col = 3,
                                                            feature_cols = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14], n_perm=1000, seed=42)

##################################################################################################
# plot
quantities = df.columns[[4, 5, 6, 7, 8, 9, 10, 11, 12, 14]]#, 15]]
mat = pvalues_table
matplotlib.use('TkAgg')
fig, ax = plt.subplots(figsize=(14, 6))

# draw matrix cells (white + grey pattern)
M = np.asarray(mat)

for i in range(M.shape[0]):
    for j in range(M.shape[1]):

        if M[i, j] == 1:
            ax.add_patch( plt.Rectangle( (j - 0.5, i - 0.5), 1, 1,
                                         facecolor='white', edgecolor='black',  hatch='xx', linewidth=2.5))

        else:
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor='white', edgecolor='white', linewidth=0))


# grid
ax.set_xticks(np.arange(-0.5, M.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, M.shape[0], 1), minor=True)
ax.grid(which="minor", color="black", linewidth=1)

ax.tick_params( which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
# column header backgrounds
for j in range(M.shape[1]):
    ax.add_patch( plt.Rectangle( (j - 0.5, -1.5), 1, 1,
                                 facecolor='lightgrey', alpha=0.5, edgecolor='none', clip_on=False ))

# row header backgrounds
for i in range(M.shape[0]):
    ax.add_patch(plt.Rectangle((-1.5, i - 0.5), 1, 1,
                               facecolor='lightgrey', alpha=0.5, edgecolor='none', clip_on=False))


# labels
for j, col in enumerate(cols):
    ax.text( j, -1, col, ha='center', va='center', fontsize=16, fontweight='bold')

for i in range(M.shape[0]):
    ax.text(-1, i, f"S {i+1}", ha='right', va='center', fontsize=22, fontweight='bold' )

ax.set_xlim(-1.5, M.shape[1] - 0.5)
ax.set_ylim(M.shape[0] - 0.5, -1.5)

ax.set_title( 'After RSI-shuffling permutations', fontsize=40, pad=30 )
fig.subplots_adjust(left=0.055)
plt.show()

