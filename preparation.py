import os
import matplotlib
matplotlib.use('TkAgg')
from utilities import *

fs = 2000
###########################################################################################################
# FINDs START END OF JUMP

dir = '../data_pkl_all/'
df= pd.DataFrame(columns = ['subject', 'trial', 'height', 'jump_start', 'jump_end', 'body_mass'])
row_count = 0
for filename in os.listdir(dir):
    row_count +=1
    data = pd.read_pickle(os.path.join(dir, filename))
    fz =data['value'][data.quantity == 'fz'].reset_index(drop = True)
    baseline = fz.iloc[:100].mean() # takes the first 0.1s of force data - unloaded plate

    if abs(baseline) > 10: # if unloaded plate records more/less than 10N, I need to zero it:
        fz = fz - baseline

    jstart_approx = int(np.where(fz>20)[0][0] - 200) # take 0.1s before force crosses a 20N threshold
    jend_approx = jstart_approx + 1000 # assume that contact time is below 0.5s (from visual inspection, I know this value works. The user may change it if needed).
    wstart = len(fz) - 2000 # computes body weight in the interval 1-0.5s before end of file (quiet standing)
    wend = len(fz) - 1000
    body_mass = np.mean(fz[int(wstart) : int(wend)])/ 9.81 # N to kg

    peak_ind = fz[jstart_approx : jend_approx].idxmax() #index of max val of the force in the time window considered
    crossing20 = np.where(np.diff(np.sign(fz-20)) !=0)[0] #index where fz crosses 20N
    pre_land = np.where(crossing20 < peak_ind)[0][-1] # last 20N crossing before peak fz, i.e. when she lands
    land = crossing20[pre_land]
    takeoff = crossing20[pre_land + 1]
    # I take from 0.2 "contact-time times" pre to 0.01s post landing/takeoff . i do it so I can compare between subjects
    jstart, jend = int(crossing20[pre_land] - 0.2*(takeoff - land)), int(crossing20[pre_land + 1] + 0.01*fs) # in samples, not s!
    time = data['time'][data.quantity == 'fz'].reset_index(drop = True)
    fz = data['value'][data.quantity == 'fz'].reset_index(drop = True)
    
    # sanity check with plots
    plt.figure()
    plt.plot(data['time'][data.quantity == 'fz'], data['value'][data.quantity == 'fz'])
    plt.plot(time[[jstart, jend, wstart, wend]], fz[[jstart, jend, wstart, wend]], 'o', color = 'red')
    plt.title("subject: " + str(data.iloc[0, 0]) + ", trial: " + str(data.iloc[0, 1]) + ", height: " + str(data.iloc[0, 2]))
    plt.show()
  
    
    
   

    row = [data.iloc[0, 0], data.iloc[0, 1],data.iloc[0, 2], jstart, jend, body_mass]
    df.loc[len(df)] = row

for subj in df.subject.unique():
    df.loc[df["subject"] == subj, "body_mass"] = df["body_mass"].loc[df["subject"] == subj].mean()


df.to_pickle('time_info_all.pkl')

###################################################################################################
###################################################################################################
###################################################################################################
# This part of the script trims the full files (it can be run independently or separately from the previous one)
# retaining only the data from immediately before to after the contact time

# time_info = df #uncomment if this part is run after the previous without stopping the algorithm and v.v.
time_info = pd.read_pickle('time_info_all.pkl') # comment this line if if this part is run after the previous without stopping the algorithm and v.v.

df = pd.DataFrame()
for file in os.listdir("../data_pkl_all"):
    print(file)
    data = pd.read_pickle("../data_pkl_all/" + file)
    subj = int(file.split("s")[1].split("_")[0])
    h = int(file.split("_")[1].split("_")[0])
    trial = int("_".join(file.split("_")[2:]).split(".")[0])

    start = time_info['jump_start'][
                (time_info.subject == subj) & (time_info.height == h) & (time_info.trial == trial)].item() / fs # samples to s
    end = time_info['jump_end'][
              (time_info.subject == subj) & (time_info.height == h) & (time_info.trial == trial)].item() / fs # samples to s

    data_trial = data[(data.time >= start) & (data.time <= end)]
    df = pd.concat([df, data_trial], axis=0)

df.insert(6, 'body_mass', 0) # add body mass data from data frame previously created
for subj in df.subject.unique():
    df.loc[df['subject'] == subj, 'body_mass'] = time_info['body_mass'][time_info['subject'] == subj].iloc[0].item()

df = df.sort_values(by=["subject", "height", "trial", "muscle", "quantity"], ascending=True)
df = df.reset_index(drop=True)
df.to_pickle('data_clean.pkl')





