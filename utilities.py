import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(legacy = '1.25')
from scipy import signal

# this function renames columns from software (Noraxon Myo Research to
# custom names
def column_rename():
    return {
        'time' : 'time',
        "BICEPS FEM. RT (uV)": "bf_r",
        "SEMITEND. RT (uV)": "st_r",
        "LAT. GASTRO RT (uV)": "gl_r",
        "MED. GASTRO RT (uV)": "gm_r",
        "SOLEUS RT (uV)": "so_r",
        "RECTUS FEM. RT (uV)": "rf_r",
        "VLO RT (uV)": "vl_r",
        "VMO RT (uV)": "vm_r",
        "TIB.ANT. RT (uV)": "ta_r",
        "Hip Flexion RT (deg)": "hip_fe_r",
        "Hip Abduction RT (deg)": "hip_aa_r",
        "Hip Rotation Ext RT (deg)": "hip_ie_r",
        "Knee Flexion RT (deg)": "knee_fe_r",
        "Ankle Dorsiflexion RT (deg)": "ankle_fe_r",
        "Ankle Abduction RT (deg)": "ankle_aa_r",
        "Ankle Inversion RT (deg)": "ankle_ie_r",
        "BICEPS FEM. LT (uV)": "bf_l",
        "SEMITEND. LT (uV)": "st_l",
        "LAT. GASTRO LT (uV)": "gl_l",
        "MED. GASTRO LT (uV)": "gm_l",
        "SOLEUS LT (uV)": "so_l",
        "RECTUS FEM. LT (uV)": "rf_l",
        "VLO LT (uV)": "vl_l",
        "VMO LT (uV)": "vm_l",
        "TIB.ANT. LT (uV)": "ta_l",
        "Hip Flexion LT (deg)": "hip_fe_l",
        "Hip Abduction LT (deg)": "hip_aa_l",
        "Hip Rotation Ext LT (deg)": "hip_ie_l",
        "Knee Flexion LT (deg)": "knee_fe_l",
        "Ankle Dorsiflexion LT (deg)": "ankle_fe_l",
        "Ankle Abduction LT (deg)": "ankle_aa_l",
        "Ankle Inversion LT (deg)": "ankle_ie_l",
        "Force plate Fx (N)": "fx",
        "Force plate Fy (N)": "fy",
        "Force plate Fz (N)": "fz",
        'Force plate Mx (N*m)' : "mx",
        'Force plate My (N*m)' : "my",
        'Force plate Mz (N*m)' : "mz",
        'Force plate group-Force plates-Force plate-Center of pressure-x (mm)' : "px",
        'Force plate group-Force plates-Force plate-Center of pressure-y (mm)': "py"
    }

def zeroing(fz):
    # This function zero-es the force plates if the experimenter did not do it prior to a certain trial

    baseline = np.mean(fz[:10]) # check the first 10 samples and get their mean value
    if abs(baseline) > 10:  # if unloaded plate records more/less than 10N, I need to zero it:
        fz = fz - baseline

    return fz

def dj(data, fz, fs, body_mass):
    # This function computes the drop jump metreics as described in the paper
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    import numpy as np

    fz = zeroing(fz) # zero the data, if needed
    # the line below is a custom function that allows to click on a plot that combines force plate GRF data with knee joint angle
    # It allows for selecting the peak in knee angular velocity, i.e. the transition from impact-dominated to braking-dominated phase
    # if no impact dominated phase is persent, one should not click, but sumply close the figure
    max_loc = impact_peak(data, fz, fs)

    g = 9.81
    unloaded_start = 0
    # Line below: when trimming the force pate data I took the whole contact time plus a 20% of its duration BEFORE initial contact
    # --> the first 1/6 of the samples (~17%) corresponds to the unloaded force plate
    unloaded_end = int(len(fz) * 0.1) # for subsequent analysis I use from start to  10% of the signal (by definitiion all unloaded)

    #############################################################################
    # finding time stamps
    body_weight = body_mass * 9.81
    threshold = np.mean(fz[unloaded_start: unloaded_end]) + 5 * np.std(fz[unloaded_start:unloaded_end]) # determining on- off plate threshold
    peak_force_ind = np.argmax(fz) # finding index of max value in force curve
    ground_contact = np.where(fz[unloaded_end:peak_force_ind] < threshold)[0][-1] + unloaded_end # last sample < threshold before force increases
    takeoff = np.where(fz[peak_force_ind:] < threshold)[0][0] + peak_force_ind # like above
    ecc_start = np.where(fz[ground_contact:] > body_weight + threshold)[0][0] + ground_contact # where force first exceeds body weight
    conc_end = np.where(fz[ground_contact:] > body_weight + threshold)[0][-1] + ground_contact # where force returns below body weight
    act_ecc_fstart = ecc_start if max_loc <= ecc_start  else max_loc # find start of braking dominated ecc phase: max_loc is the peak in knee angular velocity upon impact, see paper
    pp = 0 if max_loc <= ecc_start else 1 # set pp to 0 if no impact-dominated is present, else set it to 1
    #############################################################################
    # computing COM kinematics
    com_acc = fz[ground_contact: takeoff] / body_mass - g  # get COM net acceleration [m/s2]
    com_vel_nobc = np.cumsum(com_acc * (1 / fs))  # integrate to get velocity

    #I need to implement a boundary cond to say "at takeoff, vertical disp of COM = X!"
    takeoff_disp = 0.05 # [m] vertical COM disp at takeoff, i.e. from plantar flexion (0 = flat foot)
    zero_disp_where = np.argmin([abs((np.sum(com_vel_nobc - com_vel_nobc[i]) * (1/fs)) - takeoff_disp)
                                 for i in np.arange(len(com_vel_nobc))])

    com_vel = com_vel_nobc - com_vel_nobc[zero_disp_where]  # shifting down velocity based on selected boundary condition
    com_vel_pass = com_vel[ecc_start - ground_contact] # COM velocity at start of impact- dominant ecc phase
    com_vel_att = com_vel[act_ecc_fstart - ground_contact] # same for braking-dominant
    print(com_vel_pass)

    com_disp = np.cumsum(com_vel * (1 / fs))  # takeoff_disp  integrate to get displacement
    conc_start = np.argmin(abs(com_vel)) + ground_contact # concentric phase starts at the lowest point of COM trajectory

    #############################################################################
    # computing forces and kinetics
    net_force_norm = fz / body_mass - g # units : [N/kg]
    avg_act_ecc_f = np.mean(net_force_norm[act_ecc_fstart : conc_start]) # [N/kg]; average force during braking-dominant ec phase
    avg_con_f = np.mean(net_force_norm[conc_start : conc_end]) # [N/kg] same for concentric phase

    #############################################################################
    # computing jump related metrics
    takeoff_vel = (np.sum(fz[conc_start: conc_end] - body_weight) / fs) / body_mass # theoretically this value coincides with last value of com_vel timeseries
    ttto = (conc_end - ecc_start)/fs # in s; this is what we call BWct in the paper, i.e. the "sub-portion" of CT where force is >= body weight
    jump_height = takeoff_vel**2 / (2*g) + takeoff_disp # computed adding initial vertical displ due to planter flexion (5 cm, specified above)
    rsi = jump_height / ttto # adimensional
    com_disp_min = np.min(com_disp)  # in m, lowest point of COM trajectory with respect to quiet standing

    #############################################################################
    # computing power related metrics
    delta_sample = act_ecc_fstart - ground_contact # com_vel and fz time series were not synchronnized previously, I need this to do that
    power = fz[act_ecc_fstart : conc_end] * com_vel[delta_sample: conc_end - ground_contact] / body_mass  # W/kg; ho sottratto bw perche' quando sto fermo produco 0 forza
    max_power, vel_max_power = np.max(power), com_vel[delta_sample:][power.argmax()] # W/kg
    time_min_power, time_max_power = power.argmin() + act_ecc_fstart, power.argmax() + act_ecc_fstart
    avg_pos_power = np.mean(power[conc_start - act_ecc_fstart :]) # average power during concentic phase [W/kg], if necessary
    peak_power_instant = (time_max_power - conc_start) / (conc_end - conc_start) * 100 #% of concentric phase

    #############################################################################
    #computing stiffness related metrics
    k_leg = - net_force_norm[ecc_start : conc_start].max() / com_disp_min #[N/m/kg]

    #############################################################################
    # computing phase-duration related metrics, if necessary
    conc_dur = (conc_end - conc_start) / fs
    ecc_dur = (conc_start- ecc_start) / fs
    act_ecc_dur = ecc_dur if pp ==0 else (conc_start - act_ecc_fstart) / fs
    pass_ecc_dur = ecc_dur - act_ecc_dur

    time = np.arange(len(fz)) / fs # for the plot below - remove triple quotes to activate the code
    matplotlib.use('TkAgg')
    """
    plt.figure()
    plt.plot(time, fz, linewidth=3, color='k', label='Vertical force')
    plt.plot(ground_contact/fs, fz[ground_contact], markersize = 16, marker = 's', color = 'k')
    plt.plot(takeoff / fs, fz[takeoff], markersize=16, marker='s', color='k', label = 'ground contact / takeoff')
    plt.plot(ecc_start / fs, fz[ecc_start], markersize=16, marker='^', color='k', label = 'ecc start / conc end')
    plt.plot(conc_end / fs, fz[conc_end], markersize=16, marker='^', color='k')
    plt.xlabel('Time [s]', fontsize=45)
    plt.ylabel('GRF [N]', fontsize=45)

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.legend(fontsize = 35)
    plt.show()
    """

    """
    # Create figure and first axis
    time = np.arange(len(fz)) / fs
    # Plot first series (left y-axis)

    matplotlib.use('TkAgg')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    l1, = ax1.plot(time , fz / body_weight,  linestyle='-', linewidth=2.5, color='k', label='Force')
    #l2, = ax1.plot(time[ground_contact : takeoff], com_disp, linestyle='--', linewidth=2.5, color='k', label='COM velocity')
    #l3, = ax2.plot(time[delta_sample: ], power, linestyle=':', linewidth=2.5, color='k', label='Power')
    #l4, = ax2.plot(time[time_min_power + delta_sample], power[time_min_power], 'o', markersize = 14,  color='k', label = 'max/min power')
    #l5 = ax1.axvline(x = time[zero_disp_where], linestyle='-', color='k', label = 'Eccentric to concentric')
    ax2.plot(time[ground_contact : takeoff], com_disp, linestyle='-', linewidth=2.5, color='k', label='disp')

    
    ax1.set_xlabel('Time', fontsize=30)
    ax1.set_ylabel('Vertical force [BW]', fontsize=30)
    ax1.tick_params(axis='both', labelsize=30)

    ax2.set_ylabel('displacement', fontsize=30)
    ax2.tick_params(axis='both', labelsize=30)

    #lines = [l1, l2, l3, l4, l5]
    #labels = [l.get_label() for l in lines]
    #ax1.legend(lines, labels, loc='lower right', fontsize=25)

    #plt.title('Time ratio: ' + str(np.round(abs((power.argmax() - power.argmin()) / fs) / ttto * 100, 1)),
              #fontsize = 40)
    plt.show()
    """



    return list([rsi, jump_height, ttto, com_disp_min,
            pass_ecc_dur, act_ecc_dur, conc_dur,
            k_leg,
            avg_act_ecc_f, avg_con_f,
            com_vel_pass, com_vel_att,
            avg_pos_power, max_power, peak_power_instant])


def impact_peak(data, fz, fs):
    # This function allows for the identification of transition from impact- to braking-dominant eccentric phase
    # The function creates a plot with 2 y-axis, plotting GRF and knee angular velocity
    # to select the transition instant, it is necessary to left-click ONCE on the angular velocity line, if peaks (associated with the transition) are present
    # at this point, a black dot should appear on the selected point. The user may now close the plot
    # If no peaks are present, the user may close the plot without clicking anywhere
    
    b, a = signal.butter(2, 50 / (fs / 2), 'Lowpass') 
    angvel = np.diff(data.value[data.quantity == 'knee_fe']) * fs # computing 1st derivative for knee flex-ext
    angvel = signal.filtfilt(b, a, angvel) # low-pass filtering angular velocity at 50Hz
    fig, ax1 = plt.subplots()

    # Left y-axis
    line1, = ax1.plot(fz, label='GRF', linewidth=3, color='black')
    ax1.set_xlabel('Samples', fontsize=45)
    ax1.set_ylabel('GRF [N]', fontsize=45)

    # Right y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot( angvel, color='black', linestyle='--', label='Knee ang vel', linewidth=3)
    ax2.set_ylabel('Knee angular velocity [°/s]', fontsize=45)
    # ax1.set_title("Click on approximate min location (or close window to exit)")

    ax1.tick_params(axis='both', labelsize=40)
    ax2.tick_params(axis='y', labelsize=40)
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize = 35)
    max_loc = None


    while True:

        print("Click on plot on approximate location of local minimum after passive peak(s)")
        pts = plt.ginput(1, timeout=-1) # waits for ONE mouse click

        if len(pts) == 0: # if user closes window
            print("No selection. Exiting.")
            max_loc = None
            break

        x_click = int(pts[0][0])
        window = [x_click - 20,  x_click + 20]
        sign_diff = np.sign(np.diff(angvel[window[0]: window[1]]))
        maxs_window = np.where(np.diff(sign_diff) == -2)[0] + 1

        if len(maxs_window) == 0:
            max_loc = -1
            break
            return max_loc

        maxs_window_abs = [max_window + window[0] for max_window in maxs_window]
        max_loc = maxs_window_abs[np.argmax(angvel[[maxs_window_abs]])]
        ax2.plot(max_loc, angvel[max_loc], 'o', color = 'black', markersize = 16)
        plt.show()
        fig.canvas.draw_idle()
        break

    if max_loc == None:
        max_loc = -1
    return max_loc


def jump_comparison(df, j1, j2):
    jump1 = df[(df.subject == j1[0]) & (df.height == j1[1]) & (df.trial == j1[2])]
    jump2 = df[(df.subject == j2[0]) & (df.height == j2[1]) & (df.trial == j2[2])]

    if jump1.rsi.values > jump2.rsi.values:
        data1 = jump1
        data2 = jump2
        print('Jump 1 has higher RSI')
    else:
        data1 = jump2
        data2 = jump1
        print('Jump 2 has higher RSI')

    if (data1.jump_height.values > data2.jump_height.values) & (data1.ttto.values < data2.ttto.values):
        print('Better RSI due to higher jump height and shorter contact time')

        # Jump height
        print('\nJump height: ')
        print('By definition, jump height is determined by takeoff velocity, computed with the impulse method.')
        if (data1.conc_dur.values > data2.conc_dur.values) & (data1.avg_con_force.values > data2.avg_con_force.values):
            print('Higher takeoff velocity due to higher avg concentric force and longer concentric phase duration')
            print('Higher avg concentric force due to higher muscle forces produced. '
                  'Longer concentric phase duration despite higher COM upwards acceleration was due to larger COM downwards displacement')

        elif (data1.conc_dur.values <= data2.conc_dur.values) & (data1.avg_con_force.values > data2.avg_con_force.values):
            print('Higher takeoff velocity due to higher avg concentric force that outweights shorter concentric phase duration')
            if (data1.com_disp_min.values < data2.com_disp_min.values):  # subj 1 less displacement
                print( 'Shorter concentric phase duration due to higher COM acceleration and lower COM downwards displacement')
            else:
                print( 'Shorter concentric phase duration due to higher COM acceleration that outweighted larger COM downwards displacement')


        elif (data1.conc_dur.values > data2.conc_dur.values) & (data1.avg_con_force.values < data2.avg_con_force.values):
            print('Higher takeoff velocity due to longer concentric duration that outweights lower concentric force')

            if (data1.com_disp_min.values > data2.com_disp_min.values):  # subj 1 more displacement
                print('Longer concentric phase duration due to lower COM acceleration and larger COM downwards displacement')
            else:
                print('Longer concentric phase duration due to lower COM acceleration that outweighted lower COM downwards displacement')

        # TTTO
        print('\nTTTO: ')

        if (data1.conc_dur.values > data2.conc_dur.values):
            print('Shorter contact time due to shorter eccentric phase that outweights longer concentric phase. Concentric phase duration discussed above.')

        if (data1.conc_dur.values <= data2.conc_dur.values):

            if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values <= data2.pass_ecc_dur.values + data2.act_ecc_dur.values):
                print('Shorter contact time due to both concentric and eccentric phases shorter. Concentric phase duration discussed above.')
            else:
                print('Shorter TTcontact time due to shorter concentric phase that outweights longer eccentric phase. Concentric phase duration discussed above.')

        if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values <= data2.pass_ecc_dur.values + data2.act_ecc_dur.values):

            if (data1.pass_ecc_dur.values <= data2.pass_ecc_dur.values) & (data1.act_ecc_dur.values > data2.act_ecc_dur.values):
                print('Eccentric phase shorter due to shorter impact-dominant phase that outweights longer braking-dominant phase : related to muscle - tendon properties and kinematics')

            elif (data1.act_ecc_dur.values <= data2.act_ecc_dur.values):
                if data1.pass_ecc_dur.values <= data2.pass_ecc_dur.values:
                    print('Eccentric phase shorter due to shorter impact- and braking-dominant phase')
                    print('Impact-dominant phase shorter: related to muscle - tendon properties and kinematics')
                else:
                    print('Eccentric phase shorter due to shorter braking-dominant phase')

                if (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to lower COM initial downwards velocity and higher avg braking eccentric force')
                    if data1.k_avg.values > data2.k_avg.values:
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')
                    else:
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Lower leg stiffness combined with higher force resulted in larger COM displacement')
                        else:
                            print('Interestingly, lower leg stiffness combined with higher force resulted in lower COM displacement')


                elif (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to lower COM initial downwards velocity that outweights lower avg braking eccentric force')
                    if data1.k_avg.values < data2.k_avg.values:
                        print('Lower avg braking eccentric force due to lower leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of lower leg stiffness and lower avg braking eccentric force resulted in larger COM displacement')
                        else:
                            print('This specific combination of lower leg stiffness and lower avg braking eccentric force resulted in lower COM displacement')

                    else:
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Interestingly, larger COM displacement despite higher stiffness and lower force')
                        else:
                            print('Lower COM displacement resulting from higher stiffness and lower force')



                elif (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to higher avg braking eccentric force that outweights larger COM initial downwards velocity')
                    if data1.k_avg.values > data2.k_avg.values:
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')

                    else:
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Higher force combined with lower leg stiffness resulted in larger COM displacement')
                        else:
                            print('Interestingly, higher force and lower leg stiffness resulted in lower COM displacement')

        if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values > data2.pass_ecc_dur.values + data2.act_ecc_dur.values):

            if (data1.pass_ecc_dur.values > data2.pass_ecc_dur.values) & (data1.act_ecc_dur.values <= data2.act_ecc_dur.values):
                print('Eccentric phase longer due to longer impact-dominant phase that outweights shorter braking-dominant phase : related to muscle - tendon properties and kinematics')

            elif (data1.act_ecc_dur.values > data2.act_ecc_dur.values):
                if data1.pass_ecc_dur.values > data2.pass_ecc_dur.values:
                    print('Eccentric phase longer due to longer braking-dominant and impact-dominant phase')
                    print('Impact-dominant phase shorter: related to muscle - tendon properties and kinematics')
                else:
                    print('Eccentric phase longer due to longer braking dominant  phase')

                if (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to higher COM initial downwards velocity and lower avg braking eccentric force')
                    if data1.k_avg.values < data2.k_avg.values:
                        print('Lower avg braking-dominant eccentric force due to lower leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in smaller COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                    else:  # higher leg stiffness
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Higher leg stiffness combined with lower force resulted in lower COM displacement')
                        else:
                            print('Interestingly, higher leg stiffness combined with lower force resulted in larger COM displacement')


                elif (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to higher COM initial downwards velocity that outweights higher avg braking eccentric force')
                    if data1.k_avg.values > data2.k_avg.values:  # higher stiffness
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('This specific combination of higher leg stiffness and higher avg braking eccentric force resulted in lower COM displacement')
                        else:
                            print('This specific combination of higher leg stiffness and higher avg braking eccentric force resulted in higher COM displacement')

                    else:  # lower stiffness
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Interestingly, lower COM displacement despite lower stiffness and higher force')
                        else:
                            print('higher COM displacement resulting from lower stiffness and higher force')



                elif (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to lower avg braking eccentric force that outweights lower COM initial downwards velocity')
                    if data1.k_avg.values < data2.k_avg.values:  # lower stiffness
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')

                    else:  # higher stiffness
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Lower force combined with higher leg stiffness resulted in lower COM displacement')
                        else:
                            print('Interestingly, lower force and higher leg stiffness resulted in higher COM displacement')



    elif (data1.jump_height.values > data2.jump_height.values) & (data1.ttto.values > data2.ttto.values):
        print('Better RSI due to higher jump height that outweighted longer  contact time')

        # Jump height
        print('\nJump height: ')
        if (data1.conc_dur.values > data2.conc_dur.values) & (data1.avg_con_force.values > data2.avg_con_force.values):
            print('Better jump height due to higher avg concentric force and longer concentric phase duration')
            print('Higher avg concentric force due to higher muscle forces produced. '
                  'Longer concentric phase duration despite higher COM upwards acceleration was due to larger COM downwards displacement')

        elif (data1.conc_dur.values <= data2.conc_dur.values) & (data1.avg_con_force.values > data2.avg_con_force.values):
            print('Better jump height due to higher avg concentric force that outweights shorter concentric phase duration')
            if (data1.com_disp_min.values < data2.com_disp_min.values):  # subj 1 less displacement
                print('Shorter concentric phase duration due to higher COM acceleration and lower COM downwards displacement')
            else:
                print('Shorter concentric phase duration due to higher COM acceleration that outweighted larger COM downwards displacement')


        elif (data1.conc_dur.values > data2.conc_dur.values) & (data1.avg_con_force.values < data2.avg_con_force.values):
            print('Better jump height due to longer concentric duration that outweights lower concentric force')

            if (data1.com_disp_min.values > data2.com_disp_min.values):  # subj 1 more displacement
                print('Longer concentric phase duration due to lower COM acceleration and larger COM downwards displacement')
            else:
                print('Longer concentric phase duration due to lower COM acceleration that outweighted lower COM downwards displacement')

        # TTTO
        print('\nContact time: ')

        if (data1.conc_dur.values <= data2.conc_dur.values):
            print('Shorter contact time due to longer eccentric phase that outweights shorter concentric phase. Concentric phase duration discussed above.')

        if (data1.conc_dur.values > data2.conc_dur.values):

            if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values > data2.pass_ecc_dur.values + data2.act_ecc_dur.values):
                print('Longer contact time due to both concentric and eccentric phases longer. Concentric phase duration discussed above.')
            else:
                print('Longer contact time due to longer concentric phase that outweights shorter eccentric phase. Concentric phase duration discussed above.')

        if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values <= data2.pass_ecc_dur.values + data2.act_ecc_dur.values):

            if (data1.pass_ecc_dur.values <= data2.pass_ecc_dur.values) & (data1.act_ecc_dur.values > data2.act_ecc_dur.values):
                print('Eccentric phase shorter due to shorter impact-dominant phase that outweights longer braking-dominant phase : related to muscle - tendon properties and kinematics')

            elif (data1.act_ecc_dur.values <= data2.act_ecc_dur.values):
                if data1.pass_ecc_dur.values <= data2.pass_ecc_dur.values:
                    print('Eccentric phase shorter due to shorter impact-dominant and braking-dominant phase')
                    print('Impact-dominant eccentric phase shorter: related to muscle - tendon properties and kinematics')
                else:
                    print('Eccentric phase shorter due to shorter braking-dominant phase')

                if (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to lower COM initial downwards velocity and higher avg braking eccentric force')
                    if data1.k_avg.values > data2.k_avg.values:
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')
                    else:
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Lower leg stiffness combined with higher force resulted in larger COM displacement')
                        else:
                            print('Interestingly, lower leg stiffness combined with higher force resulted in lower COM displacement')


                elif (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to lower COM initial downwards velocity that outweights lower avg braking eccentric force')
                    if data1.k_avg.values < data2.k_avg.values:
                        print('Lower avg braking eccentric force due to lower leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of lower leg stiffness and lower avg braking eccentric force resulted in larger COM displacement')
                        else:
                            print('This specific combination of lower leg stiffness and lower avg braking eccentric force resulted in lower COM displacement')

                    else:
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Interestingly, larger COM displacement despite higher stiffness and lower force')
                        else:
                            print('Lower COM displacement resulting from higher stiffness and lower force')



                elif (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to higher avg braking eccentric force that outweights larger COM initial downwards velocity')
                    if data1.k_avg.values > data2.k_avg.values:
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')

                    else:
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Higher force combined with lower leg stiffness resulted in larger COM displacement')
                        else:
                            print('Interestingly, higher force and lower leg stiffness resulted in lower COM displacement')

        if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values > data2.pass_ecc_dur.values + data2.act_ecc_dur.values):

            if (data1.pass_ecc_dur.values > data2.pass_ecc_dur.values) & (data1.act_ecc_dur.values <= data2.act_ecc_dur.values):
                print('Eccentric phase longer due to longer impact-dominant phase that outweights shorter braking-dominant phase : related to muscle - tendon properties and kinematics')

            elif (data1.act_ecc_dur.values > data2.act_ecc_dur.values):
                if data1.pass_ecc_dur.values > data2.pass_ecc_dur.values:
                    print('Eccentric phase longer due to longer braking-dominant and impact-dominant phase')
                    print('impact-dominant phase shorter: related to muscle - tendon properties and kinematics')
                else:
                    print('Eccentric phase longer due to longer braking-dominant  phase')

                if (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to higher COM initial downwards velocity and lower avg braking eccentric force')
                    if data1.k_avg.values < data2.k_avg.values:
                        print('Lower avg braking eccentric force due to lower leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in smaller COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                    else:  # higher leg stiffness
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Higher leg stiffness combined with lower force resulted in lower COM displacement')
                        else:
                            print('Interestingly, higher leg stiffness combined with lower force resulted in larger COM displacement')


                elif (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to higher COM initial downwards velocity that outweights higher avg braking eccentric force')
                    if data1.k_avg.values > data2.k_avg.values:  # higher stiffness
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('This specific combination of higher leg stiffness and higher avg braking eccentric force resulted in lower COM displacement')
                        else:
                            print('This specific combination of higher leg stiffness and higher avg braking eccentric force resulted in higher COM displacement')

                    else:  # lower stiffness
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Interestingly, lower COM displacement despite lower stiffness and higher force')
                        else:
                            print('Higher COM displacement resulting from lower stiffness and higher force')



                elif (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to lower avg braking eccentric force that outweights lower COM initial downwards velocity')
                    if data1.k_avg.values < data2.k_avg.values:  # lower stiffness
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')

                    else:  # higher stiffness
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Lower force combined with higher leg stiffness resulted in lower COM displacement')
                        else:
                            print('Interestingly, lower force and higher leg stiffness resulted in higher COM displacement')


    elif (data1.jump_height.values < data2.jump_height.values) & (data1.ttto.values < data2.ttto.values):
        print('Better RSI due to shorter contact time that outweighted lower jump height')

        # Jump height
        print('\nJump height: ')
        if (data1.conc_dur.values <= data2.conc_dur.values) & (data1.avg_con_force.values < data2.avg_con_force.values):
            print('Lower jump height due to lower avg concentric force and shorter concentric phase duration')
            print('Lower avg concentric force due to lower muscle forces produced. '
                  'Shorter concentric phase duration despite lower COM acceleration was due to lower COM downwards displacement')

        elif (data1.conc_dur.values > data2.conc_dur.values) & (data1.avg_con_force.values < data2.avg_con_force.values):
            print('Lower jump height due to lower avg concentric force that outweights longer concentric phase duration')
            if (data1.com_disp_min.values > data2.com_disp_min.values):
                print('Longer concentric phase duration due to lower COM acceleration and higher COM downwards displacement')
            else:
                print('Longer concentric phase duration due to lower COM acceleration that outweighted lower COM downwards displacement')


        elif (data1.conc_dur.values <= data2.conc_dur.values) & (data1.avg_con_force.values > data2.avg_con_force.values):
            print('Lower jump height due to shorter concentric duration that outweights higher concentric force')

            if (data1.com_disp_min.values < data2.com_disp_min.values):
                print('Shorter concentric phase duration due to higher COM acceleration and lower COM downwards displacement')
            else:
                print('Shorter concentric phase duration due to higher COM acceleration that outweighted larger COM downwards displacement')

        # TTTO
        print('\nContact time: ')

        if (data1.conc_dur.values > data2.conc_dur.values):
            print('Shorter contact time due to shorter eccentric phase that outweights longer concentric phase. Concentric phase duration discussed above.')

        if (data1.conc_dur.values <= data2.conc_dur.values):

            if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values <= data2.pass_ecc_dur.values + data2.act_ecc_dur.values):
                print('Shorter contact time due to both concentric and eccentric phases shorter. Concentric phase duration discussed above.')
            else:
                print('Shorter contact time due to shorter concentric phase that outweights longer eccentric phase. Concentric phase duration discussed above.')

        if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values <= data2.pass_ecc_dur.values + data2.act_ecc_dur.values):

            if (data1.pass_ecc_dur.values <= data2.pass_ecc_dur.values) & (data1.act_ecc_dur.values > data2.act_ecc_dur.values):
                print('Eccentric phase shorter due to shorter impact-dominant phase that outweights longer braking-dominant phase : related to muscle - tendon properties and kinematics')

            elif (data1.act_ecc_dur.values <= data2.act_ecc_dur.values):
                if data1.pass_ecc_dur.values <= data2.pass_ecc_dur.values:
                    print('Eccentric phase shorter due to shorter braking-dominant and impact-dominant phase')
                    print('impact-dominant phase shorter: related to muscle - tendon properties and kinematics')
                else:
                    print('Eccentric phase shorter due to shorter braking-dominant phase')

                if (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to lower COM initial downwards velocity and higher avg braking eccentric force')
                    if data1.k_avg.values > data2.k_avg.values:
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')
                    else:
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Lower leg stiffness combined with higher force resulted in larger COM displacement')
                        else:
                            print('Interestingly, lower leg stiffness combined with higher force resulted in lower COM displacement')


                elif (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to lower COM initial downwards velocity that outweights lower avg braking eccentric force')
                    if data1.k_avg.values < data2.k_avg.values:
                        print('Lower avg braking eccentric force due to lower leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of lower leg stiffness and lower avg braking eccentric force resulted in larger COM displacement')
                        else:
                            print('This specific combination of lower leg stiffness and lower avg braking eccentric force resulted in lower COM displacement')

                    else:
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Interestingly, larger COM displacement despite higher stiffness and lower force')
                        else:
                            print('Lower COM displacement resulting from higher stiffness and lower force')



                elif (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase shorter due to higher avg braking eccentric force that outweights larger COM initial downwards velocity')
                    if data1.k_avg.values > data2.k_avg.values:
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')

                    else:
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print('Higher force combined with lower leg stiffness resulted in larger COM displacement')
                        else:
                            print('Interestingly, higher force and lower leg stiffness resulted in lower COM displacement')

        if (data1.pass_ecc_dur.values + data1.act_ecc_dur.values > data2.pass_ecc_dur.values + data2.act_ecc_dur.values):

            if (data1.pass_ecc_dur.values > data2.pass_ecc_dur.values) & (data1.act_ecc_dur.values <= data2.act_ecc_dur.values):
                print('Eccentric phase longer due to longer impact-dominant phase that outweights shorter braking-dominant phase : related to muscle - tendon properties and kinematics')

            elif (data1.act_ecc_dur.values > data2.act_ecc_dur.values):
                if data1.pass_ecc_dur.values > data2.pass_ecc_dur.values:
                    print('Eccentric phase longer due to longer braking-dominant and impact-dominant phase')
                    print('impact-dominant phase shorter: related to muscle - tendon properties and kinematics')
                else:
                    print('Eccentric phase longer due to longer braking-dominant  phase')

                if (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to higher COM initial downwards velocity and lower avg braking eccentric force')
                    if data1.k_avg.values < data2.k_avg.values:
                        print('Lower avg braking eccentric force due to lower leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('This specific combination of force and leg stiffness resulted in smaller COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in larger COM displacement')
                    else:  # higher leg stiffness
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Higher leg stiffness combined with lower force resulted in lower COM displacement')
                        else:
                            print('Interestingly, higher leg stiffness combined with lower force resulted in larger COM displacement')


                elif (data1.com_vel_att.values > data2.com_vel_att.values) & (data1.avg_act_ecc_force.values > data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to higher COM initial downwards velocity that outweights higher avg braking eccentric force')
                    if data1.k_avg.values > data2.k_avg.values:  # higher stiffness
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('This specific combination of higher leg stiffness and higher avg braking eccentric force resulted in lower COM displacement')
                        else:
                            print('This specific combination of higher leg stiffness and higher avg braking eccentric force resulted in higher COM displacement')

                    else:  # lower stiffness
                        print('Higher avg braking eccentric force despite lower leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Interestingly, lower COM displacement despite lower stiffness and higher force')
                        else:
                            print('higher COM displacement resulting from lower stiffness and higher force')



                elif (data1.com_vel_att.values < data2.com_vel_att.values) & (data1.avg_act_ecc_force.values < data2.avg_act_ecc_force.values):
                    print('Braking-dominant eccentric phase longer due to lower avg braking eccentric force that outweights lower COM initial downwards velocity')
                    if data1.k_avg.values < data2.k_avg.values:  # lower stiffness
                        print('Higher avg braking eccentric force due to higher leg stiffness, related to muscle-tendon properties and kinematics')
                        if data1.com_disp_min.values > data2.com_disp_min.values:
                            print( 'This specific combination of force and leg stiffness resulted in larger COM displacement')
                        else:
                            print('This specific combination of force and leg stiffness resulted in lower COM displacement')

                    else:  # higher stiffness
                        print('Lower avg braking eccentric force despite higher leg stiffness')
                        if data1.com_disp_min.values < data2.com_disp_min.values:
                            print('Lower force combined with higher leg stiffness resulted in lower COM displacement')
                        else:
                            print('Interestingly, lower force and higher leg stiffness resulted in higher COM displacement')



def compute_signature_fast(X, rsi):

    n = len(rsi)
    dX = X[:, None, :] - X[None, :, :]
    dRSI = rsi[:, None] - rsi[None, :]
    #dX_score = np.sum(dX, axis=2)
    triu = np.triu_indices(n, k=1)
    dX_score = dX[triu]
    dRSI = dRSI[triu]
    concordant = np.sign(dX_score) == np.sign(dRSI[:, None])
    return np.mean(concordant, axis = 0)


def permutation_test_fast(df, rsi_col = 3, feature_cols = [4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15], n_perm=1000, seed=42):

    np.random.seed(seed)
    X = df.iloc[:, feature_cols].values
    rsi = df.iloc[:, rsi_col].values
    obs = compute_signature_fast(X, rsi)
    null_dist = np.empty((n_perm, len(feature_cols)))

    for b in range(n_perm):

        rsi_perm = np.random.permutation(rsi)
        null_dist[b, :] = compute_signature_fast(X, rsi_perm)

    #p_value_row = (np.sum(null_dist >= obs, axis = 0) + 1) / (n_perm + 1)
    p_values = (np.where(obs > 0.5, (null_dist >= obs).sum(axis=0), (null_dist <= obs).sum(axis=0) ) + 1) / (n_perm + 1)

    sig_row = p_values < 0.01
    return sig_row