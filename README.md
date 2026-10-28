# Scientific algorithm for drop jump analysis and single athlete profiling

## Description
This software analyses ground reaction force (GRF) time series combined with joint angle data in order to provide a thorough analysis of drop jump (DJ) tasks using a single force plate 
and inertial sensors for kinematics. In the present work, an AMTI force plate and Noraxon IMU sensors were used. 
The hardware was connected to the Myo Research software from Noraxon (Scottsdale, AZ, USA)

## Requirements
- Microsoft Excel
- Python 3.10 or newer

## We provide:
- A screenshot from the data frame to be prompted to the Python scripts
- The python code needed to
  -  analyze the DJs
  - provide a comprehensive comparison between DJ instances
  - determine which metrics are predictive of performance for a given athlete


## Pre-processing
Prior to running the algorithm we provide, a few preliminary steps are necessary.
In fact, for Python it is easier to deal with .pkl files as compared to .xlsx or .csv, especially when these need to be uploaded itaratively and many subjects are to be analyzed. 

To do that, every .CSV file (exported from Noraxon Myo Research) needs to be uploaded in Python the first time by running the "csv_to_pkl" script. This will return a .pkl vile for every .csv present in the directory prompted to the script.

Subsequently, one needs to run the "preparation" script. This trims the larger files to immediately before and after the contact time between athlete and force plate, discarding redundant data. It also concatenates the data from multiple instances from a given athlete in one data frame per person and, finally, concatenates the data frames from all athletes into one long data frame where all data is condensed. This long data frame may now be used as input for the other script, as explained in the next sections.


## Input data description
In the PNG file it is possible to see how the data frame used for the analysis looks like. The first three columns (left to right) contain information regarding the participant's ID, the trial and box height in centimeters. 

The column "quantity" indicates the variable (which can be a GRF or a joint angle). 

The column "side" indicates whether the data are from the left or right limb. 

The column "body mass" refers to the body mass of the subject; this can be obtained weighting the subject on a scale prior to the DJ tasks or (as in the current work) dividing by the gravity acceleration the recorded force during quite standing after landing on the plate.

The column 'time' refers to the time stamps of the recording; in our case, those come from the Myo Research software. 

Finally, the column 'value' is the numerical value of that given quantity.

