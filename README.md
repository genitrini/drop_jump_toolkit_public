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



## Input data description
In the PNG file it is possible to see how the data frame used for the analysis looks like. The first three columns (left to right) contain information regarding the participant's ID, the trial and 
box height in centimeters. 
The column "quantity" indicates the variable (which can be a GRF or a joint angle). 
The column "side" indicates whether the data are from the left or right limb. 
The column "body mass" refers to the body mass of the subject; this can be obtained weighting the subject on a scale prior to the DJ tasks or (as in the current work) dividing by the gravity acceleration the recorded force during quite standing after landing on the plate. 
The column 'time' refers to the time stamps of the recording; in our case, those come from the Myo Research software. 
Finally, the column 'value' is the numerical value of that given quantity.

