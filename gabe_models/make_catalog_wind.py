#! /usr/bin/env python
# 
# Program: MAKE_CATALOG.py
#
# Author: Nick Lee
#
# Usage: ./make_catalog.py [-vh]
#
# Description: Insert Description Here
#
# Revision History:
#    Date        Vers.    Author            Description
#    12/24/14    1.0a0    Nick Lee          First checked in
#    1/1/15               Gabriel Dima      At lines 49, 51, 53, 71, 73, 75, 190 and 300 changed the filenames by removing
#                                           computer specific locations. Now the files are read/created in the same folder where
#                                           the functions run.
#                                           Line 301 modified table header to show the name of the sample chosen rather than Two Rock.
#                                           Line 316 added check to verify if actual treatment was applied. 
#                                           Line 132 and 137 changed the num_max_temp check from > to == 
#To Do:
#    
#

# Import Libraries
import numpy as np
import pylab as pl
import scipy as sp
import math
import pdb
import pandas as pd

# Constants - all prefixed with a lower-case 'k'

# Globals - all prefixed with a lower-case 'g'

####################
#### Functions #####
####################
## Get Unique elements of an array while retaining order. 
def ordered_uniq(seq, idfun=None): 
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked

#####################
# Read in weather data from various catalogs.
def get_weather(sample_name):

    # Read in data 
    if sample_name=='two_rock':
        filename='two_rock.txt'
    elif sample_name=='frei':
        filename='frei.txt'
    else:
        filename='laguna_upper.txt'

    # Array of dates & times
    time = np.genfromtxt(filename,dtype=np.str,usecols=[0,1],skiprows=1)

    # Array of weather information
    # Temperature, Precipitation, Wetness, Humidity
    weather = np.genfromtxt(filename,usecols=[2,3,4,5],skiprows=1)
    
    # Outputs
    return time, weather 

#####################
# Read in LAMP data from various catalogs
def get_LAMP(sample_name):

    # Set filenames
    if sample_name=='two_rock':
        filename='two_rock_LAMP.txt'
    elif sample_name=='frei':
        filename='frei_LAMP.txt'
    else:
        filename='laguna_LAMP.txt'

    # Array of dates (start and end)
    LAMP_dates = np.genfromtxt(filename,usecols=[0,1],dtype=np.str)

    # Array of LAMP Results
    LAMP_results = np.genfromtxt(filename,usecols=[2])

    # Array of treatments
    LAMP_treatments = np.genfromtxt(filename,usecols=[4],dtype=np.str)

    return LAMP_dates, LAMP_results, LAMP_treatments

#####################
'''
Function: GEN_PMI
Generate daily measurement of PMI.
Inputs:
   TIME - 2 column array that contains date & times of weather data, of size (N x 2)
   TEMPERATURE - N-element array that contains corresponding temperature measurements (in Farenheit)
   TIME_BIN - Amount of time (in hours) corresponding to each time interval (Default is 0.25, or 15 min)
Outputs:
   DATES - Ordered list of unique dates, of length [N_OUT]
   PMI - PMI measured at end of each corresponding date, of length [N_OUT]
'''
def gen_PMI(time,temperature,time_bin=0.25):

    # Create ordered list of dates
    dates = ordered_uniq(time[:,0])
    n_days = np.size(dates)

    # Set how many time bins are needed for 6 hours:
    ideal_time_limit = 6/(time_bin*1.)

    # Empty arrays
    #max_temp_flag = np.zeros(n_days)
    #ideal_temp_flag = np.zeros(n_days)
    PMI = np.zeros(n_days)
    PMI_trigger = 0.

    # Loop over dates
    for ii, date in enumerate(dates):
        # Find times that correspond to this date.
        ind_day = np.where(time[:,0]==date)[0]
        # Check if weather ever gets above 95 degrees:
        num_max_temp = np.sum(temperature[ind_day] >= 95)
        # Check number of temperature segments in ideal range 
        num_ideal_temp = np.sum(np.all([temperature[ind_day]>=70, temperature[ind_day]<=85],axis=0))

        # Calculate PMI
        # Handle day 0 separately:
        if ii==0:
            if (num_ideal_temp >= ideal_time_limit and num_max_temp == 0): PMI_temp = 20
            else: PMI_temp = 0
        else:
            PMI_temp = PMI[ii-1]  # Start off with yesterday's PMI
            # Reduce PMI by 10 if temperature above max
            if (num_max_temp > 0): PMI_temp-=10
                
            # Increase PMI by 20 if temperature was in ideal range for 6+ hours
            if (num_ideal_temp >= ideal_time_limit): PMI_temp+=20
            elif (PMI_trigger==0): PMI_temp=0  # Reduce to 0 if PMI not triggered
            else: PMI_temp-=10
                
            # If PMI is greater than 60, and trigger hasn't been set, set trigger.
            if (PMI_temp >= 60 and PMI_trigger==0): PMI_trigger = 1.

        # Make sure PMI isn't outside of 0-100
        if PMI_temp > 100: PMI_temp = 100.
        if PMI_temp < 0: PMI_temp = 0.

        # Save PMI to array
        PMI[ii] = PMI_temp
        #print date, num_ideal_temp, num_max_temp, PMI_temp
    return dates, PMI

#####################
'''
Function: MAKE_TABLE
Generate a table of parameters and answers that can be fed to machine learning algorithm
Inputs:
   SAMPLE_NAME - Name of desired sample. Options are 'two_rock','laguna', or 'frei'.
   TIME_BIN - Amount of time (in hours) corresponding to each time interval (Default is 0.25, or 15 min)

'''
def make_table(sample_name,time_bin=0.25):

    # Get weather data and LAMP data
    time, weather = get_weather(sample_name)
    LAMP_dates,LAMP_results,LAMP_treat = get_LAMP(sample_name)
    n_LAMP = np.size(LAMP_results)

    #######################################
    ### Determine Parameters to include ###
    #######################################
    # Set temperature ranges for binning
    temp_bins = [0,60,65,70,75,80,85,90,95,1000]
    n_temp = np.size(temp_bins)-1
    temp = weather[:,0]

    # Set which Weather factors to use
    weather_str = ['Precipitation', 'Leaf Wetness', 'Humididty']
    weather_flag = [1,1,1]
    n_weather = len(weather_str)

    # Set which Treatments to include
    all_treatments = ['Oil','Mettle','Flint','Kocide','Sulfur','Sylcoat','Switch','Sovran']
    treatment_flag = [1,1,1,1,1,1,1,1]
    n_treat = len(all_treatments)

    #####################################
    ### Open Text file to write Table ###
    #####################################
    tab = open(sample_name+'_param.txt','w')
    tab.write('# Parameters and Answers for '+sample_name+'\n')

    # Create header
    header_str = '# '
    h_count = 1
    for hh,temp_min in enumerate(temp_bins[0:-1]):
        header_str += str(temp_min) + '<T<' + str(temp_bins[hh+1]) + '  '
        tab.write('# Column '+str(h_count)+': Time (hours) in Temperature Range '+str(temp_min) + '<T<' + str(temp_bins[hh+1]) +'\n')
        h_count+=1
    for hh,ww in enumerate(weather_str):
        if weather_flag[hh] == 1:
            header_str += ww + '  '
            tab.write('# Column '+str(h_count)+': Average ' + ww +'\n')
            h_count+=1
    for hh,tt in enumerate(all_treatments):
        if treatment_flag[hh] == 1:
            header_str += tt + '  '
            tab.write('# Column '+str(h_count)+': Treated with ' + tt +' (1 = Treatment used)\n')
            h_count+=1
    tab.write('# Column '+str(h_count)+': LAMP Result (1 = Detection)\n')
    
    # Write Last header
    tab.write(header_str + 'LAMP Result\n')
        
    #######################################
    ### Loop Over All LAMP measurements ###
    #######################################
    # Empty arrays
    temp_arr = np.zeros([n_LAMP,n_temp])  # Temperature bins
    weather_arr = np.zeros([n_LAMP,n_weather])  # Precipitation, Leaf Wetness, Humididty
    treat_arr = np.zeros([n_LAMP,n_treat]) # Treatments

    # Loop over the date ranges for each LAMP measurement
    for ii, end_date in enumerate(LAMP_dates[:,1]):

        # Initialize string to be written to table
        tab_str = ''

        # Find indices of time bins corresponding to date range. 
        start_date = LAMP_dates[ii,0]
        ind_start = np.where(time[:,0] == start_date)[0][0]
        # Assuming that measurement is taken the morning of end date (so we don't count times during end date) 
        ind_end = np.where(time[:,0] == end_date)[0][0]

        # Find amount of time (in hours) in each temperature bin
        hist, bins = np.histogram(temp[ind_start:ind_end],bins=temp_bins)
        temp_arr[ii,:] = hist*time_bin
        
        # Find average Precipitation, Leaf Wetness, Humididty
        avg_weather = np.mean(weather[ind_start:ind_end,1:],axis=0)
        weather_arr[ii,:] = avg_weather

        # Find treatments
        for jj, treat in enumerate(all_treatments):
            if LAMP_treat[ii].find(treat) > -1: treat_arr[ii,jj] = 1
        
        ### Write all results to Table ###
        for jj, num in enumerate(temp_arr[ii,:]): tab_str+='{:7.2f}  '.format(num)
        for jj,num in enumerate(avg_weather):
            if weather_flag[jj] == 1: tab_str+='{:7.3f}  '.format(num)
        for jj,num in enumerate(treat_arr[ii,:]):
            if treatment_flag[jj] == 1: tab_str+='{:1}  '.format(num)
        tab.write(tab_str + str(LAMP_results[ii]) + '\n')
        
    # Close Table
    tab.close()

    # Return Outputs
    return temp_arr, weather_arr, treat_arr


#####################
'''
Function: MODEL_CONC 
Model the concentration of powdery mildew in the air for a single location.
Uses weather data from existing samples and can model collecting for multiple days.

Produces a text table of measurements that can be correctly read by GET_LAMP.

Inputs:
   SAMPLE_NAME - Name of desired sample to use as weather data. Options are 'two_rock','laguna', or 'frei'.
   COLLECTION_TIME - Time in [days] to collect samples over. Default is 2 [days].

Outputs:
   CONCENTRATION - A daily measure of spore concentration in arbitrary units, given in vector of length N_DAYS
   CONC_MEASURE - A daily measure of what a trap would have measured if left in field for COLLECTION_TIME days.
                  If COLLECTION_TIME > 1, this array is still of length N_DAYS.
'''
def model_conc(sample_name,collection_time=2):

    # Get Weather data and Calculate PMI
    time, weather = get_weather(sample_name)
    dates, PMI_arr = gen_PMI(time,weather[:,0])
    n_dates = len(dates)

    # Get LAMP results to determine when treatments occured
    LAMP_dates,LAMP_results,LAMP_treat = get_LAMP(sample_name)
    
    # Set modeling constants
    grow_rate = 2.0  # Growth factor for the pathogen
    treat_eff = 0.5  # Efficiency of spray treatment. Each spray reduces the concentration by this factor
    conc_start = 1.0    # Initial concentration [arbitrary units]

    # Empty arrays
    conc_spore = np.zeros(n_dates)
    conc_int = np.zeros(n_dates)
    conc_measure = np.zeros(n_dates)

    ### Open text file to write results ###
    tab = open(sample_name+'_model_LAMP.txt','w')
    tab.write('# Modeled Concentration measurements from '+sample_name+' data\n')
    tab.write('# Start Date | End Date | Collected Spores | PMI | Treatments\n')
 
    # Loop over all days
    for ii,PMI in enumerate(PMI_arr):

        # Initialize string to be written to table
        tab_str = ''
        
        # Calculate spore growth rate from PMI
        t_pmi = f_t_pmi(PMI)

        # Check if a treatment was applied
        if np.sum(LAMP_dates[:,1]==dates[ii]) >= 1:
            ind_treat = np.where(LAMP_dates[:,1]==dates[ii])[0][0]
            treat_str = LAMP_treat[ind_treat]+'\n'
            if LAMP_treat[ind_treat] != 'None':
                spray_factor = (1.-treat_eff)                  
        else:
            spray_factor = 1.
            treat_str = 'None\n'

        # Calculate Daily Spore concentration and integrated spore concentration
        # Handle day 0 separately 
        if ii == 0:
            # Calculate spore concentration 
            conc_spore[ii]=grow_rate**(1.0/t_pmi)*conc_start*spray_factor
            # Calculate Integrated spore concentration
            conc_int[ii] = t_pmi/math.log(grow_rate)*(conc_spore[ii]-conc_start)
        else:
            # Calculate spore concentration 
            conc_spore[ii]=grow_rate**(1.0/t_pmi)*conc_spore[ii-1]*spray_factor
            # Calculate (daily) Integrated Spore concentration (how many spores collected by a trap today?)
            conc_int[ii]=t_pmi/math.log(grow_rate)*(conc_spore[ii]-conc_spore[ii-1])+(t_pmi/math.log(grow_rate))**2*(1.-spray_factor)*conc_spore[ii-1]*(grow_rate**(1.0/t_pmi)-1.)

        # Calculate number of spores captured by trap since COLLECTION_TIME days ago.
        if ii < collection_time:
            # Add up all the days up til now
            conc_measure[ii] = np.sum(conc_int[0:ii+1])
            # Set start date 
            tab_str+='{:10s}  {:10s}'.format(dates[0],dates[ii])
        else:
            conc_measure[ii] = np.sum(conc_int[ii-collection_time:ii+1])
            tab_str+='{:10s}  {:10s}'.format(dates[ii-collection_time],dates[ii])

        ### Fill out rest of table entry
        tab_str+='  {:8.3f}  {:3.0f}  '.format(conc_measure[ii],PMI)
        tab_str+=treat_str
        tab.write(tab_str)

    # Close Table
    tab.close()

    # Return outputs
    return conc_measure

#####################
'''
Function: MODEL_CONC 
Model the concentration of powdery mildew in the air for an array of locations and include wind information 
Uses weather data from existing samples and can model collecting for multiple days.

Produces a text table of measurements that can be correctly read by GET_LAMP.

Inputs:
   SAMPLE_NAME - Name of desired sample to use as weather data. Options are 'two_rock','laguna', or 'frei'.
   COLLECTION_TIME - Time in [days] to collect samples over. Default is 2 [days].

Outputs:
   CONCENTRATION - A daily measure of spore concentration in arbitrary units, given as a 3d array of size [n,n,N_DAYS]
   CONC_MEASURE - A daily measure of what a trap would have measured if left in field for COLLECTION_TIME days.
                  If COLLECTION_TIME > 1, this array is still of length N_DAYS.
'''

def model_conc_wind(sample_name,collection_time=2):

    # Get Weather data and Calculate PMI
    time, weather = get_weather(sample_name)
    dates, PMI_arr = gen_PMI(time,weather[:,0])
    n_dates = len(dates)
    n = 3   

    # Get LAMP results to determine when treatments occured
    LAMP_dates,LAMP_results,LAMP_treat = get_LAMP(sample_name)
    
    # Set modeling constants
    grow_rate = 2.0  # Growth factor for the pathogen
    treat_eff = 0.0  # Efficiency of spray treatment. Each spray reduces the concentration by this factor
    conc_start = np.zeros(n,n)+1.0  # Initial concentration [arbitrary units]. Add randomn values to this
    beta = 0.1       #Efficiency of the wind to disperse spores
    
    # Empty arrays
    conc_spore = np.zeros(n,n,n_dates)
    conc_spore_temp = np.zeros(n,n)  #This is a temporary array used to hold intermediary values (avoid double counting spores)
    PMI_field = np.zeros(n,n,n_dates) #PMI value for all fields and days
    w_field = np.zeros(n,n,n_dates) #wind value for all fields and days. 
    th_field = np.zeros(n,n,n_dates) #direction value field. Coordinate system is in NOASS system. (e.g. S wind is 180)
    conc_int = np.zeros(n_dates)
    conc_measure = np.zeros(n_dates)

    # Add the PMI to each time-step in the PMI_field array. Right now assume no variation in the PMI 
    # between cells but this can be corrected later.
    PMI_field[:,:,]+=PMI_arr 

    ### Open text file to write results ###
    tab = open(sample_name+'_model_LAMP.txt','w')
    tab.write('# Modeled Concentration measurements from '+sample_name+' data\n')
    tab.write('# Start Date | End Date | Collected Spores | PMI | Treatments\n')
 
    # Loop over all days
    for time_st in enumerate(dates):
        
        # Initialize string to be written to table
        tab_str = ''
        
        # Loop over all the fields in the array and calculate the concentration at the end of each day
        for ii in range(shape(conc_spore)[0])
            for jj in range(shape(conc_spore)[1])
                
                # Calculate spore growth rate from PMI for the cell [ii,jj]
                t_pmi = f_t_pmi(PMI_field[ii,jj,time_st])

                # Check if a treatment was applied
                if np.sum(LAMP_dates[:,1]==dates[ii]) >= 1:
                    ind_treat = np.where(LAMP_dates[:,1]==dates[ii])[0][0]
                    treat_str = LAMP_treat[ind_treat]+'\n'
                    if LAMP_treat[ind_treat] != 'None':
                        spray_factor = (1.-treat_eff)                  
                    else:
                        spray_factor = 1.
                        treat_str = 'None\n'

                # Calculate Daily Spore concentration and integrated spore concentration
                # Handle day 0 separately 
                if time_st == 0:
                    # Calculate spore concentration 
                    conc_spore_temp[ii,jj] = grow_rate**(1.0/t_pmi)*conc_start[ii,jj]*spray_factor
                    # Calculate Integrated spore concentration
                    #conc_int[ii] = t_pmi/math.log(grow_rate)*(conc_spore[ii]-conc_start)
                else:
                    # Calculate spore concentration 
                    conc_spore_temp[ii,jj] = grow_rate**(1.0/t_pmi)*conc_spore[ii,jj,time_st-1]*spray_factor
                    
                    #Add the contributions from each side of the cell
                    #Side 1
                    if (th_field[ii,jj-1,time_st] >= 0 and th_field[ii,jj-1,time_st] < 90) or \
                       (th_field[ii,jj-1,time_st] > 270 and th_field[ii,jj-1,time_st] <= 360):
                          conc_spore_temp[ii,jj] += w[ii,jj-1,time_st]*beta*abs(cos(th_field[ii,jj-1,time_st]))*conc_spore[ii,jj-1,time_st-1]
                          
                    #Side 2      
                    if (th_field[ii+1,jj,time_st] > 0 and th_field[ii+1,jj,time_st] < 180):
                          conc_spore_temp[ii,jj] += w[ii+1,jj,time_st]*beta*abs(sin(th_field[ii+1,jj,time_st]))*conc_spore[ii+1,jj,time_st-1]
                          
                    #Side 3
                    if (th_field[ii,jj+1,time_st] > 90 and th_field[ii,jj+1,time_st] < 270):
                          conc_spore_temp[ii,jj] += w[ii,jj+1,time_st]*beta*abs(cos(th_field[ii,jj+1,time_st]))*conc_spore[ii,jj+1,time_st-1]
                    
                    #Side 4
                    if (th_field[ii-1,jj,time_st] >180 and th_field[ii-1,jj,time_st] < 360):                  
                          conc_spore_temp[ii,jj] += w[ii-1,jj,time_st]*beta*abs(sin(th_field[ii-1,jj,time_st]))*conc_spore[ii-1,jj,time_st-1]

                    # Subtract the spore numbers leaving the cell. 
                    conc_spore_temp[ii,jj] -= w[ii,jj,time_st]*beta*conc_spore[ii,jj,time_st-1]
                       
                    # Calculate (daily) Integrated Spore concentration (how many spores collected by a trap today?)
                    #conc_int[ii]=t_pmi/math.log(grow_rate)*(conc_spore[ii]-conc_spore[ii-1])+(t_pmi/math.log(grow_rate))**2*(1.-spray_factor)*conc_spore[ii-1]*(grow_rate**(1.0/t_pmi)-1.)

                    # Calculate number of spores captured by trap since COLLECTION_TIME days ago.
                    if ii < collection_time:
                        # Add up all the days up til now
                        #conc_measure[ii] = np.sum(conc_int[0:ii+1])
                        # Set start date 
                        #tab_str+='{:10s}  {:10s}'.format(dates[0],dates[ii])
                    else:
                        #conc_measure[ii] = np.sum(conc_int[ii-collection_time:ii+1])
                        #tab_str+='{:10s}  {:10s}'.format(dates[ii-collection_time],dates[ii])
        
        #Save the computed field array into the temporal structure and reset the temporary array
        conc_spore[:,:,time_st]=conc_spore_temp
        conc_spore_temp*=0.0

        ### Fill out rest of table entry
        tab_str+='  {:8.3f}  {:3.0f}  '.format(conc_measure[ii],PMI)
        tab_str+=treat_str
        tab.write(tab_str)
                
        # Close Table
        tab.close()

    # Return outputs
    return conc_measure

#####################
# Function to calculate spore growth rate as function of PMI
def f_t_pmi(pmi):

    if pmi >= 0 and pmi <= 30: t_pmi = 15.
    elif pmi >= 40 and pmi <=50: t_pmi = 10.
    elif pmi >= 60 and pmi <= 100: t_pmi = 5.

    return t_pmi

#####################
###### Run Code #####
#####################
if __name__ == '__main__':

    # Set sample names
    all_samples = ['frei','two_rock','laguna']

    # Loop over all samples and make parameter tables
    colors = ['r','b','k']
    for ii,sample_name in enumerate(all_samples):
        t, w, tr = make_table(sample_name,time_bin=0.25)
        c_meas = model_conc(sample_name,collection_time=2)
        
        # Plot model results
        pl.plot(c_meas,colors[ii], label=sample_name)

    # Legend
    pl.legend()
    pl.show()
