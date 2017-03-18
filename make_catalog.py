#! /usr/bin/env python
# 
# Program: MAKE_CATALOG.py
#
# Author: Nick Lee
#
# Usage: ./make_catalog.py
#
# Description: Routines to create a table of features & answers that can be fed into machine learning algorithms to predict powdery mildew growth.
#              Also contains functions that can model mildew concentration using a simple analytical model. 
#
# Revision History:
#    Date        Vers.    Author      Description
#    12/24/14    1.0a0    Nick Lee    First checked in
#    1/2/15      1.1a0    Nick Lee    Modelling of single farm complete
#    2/6/15      1.2a0    Nick Lee    Modelling of multiple farms with wind data complete
#
# To Do:
#    Determine which features should be used
#    Model multi-farm systems that include wind

# Import Libraries
import numpy as np
import pylab as pl
import scipy as sp
import math
import pdb
import pandas as pd
import os
import timeit

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
## Calculate distance between two points.
## LOC1 and LOC2 and both tuples containing (x,y) coords
def dist(loc1,loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)


#####################
## Change time format of '00:00:00 AM' to miltary time ('00:00:00')
def fix_clock(time_str):

    noon_flag = np.str.split(time_str,' ')[1]
    hh, mm, ss = np.str.split(np.str.split(time_str,' ')[0],':')
    
    if hh == '12' and noon_flag == 'AM':
        hh = '00'
    if noon_flag == 'PM' and hh != '12':
        hh = np.str(np.int(hh) + 12)

    return hh+':'+mm+':'+ss
        

#####################
# Read in weather data from various catalogs.
def get_weather(sample_name,year=2014):

    # Read in data
    filename = 'catalogs/'+sample_name+'.txt'
    if not os.path.isfile(filename): filename = '/Users/nlee/Computing/Smart-Dart/powdery_mildew/catalogs/'+sample_name+'.txt'

    # Original Dataset
    if year!= 2015:
        # Array of dates & times
        time = np.genfromtxt(filename,dtype=np.str,usecols=[0,1],skiprows=1)
        # Array of we ather information
        # Temperature, Precipitation, Wetness, Humidity
        weather = np.genfromtxt(filename,usecols=[2,3,4,5],skiprows=1)

    # 2015 Data
    else:

        ## Find the closest location to figure out which weather data to extract
        # Weather locations 
        weather_loc = np.array([[-122.961129,38.743031], # a21
                                [-122.975748,38.754881], # a09
                                [-122.979780,38.757900]]) # a06
        weather_name = ['A21','A09','A06']
        # Farm locations                       
        farm_name = np.array(['A01','A08','A11','A17','A20','A40'])
        farm_loc = np.array([[-122.989601,38.758091], # a01 
                            [-122.977784,38.757915], # a08
                            [-122.975895,38.752221], # a11
                            [-122.969639,38.744886], # a17
                            [-122.960904,38.745568], # a20
                            [-122.971815,38.752969]]) # a40
        # Find location of farm
        farm_pos = farm_loc[np.argwhere(farm_name == sample_name)[0][0]]

        # Find which weather station is closest
        indmin = np.argmin([dist(farm_pos,weather_loc[i]) for i in range(3)])
        print indmin
        
        # Read in weather data from correct file
        filename = 'catalogs/2015_Results/Weather/'+weather_name[indmin]+'_RawWeather.csv'
        time = np.genfromtxt(filename,usecols=[0,1],skiprows=1,delimiter=';',dtype=np.str)
        if weather_name[indmin] == 'A21':
            temp = np.genfromtxt(filename,usecols=[4],skiprows=1,delimiter=';')
            rh = np.genfromtxt(filename,usecols=[2],skiprows=1,delimiter=';')
            precip = np.genfromtxt(filename,usecols=[5],skiprows=1,delimiter=';')
            wet = np.genfromtxt(filename,usecols=[3],skiprows=1,delimiter=';')
            print 'using a21'
        elif weather_name[indmin] == 'A09':
            temp = np.genfromtxt(filename,usecols=[2],skiprows=1,delimiter=';')
            precip = np.genfromtxt(filename,usecols=[3],skiprows=1,delimiter=';')
            rh = np.genfromtxt(filename,usecols=[4],skiprows=1,delimiter=';')
            wet = np.genfromtxt(filename,usecols=[5],skiprows=1,delimiter=';')
            print 'using a09'
        else:
            temp = np.genfromtxt(filename,usecols=[2],skiprows=1,delimiter=';')
            rh = np.genfromtxt(filename,usecols=[3],skiprows=1,delimiter=';')
            # Make column of 0's for missing weather data (precipitation, wetness)
            precip = np.zeros(len(temp))
            wet = np.zeros(len(temp))
            print 'using a06'
        weather = np.column_stack((temp,precip,wet,rh))

    # Outputs
    return time, weather 

#####################
# Read in LAMP data from various catalogs
# Set YEAR = 2015 to use data from 2015
def get_LAMP(sample_name,year=2014):

    # Original data from Frei, 2 rock, Laguna
    if year != 2015:
        # Set filenames, checking for the table in the same directory first.
        filename = 'catalogs/'+sample_name+'_LAMP.txt'
        if not os.path.isfile(filename): filename = '/Users/nlee/Computing/Smart-Dart/powdery_mildew/catalogs/'+sample_name+'_LAMP.txt'
        
        # Array of dates (start and end)
        LAMP_dates = np.genfromtxt(filename,usecols=[0,1],dtype=np.str)

        # Array of LAMP Results
        LAMP_results = np.genfromtxt(filename,usecols=[2])

        # Array of treatments
        LAMP_treatments = np.genfromtxt(filename,usecols=[4],dtype=np.str)

    # 2015 Data
    else:
        # Set filename
        filename = 'catalogs/2015_spore_results.csv'
        
        # Get dates
        end_dates = np.genfromtxt(filename,usecols=[0],dtype=np.str,delimiter=',')
        start_dates = np.insert(end_dates[:-1],0,'4/10/15')
        LAMP_dates = np.column_stack((start_dates,end_dates))

        # Figure out which column to get LAMP data from
        all_2015_names = np.array(['A01','A08','A11','A17','A20','A40'])
        try:
            col_ind = np.argwhere(all_2015_names == sample_name)[0][0]+2
        except:
            print 'Invalid sample name provided to get_LAMP'
        LAMP_reading = np.genfromtxt(filename,usecols=[col_ind],delimiter=',',dtype=np.str)
        LAMP_results = np.where(LAMP_reading=='+',np.zeros(len(LAMP_reading))+1,np.zeros(len(LAMP_reading)))

        LAMP_treatments = np.repeat(np.array(['None']),len(LAMP_reading))
        
        
    return LAMP_dates, LAMP_results, LAMP_treatments

#####################
# Read in Wind data
# Set keyword UV = False to use old Napa data. Else, use matched Temperature & wind data from Erik
def get_wind(uv=True,month='Oct'):
    
    if uv!=True:
        # Set filename:
        filename = 'catalogs/napa_wind.txt'
    
        # Read in data
        hours = np.genfromtxt(filename,usecols=[0])
        all_wind_pos = np.genfromtxt(filename,usecols=[1,2,3,4])
        wind_spd = np.genfromtxt(filename,usecols=[5,6])

        # Find number of locations in grid by seeing how many unique entries there are for the first hour
        n_grid = len(np.where(hours == hours[0])[0])

        # Make new array that contains one row for each position, with different wind direction & speeds in different columns.
        n_hrs = len(ordered_uniq(hours))
        wind_dat = np.zeros([n_grid,2+(2*n_hrs)])

        # Fill in wind array
        wind_dat[:,0:2] = all_wind_pos[0:n_grid,0:2]
        for ii, hr in enumerate(ordered_uniq(hours)): wind_dat[:,2+(2*ii):4+(2*ii)] = wind_spd[n_grid*ii:n_grid*(ii+1),:]

        # Return wind array
        return wind_dat

    else:

        # Set filename:
        if month=='Oct': filename = 'catalogs/October2014.csv'
        else: filename = 'catalogs/August2014.csv'

        # Get date and time of each measurement
        dates_n_times = np.genfromtxt(filename,delimiter=',',dtype=np.str,usecols=[0])
        uniq_times = np.array(ordered_uniq(dates_n_times))

        # Number of unique positions and times
        n_times = uniq_times.shape[0]
        n_pos = dates_n_times.shape[0]/n_times

        # Get corresponding location, wind and temperature data
        all_pos = np.genfromtxt(filename,delimiter=',',usecols=[1,2])
        wind_n_temp = np.genfromtxt(filename,delimiter=',',usecols=[5,6,7])
        
        # Create array that contains wind and temperature data in each position and time
        wind_dat = np.empty((n_times,n_pos,3))
        times = np.empty((n_times,2),dtype='S10')
        for ii, date_str in enumerate(uniq_times):
            times[ii] = [date_str[4:6]+'/'+date_str[6:8]+'/'+date_str[0:4],date_str[8:10]+':'+date_str[10:]]
            wind_dat[ii,:,:] = wind_n_temp[ii*n_pos:(ii+1)*n_pos]

        # Array of unique positions
        pos = all_pos[0:n_pos]
        
        return times, pos, wind_dat
        
        

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
Keyword Arguments:
   WIND - [N x 2] Array that contains U and V wind strength at each day. 
'''
def gen_PMI(time,temperature,time_bin=0.25,wind='none'):
    
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
    if wind != 'none': avg_wind = np.empty((n_days,2))
    
    # Loop over dates
    for ii, date in enumerate(dates):
        # Find times that correspond to this date.
        ind_day = np.where(time[:,0]==date)[0]

        # Calculate average wind
        if wind != 'none': avg_wind[ii] = np.mean(wind[ind_day,:],axis=0)

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

    if wind == 'none': return np.array(dates), PMI
    else: return np.array(dates), PMI, avg_wind

#####################
'''
Function: MAKE_TABLE
Generate a table of features and answers that can be fed to machine learning algorithm
Inputs:
   SAMPLE_NAME - Name of desired sample. Options are 'two_rock','laguna', or 'frei'.
   TIME_BIN - Amount of time (in hours) corresponding to each time interval (Default is 0.25 hrs, or 15 min)

'''
def make_table(sample_name,time_bin=0.25,year=2014):

    # Get weather data and LAMP data
    time, weather = get_weather(sample_name,year=year)
    LAMP_dates,LAMP_results,LAMP_treat = get_LAMP(sample_name,year=year)
    n_LAMP = np.size(LAMP_results)

    #######################################
    ### Determine Features to include ###
    #######################################
    # Set temperature ranges for binning
    temp_bins = [0,60,65,70,75,80,85,90,95,1000]
    n_temp = np.size(temp_bins)-1
    temp = weather[:,0]

    # Set which Weather factors to use
    weather_str = ['Precipitation', 'Leaf Wetness', 'Humidity']
    if year == 2015: weather_flag = [0,0,1]
    else: weather_flag = [0,1,1]
    n_weather = len(weather_str)

    # Set which Treatments to include
    # 'ALL' refers to treating all treatments the same
    all_treatments = ['Any Treatment','Oil','Mettle','Flint','Kocide','Sulfur','Sylcoat','Switch','Sovran']
    if year == 2015: treatment_flag = [0,0,0,0,0,0,0,0,0]
    else: treatment_flag = [1,0,0,0,0,0,0,0,0]
    n_treat = len(all_treatments)

    #####################################
    ### Open Text file to write Table ###
    #####################################
    if year==2015: tab = open('/Users/nlee/Computing/Smart-Dart/powdery_mildew/catalogs/'+sample_name+'_feat2015.txt','w')
    else: tab = open('/Users/nlee/Computing/Smart-Dart/powdery_mildew/catalogs/'+sample_name+'_feat.txt','w')
    tab.write('# Features and Answers for '+sample_name+'\n')

    # Create header
    header_str = '# '
    h_count = 1
    for hh,temp_min in enumerate(temp_bins[0:-1]):
        header_str += str(temp_min) + '<T<' + str(temp_bins[hh+1]) + ' | '
        tab.write('# Column '+str(h_count)+': Time (hours) in Temperature Range '+str(temp_min) + '<T<' + str(temp_bins[hh+1]) +'\n')
        h_count+=1
    for hh,ww in enumerate(weather_str):
        if weather_flag[hh] == 1:
            header_str += ww + ' | '
            tab.write('# Column '+str(h_count)+': Average ' + ww +'\n')
            h_count+=1
    for hh,tt in enumerate(all_treatments):
        if treatment_flag[hh] == 1:
            header_str += tt + ' | '
            tab.write('# Column '+str(h_count)+': Treated with ' + tt +' (1 = Treatment used)\n')
            h_count+=1
    # LAMP results from last week and this week
    tab.write('# Column '+str(h_count)+': Previous LAMP Result (1 = Detection)\n')
    tab.write('# Column '+str(h_count+1)+': LAMP Result (1 = Detection)\n')
    
    # Write Last header
    tab.write(header_str + 'Previous LAMP | LAMP Result\n')
        
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
            # Handle ANY treatments separately:
            if jj == 0:
                if LAMP_treat[ii].find('None') == -1: treat_arr[ii,jj] = 1 # If "None" is not found, then we know something was applied.
            elif LAMP_treat[ii].find(treat) > -1: treat_arr[ii,jj] = 1
        
        ### Write all results to Table ###
        for jj, num in enumerate(temp_arr[ii,:]): tab_str+='{:7.2f}  '.format(num)
        for jj,num in enumerate(avg_weather):
            if weather_flag[jj] == 1: tab_str+='{:7.3f}  '.format(num)
        for jj,num in enumerate(treat_arr[ii,:]):
            if treatment_flag[jj] == 1: tab_str+='{:1}  '.format(num)
        # Previous week's LAMP results (use 0 if there was no previous week)
        if ii == 0: tab_str+='0.0  '
        else: tab_str+=str(LAMP_results[ii-1])+'  '
        # Add this week's LAMP result, and make a new line
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
    grow_rate = 1.5  # Growth factor for the pathogen
    treat_eff =  0.0 #0.5  # Efficiency of spray treatment. Each spray reduces the concentration by this factor
    conc_start = 1.0    # Initial concentration [arbitrary units]

    # Empty arrays
    conc_spore = np.zeros(n_dates)
    conc_int = np.zeros(n_dates)
    conc_measure = np.zeros(n_dates)

    ### Open text file to write results ###
    tab = open('/Users/nlee/Computing/Smart-Dart/powdery_mildew/catalogs/'+sample_name+'_model_LAMP.txt','w')
    tab.write('# Modeled Concentration measurements from Two Rock data\n')
    tab.write('# Start Date | End Date | Collected Spores | PMI | Treatments\n')
 
    # Loop over all days
    for ii,PMI in enumerate(PMI_arr):

        # Initialize string to be written to table
        tab_str = ''
        
        # Calculate spore growth rate from PMI
        t_pmi = f_t_pmi(PMI)

        # Check if a treatment was applied
        ind_treat = np.where(LAMP_dates[:,1]==dates[ii])[0]  # Returns an array
        if LAMP_treat[ind_treat] != ['None']:
            spray_factor = 1.-treat_eff
            treat_str = LAMP_treat[ind_treat[0]]+'\n'
        else:
            spray_factor = 1
            treat_str = 'None\n'

        # Calculate Daily Spore concentration and integrated spore concentration
        # Handle day 0 separately 
        if ii == 0:
            # Calculate spore concentration 
            conc_spore[ii]=grow_rate**(1.0/t_pmi)*conc_start*spray_factor
            # Calculate Integrated spore concentration
            conc_int[ii] = t_pmi/math.log(grow_rate)*(conc_spore[ii]-conc_start)#*spray_factor)
        else:
            # Calculate spore concentration 
            conc_spore[ii]=grow_rate**(1.0/t_pmi)*conc_spore[ii-1]*spray_factor
            # Calculate (daily) Integrated Spore concentration (how many spores collected by a trap today?)
            conc_int[ii]=t_pmi/math.log(grow_rate)*(conc_spore[ii]-conc_spore[ii-1])#*spray_factor)
            conc_int[ii]=t_pmi/math.log(grow_rate)*(conc_spore[ii]-conc_spore[ii-1])+(t_pmi/math.log(grow_rate))**2*(1.-spray_factor)*conc_spore[ii-1]*(grow_rate**(1.0/t_pmi)-1.)

        # Calculate number of spores captured by trap since COLLECTION_TIME days ago.
        if ii < collection_time:
            # Add up all the days up til now
            conc_measure[ii] = np.sum(conc_int[0:ii+1])
            # Set start date & end date for table
            tab_str+='{:10s}  {:10s}'.format(dates[0],dates[ii])
        else:
            # Add up last two days
            conc_measure[ii] = np.sum(conc_int[ii-collection_time:ii+1])
            tab_str+='{:10s}  {:10s}'.format(dates[ii-collection_time],dates[ii])

        ### Fill out rest of table entry
        tab_str+='  {:8.3f}  {:3.0f}  '.format(conc_measure[ii],PMI)
        tab_str+=treat_str
        tab.write(tab_str)

    # Close Table
    tab.close()

    # Return outputs
    return conc_spore, conc_measure

#####################
# Function to calculate spore growth rate as function of PMI
def f_t_pmi(pmi):
    
    if pmi >= 0 and pmi <= 30: t_pmi = 15.
    elif pmi >= 40 and pmi <=50: t_pmi = 10.
    elif pmi >= 60 and pmi <= 100: t_pmi = 5.
        
    return t_pmi



#####################
'''
Function: MODEL_CONC_WIND
Model the concentration of powdery mildew in the air for a grid of locations, taking into account dispersal through wind.
Model produces instantaneous concentration (as opposed to integrated results from MODEL_CONC).

Inputs:
   SAMPLE_NAME - Name of desired sample to use as weather data. Options are 'two_rock','laguna', or 'frei'.
   TIME_BIN - Amount of time (in hours) corresponding to each time interval (Default is 0.25 hrs, or 15 min)
   MONTH - Set to Month of napa data to use
Outputs:
   CONCENTRATION - A daily measure of spore concentration in arbitrary units, given in vector of length N_DAYS
'''
def model_conc_wind(sample_name,time_bin=1,test=False,month='Oct'):

    # Get wind and weather data
    time, grid_pos, wind_dat = get_wind(month=month)

    # Figure out lengths of all arrays
    n_time, n_grid, n_winddat = np.shape(wind_dat)
    n_dates = len(ordered_uniq(time[:,0]))

    # Make array of PMI and wind for each position and date
    PMI_arr = np.empty((n_dates,n_grid))
    wind = np.empty((n_dates,n_grid,2))
    for ii, pos in enumerate(grid_pos):
        dates, PMI, avg_wind = gen_PMI(time,wind_dat[:,ii,2],time_bin=time_bin,wind=wind_dat[:,ii,0:2])
        PMI_arr[:,ii] = PMI
        wind[:,ii] = avg_wind

    # test wind
    if test==True:
        PMI_arr = np.zeros((n_dates,n_grid))+60
        wind = np.zeros((n_dates,n_grid,2))
        wind[:,:,0] = 1

    ### Temperature bins for Machine Learning ###
    temp_bins = [0,60,65,70,75,80,85,90,95,1000]
    n_temp = np.size(temp_bins)-1

    # Find indices of neighbors of each grid position
    neighbors = np.zeros([n_grid,4])
    for ii, pos in enumerate(grid_pos):
        neighbors[ii,:] = find_neighbors(pos,grid_pos)
    
    # Set modeling constants
    grow_rate = 1.5  # Growth factor for the pathogen
    treat_eff = 0.0  # Efficiency of spray treatment. Each spray reduces the concentration by this factor
    wind_coeff = 0.05  # Efficiency of wind at dispersing spores

    # Empty arrays
    conc_spore = np.zeros([n_dates,n_grid])
    temp_arr = np.zeros([n_dates,n_grid,n_temp])
    
    # Make initial concentrations (day 0)
    conc_start = np.zeros(n_grid)+1.0    # Initial concentration [arbitrary units]

    ### Open text file to write results ###
    if test==True:
        tab = open('catalogs/test_model_wind_'+month+'.txt','w')
        feat = open('catalogs/test_feat_wind_'+month+'.txt','w')
    else:
        tab = open('catalogs/'+sample_name+'_model_wind_'+month+'.txt','w')
        feat = open('catalogs/'+sample_name+'_feat_wind_'+month+'.txt','w')

    # Write results of wind model
    tab.write('# Modeled Concentration measurements from '+sample_name+' data, modeling wind using Napa wind \n')
    tab.write('# Day Number | X | Y | Wind Speed U-Component | Wind Speed V-Component | Concentration\n')

    ### Text file to write features to feed into machine learning algorithm ###
    feat.write('# Modeled features and answers for '+sample_name+'in month of '+month+'\n')
    # Create header
    header_str = '# Previous Concentration | '
    h_count = 2
    for hh,temp_min in enumerate(temp_bins[0:-1]):
        header_str += str(temp_min) + '<T<' + str(temp_bins[hh+1]) + ' | '
        h_count+=1
    header_str += 'Wind Pressure In | Wind Pressure Out | Answer (Concentration)\n'
    feat.write(header_str)
        
    # Loop over all dates
    for ii, date in enumerate(dates):

        # Loop over all grid positions
        for jj, pos in enumerate(grid_pos):

            # Initialize string to be written to table
            x = pos[0]
            y = pos[1]
            tab_str = '{:3.0f}  {:2.0f}  {:3.0f}  '.format(ii,x,y)
            
            # Calculate Growth rate from PMI
            t_pmi = f_t_pmi(PMI_arr[ii,jj])

            # Figure out yesterday's concentration for full grid
            if ii == 0: conc_old = conc_start
            else: conc_old = conc_spore[ii-1,:]

            # Calculate Daily Spore concentration and integrated spore concentration from temperature
            conc_spore[ii,jj]=grow_rate**(1.0/t_pmi)*conc_old[jj] # *spray_factor
            #conc_spore[ii,jj] = conc_old[jj]

            ## Add contribution from wind
            # Loss of spores due to wind in this cell (only depends on wind speed)
            # wind_loss = -1. * wind_coeff * conc_old[jj] * wind[ii,jj,1]  ## For Wind given in speed + direction
            wind_loss = -1. * wind_coeff * conc_old[jj] * (wind[ii,jj,0]**2 + wind[ii,jj,1]**2)**0.5  # For wind given in U and V

            ## Addition of spores from neighboring cells
            # Get the indices of the neighbor indices
            wind_gain = wind_coeff * wind_influx(wind[ii,:,0],wind[ii,:,1],conc_old,neighbors[jj,:],wind_fmt='uv')

            ## Adjust spore concentration by flux in and out due to wind
            conc_spore[ii,jj] += wind_gain + wind_loss

            ## Add to table input
            tab_str+='{:7.3f}  {:7.3f}  {:8.3f}\n'.format(wind[ii,jj,0],wind[ii,jj,1],conc_spore[ii,jj]) 
            tab.write(tab_str)
            
            #### Find features for machine learning
            # Initialize string to be written to table with yesterday's concentration
            feat_str = '{:8.3f}  '.format(conc_old[jj])

            # Find indices of time bins corresponding to date range.
            ind_day = np.where(time[:,0] == dates[ii])[0]
            # Find amount of time (in hours) in each temperature bin
            hist, bins = np.histogram(wind_dat[ind_day,jj,2],bins=temp_bins)
            ## Save temperature data to array and add to string for table
            temp_arr[ii,jj,:] = hist*time_bin
            for kk, num in enumerate(temp_arr[ii,jj,:]): feat_str+='{:7.2f}  '.format(num)

            ## Add incoming wind and outgoing wind
            feat_str+='{:8.3f}  {:8.3f}  '.format(wind_gain/wind_coeff,-1.*wind_loss/wind_coeff)
            # Add current concentration (answer)
            feat_str+='{:8.3f}\n'.format(conc_spore[ii,jj])

            ## Write to table
            feat.write(feat_str)

    # Close Tables
    tab.close()
    feat.close()
    
    # Return Result
    return grid_pos, wind, conc_spore
            

#####################
'''
Function: FIND_NEIGHBORS
Function that returns the indices of the 4 neighboring cells to the given cell.

Inputs:
   POS - 2-element vector that contains x and y position of target cell
   ALL_POS - [n x 2] array that lists all the x and y positions of every cell
Outputs:
   IND - 4-element vector that contains the indices of the adjacent cells, in clockwise order: (North, East, South, West).
         If one of the directional neighbors does not exist, returns a -99. For example, if Target cell is already on northern edge,
         the output will look like: [-99,e_ind,s_ind,w_ind]
'''
def find_neighbors(pos,all_pos):

    # Find North Edge:
    north_ind = np.where(np.all([all_pos[:,0]==pos[0],all_pos[:,1]==pos[1]+1],axis=0))[0]
    if len(north_ind) == 0:
        north = -99
    else:
        north = north_ind[0]

    # Find East Edge:
    east_ind = np.where(np.all([all_pos[:,0]==pos[0]+1,all_pos[:,1]==pos[1]],axis=0))[0]
    if len(east_ind) == 0:
        east = -99
    else:
        east = east_ind[0]

    # Find South Edge:
    south_ind = np.where(np.all([all_pos[:,0]==pos[0],all_pos[:,1]==pos[1]-1],axis=0))[0]
    if len(south_ind) == 0:
        south = -99
    else:
        south = south_ind[0]

    # Find West Edge:
    west_ind = np.where(np.all([all_pos[:,0]==pos[0]-1,all_pos[:,1]==pos[1]],axis=0))[0]
    if len(west_ind) == 0:
        west = -99
    else:
        west = west_ind[0]
        
    return np.array([north,east,south,west])

#####################
'''
Function: WIND_INFLUX
Function that calculates the total influx of spores due to wind from neighbors. 

Inputs:
   WIND_DIR - Full array of wind directions for all cells in a grid at a given time.
              If WIND_FMT is set to 'UV', this is instead all U-direction wind speeds.
   WIND_SPEED - Full array of wind speeds for all cells in a grid at a given time.
                If WIND_FMT is set to 'UV', this is instead all V-direction wind speeds.
   CONC - Full array  of spore concentration for all cells in a grid at a given time.
   INDS - 4-element array with the indices representing the North, East, South, and West neighbors (in that order)
Outputs:
   WIND_FLUX - The total flux of wind coming into the center cell. 
'''
def wind_influx(wind_dir,wind_speed,conc,inds,wind_fmt='uv'):
    
    ## Initialize fluxes
    wind_flux = 0.

    ## Loop over all cardinal directions
    for ii, ind in enumerate(inds):
        # Make sure index isn't -99
        if ind != -99:
            # North or south directions
            if ii % 2 == 0:
                # If wind is given in speed/direction
                if wind_fmt != 'uv': flux = -1.*wind_speed[ind] * math.cos(math.radians(wind_dir[ind])) * conc[ind]  # negative because 0 degrees is south
                else: flux = wind_speed[ii]  # V-direction
            else:
                if wind_fmt != 'uv': flux = -1.*wind_speed[ind] * math.sin(math.radians(wind_dir[ind])) * conc[ind] # negative because 0 degrees is south
                else: flux = wind_dir[ii] # U-direction
            # Add to wind_flux only if correct sign
            if (ii <= 1 and flux < 0) or (ii >=2 and flux > 0):
                wind_flux += abs(flux)

    return wind_flux

#####################
###### Run Code #####
#####################
if __name__ == '__main__':

    farm_name = np.array(['A01','A08','A11','A17','A20','A40'])
    for name in farm_name:
        make_table(name,year=2015)

    ## Get wind data
    #time, grid_pos, wind_dat = get_wind()

    #wind_dat = get_wind()
    #grid_pos, wind_model, c_spore_grid = model_conc_wind('napa',month='Oct')
    #grid_pos, wind_model, c_spore_grid = model_conc_wind('napa',month='Aug')


    #dates, PMI_arr = gen_PMI(time,temp,time_bin=1)


    #temp = model_conc_wind('frei')
   







    ## Test models
    #c_spore, c_meas = model_conc('frei')
    #grid_pos, wind_model, c_spore_grid = model_conc_wind('frei')

    #ind = 7

    #pl.plot(wind_dat[:,0],wind_dat[:,1],'k+')
    #pl.plot(wind_dat[ind,0],wind_dat[ind,1],'r+')

    #pl.plot(c_spore,c_spore_grid[:,ind])
    #pl.show()






    ###############
    #### TESTS ####
    ###############
    ## Test wind influx
    # n_grid = np.shape(wind_dat)[0]
    # grid_pos = wind_dat[:,0:2]

    # neighbors = np.zeros([n_grid,4])
    # for ii, pos in enumerate(grid_pos):
    #    neighbors[ii,:] = find_neighbors(pos,grid_pos)
    
    # wind_uniform = np.zeros([n_grid,2])
    # wind_uniform[:,0] = 45.  # direction in degrees - wind blowing from south to north
    # wind_uniform[:,1] = 1.    # speed
    # wind = wind_uniform
    # conc_spore = np.zeros(n_grid)+1.

    # Add some unique numbers to neighbors
    # ind = 7
    # wind_uniform[neighbors[ind,:].astype(int),1] = [2,2,6,8]
    
    # wind_gain = wind_influx(wind[:,0],wind[:,1],conc_spore,neighbors[ind,:])
    # print wind_gain
   
    # Plots
    # pl.plot(wind_dat[:,0],wind_dat[:,1],'k+')
    # pl.plot(wind_dat[ind,0],wind_dat[ind,1],'r+')
    # pl.show()



    ## Set sample names
    #all_samples = ['frei','two_rock','laguna']

    ## Loop over all samples and make table of features
    #colors = ['r','b','k']
    #for ii,sample_name in enumerate(all_samples):
    #    t, w, tr = make_table(sample_name,time_bin=0.25)
    #    c_meas = model_conc(sample_name,collection_time=2)
        
    # Plot model results
    #    pl.plot(c_meas,colors[ii], label=sample_name)

    # Legend
    #pl.legend()
    #pl.show()

    ## Time functions
    #s1 = "a = read_table('frei')"
    #print(timeit.timeit(stmt=s1,setup="from __main__ import read_table",number=10))

    #s2 = "b = make_wind_table('frei')"
    #print(timeit.timeit(stmt=s2,setup="from __main__ import make_wind_table",number=10))
