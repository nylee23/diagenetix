#!/usr/bin/env python
'''
Display the wind field and the concentration variation. The information has be read from a file containing
the columns Day X Y U V Concentration 

I assume U is positive to the right and V is positive up so that (U,V) correspond to cartesian system (x,y) system.
Right now it is also possible to read in (theta,W) for the wind direction where theta is 0 for straight North wind
and increases clockwise. This is the original system we worked in. 

'''

from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np
import pylab as pl

def ordered_uniq(seq, idfun=None): 
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked

# Read the data from the appropriate file
def get_weather():
    # Choose the file containing the weather and real concentration data.
    # Remember if the file used (theta,W) coordinates for the wind need to uncomment two more lines below.
    filename='napa_model_wind_Aug.txt'
    #filename='frei_model_wind.txt'
    
    # Choose the file containing the machine learning concentration data. (Assume it has the same format as the file above.
    # Could change this in the future so that the same file contains both real and machine learning data.
    #filename_machine='napa_model_wind.txt'
    filename_machine='napa_model_Aug_full.txt'

    # Array of days
    time = np.genfromtxt(filename,dtype=np.int,usecols=[0],comments='#')

    # Arrays of coordinates, velocities and concentration for the real data 
    coor = np.genfromtxt(filename,dtype=np.int,usecols=[1,2],comments='#')
    vel_file = np.genfromtxt(filename,dtype=np.float,usecols=[3,4],comments='#')
    c = np.genfromtxt(filename,dtype=np.float,usecols=[5],comments='#')
    
    # Array containing the machine concentration data
    c_machine = np.genfromtxt(filename_machine,dtype=np.float,usecols=[5],comments='#')

    #If the wind data is in (U,V coordinates leave as is)
    vel=vel_file
    
    #If the wind data is in Direction/Speed coordinates uncomment to convert to U,V coordinates
    #vel[:,0]=-vel_file[:,1]*sin((pi/180.0)*vel_file[:,0])
    #vel[:,1]=-vel_file[:,1]*cos((pi/180.0)*vel_file[:,0])
  
    # Outputs
    return time,coor,vel,c,c_machine

# Create the 3D arrays containing the concentration and wind data
def gen_arrays(days,coor,vel,c,c_machine):
    #Order the days into unique values
    days=ordered_uniq(time[:])
    n_days=np.size(days)
    
    #Find the x,y maximum and minimum coordinates 
    max_x=coor[:,0].max()
    min_x=coor[:,0].min()
    max_y=coor[:,1].max()
    min_y=coor[:,1].min()
    
    #Define the plotting arrays. Two coordinate arrays, two velocity arrays and one concentration array
    coor_arr_x=np.zeros([n_days,max_x+1,max_y+1])
    coor_arr_y=np.zeros([n_days,max_x+1,max_y+1])
    vel_arr_x=np.zeros([n_days,max_x+1,max_y+1])
    vel_arr_y=np.zeros([n_days,max_x+1,max_y+1])
    c_arr=np.zeros([n_days,max_x+1,max_y+1])
    c_arr_machine=np.zeros([n_days,max_x+1,max_y+1])

    #Populate the arrays
    for ii,day in enumerate(days):
        #Select the values that correspond to each day 
        ind_dat = np.where(time[:]==day)[0]
        
        #Treat the X,Y positions in the file as X,Y positions in the arrays
        vel_arr_x[ii,coor[ind_dat,0],coor[ind_dat,1]]=vel[ind_dat,0]
        vel_arr_y[ii,coor[ind_dat,0],coor[ind_dat,1]]=vel[ind_dat,1]
        coor_arr_x[ii,coor[ind_dat,0],coor[ind_dat,1]]=coor[ind_dat,0]
        coor_arr_y[ii,coor[ind_dat,0],coor[ind_dat,1]]=coor[ind_dat,1]
        c_arr[ii,coor[ind_dat,0],coor[ind_dat,1]]=c[ind_dat]
        c_arr_machine[ii,coor[ind_dat,0],coor[ind_dat,1]]=c_machine[ind_dat]
        
    #Outputs
    return coor_arr_x,coor_arr_y,vel_arr_x,vel_arr_y,c_arr,c_arr_machine,n_days,max_x,min_x,max_y,min_y


# Change the day of the simulation using the slider object and update the plot window
def update(val):
    
    #Select the day of the simulation (value passed from the slider object sday)
    d_val = sday.val
    
    # Redefine the arrays to plot 
    x=coor_arr_x[d_val,:,:]
    y=coor_arr_y[d_val,:,:]
    vx=vel_arr_x[d_val,:,:]
    vy=vel_arr_y[d_val,:,:]
    ctration=c_arr[d_val,:,:]
    ctration=ctration.T      #Transpose the concentration array so that it will plot correctly using pl.imshow()
    ctration_machine=c_arr_machine[d_val,:,:]
    ctration_machine=ctration_machine.T
    
    # Mask out the pixels with no or very low wind values
    v=sqrt(vx**2+vy**2)    # This is the absolute value of the wind
    low_v=find(v.flat[:]<0.001)     
    vx.flat[low_v]=nan
    vy.flat[low_v]=nan
    
    # Establish reasonable scaling for the quiver function based on real available data rather the full array
    # that contains empty pixels.
    high_v=find(v.flat[:]>0.001)
    scale=1.8*v.flat[high_v].mean()*max(10,math.sqrt(v.flat[high_v].size))
    l.scale=scale
    l_machine.scale=scale
    l_diff.scale=scale

    # Update the displays with the data appropriate for the two days
    # Display with real concentration
    im_l.set_data(ctration)
    l.set_UVC(vx,vy)
    
    # Display with machine concentration
    im_l_machine.set_data(ctration_machine)
    l_machine.set_UVC(vx,vy)

    # Display with the difference between the concentrations scaled to the real concentration
    im_l_diff.set_data((ctration-ctration_machine)/ctration)
    l_diff.set_UVC(vx,vy)
    
    draw()

#Obtain the time and data from the files
time,coor,vel,c,c_machine=get_weather()
coor_arr_x,coor_arr_y,vel_arr_x,vel_arr_y,c_arr,c_arr_machine,n_days,max_x,min_x,max_y,min_y=gen_arrays(time,coor,vel,c,c_machine)

# Initialize the plot window starting on Day 1
x=coor_arr_x[0,:,:]
y=coor_arr_y[0,:,:]
vx=vel_arr_x[0,:,:]
vy=vel_arr_y[0,:,:]
ctration=c_arr[0,:,:]
ctration=ctration.T
ctration_machine=c_arr_machine[0,:,:]
ctration_machine=ctration_machine.T
c_max=c_arr.max()
c_min=c_arr.min()

# Need to mask out the velocity field where there is no data from the main array
v=sqrt(vx**2+vy**2)
low_v=find(v.flat[:]<0.001)   
vx.flat[low_v]=nan
vy.flat[low_v]=nan

# Generate appropriate scaling for the pixels where there is data (base on the algorithm that quiver uses)
high_v=find(v.flat[:]>0.001)
scale=1.8*v.flat[high_v].mean()*max(10,math.sqrt(v.flat[high_v].size))

# Plot the initial image with real concentration
ax = subplot(131)
subplots_adjust(left=0.075, bottom=0.2,wspace=0.2,right=0.925,top=0.95)
im_l=pl.imshow(ctration,origin='lower',aspect=1.0,vmin=c_min,vmax=c_max)
pl.title('Real Concentration')
pl.colorbar(orientation='horizontal')
# Plot the initial velocity field
l=quiver(x, y, vx, vy, pivot='middle',scale=scale, headwidth=4, headlength=6)
# Zoom the plot the portion of the plot where there is relevant data
pl.xlim(min_x,max_x)
pl.ylim(min_y,max_y)

# Plot the initial image with machine concentration 
ax_machine = subplot(132)
#subplots_adjust(left=0.1, bottom=0.1)
im_l_machine=pl.imshow(ctration_machine,origin='lower',aspect=1.0,vmin=c_min,vmax=c_max)
pl.title('Machine Concentration')
pl.colorbar(orientation='horizontal')
l_machine=quiver(x, y, vx, vy, pivot='middle',scale=scale, headwidth=4, headlength=6)
pl.xlim(min_x,max_x)
pl.ylim(min_y,max_y)

# Plot the difference between the concentration and machine code scaled to the real concentration
ax_diff = subplot(133)
#subplots_adjust(left=0.1, bottom=0.1)
im_l_diff=pl.imshow((ctration-ctration_machine)/ctration,origin='lower',aspect=1.0,vmin=-0.15,vmax=0.15)
pl.title('Normalized Difference')
pl.colorbar(orientation='horizontal')
l_diff=quiver(x, y, vx, vy, pivot='middle',scale=scale, headwidth=4, headlength=6)
pl.xlim(min_x,max_x)
pl.ylim(min_y,max_y)

#Generate the slider for the day number
axcolor = 'lightgoldenrodyellow'
axday  = axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
sday = Slider(axday, 'Day', 0, n_days-1, valinit=0)

#Generate the event that receives input and updates the entire plot
sday.on_changed(update)

#Generate a reset button to go back to day 1
resetax = axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sday.reset()
button.on_clicked(reset)

# Generate a button to show the deviation plots for each day
resetax = axes([0.4, 0.025, 0.2, 0.04])
button_d = Button(resetax, 'Deviations', color=axcolor, hovercolor='0.975')

# Create the function that computes the deviation
def deviations(event):
    # Generate the new plot window
    plt.ion()
    fig2=plt.figure()
    ax2=fig2.add_subplot(111)
    
    # Generate the arrays holding the plotting variables for each day
    val_day=np.zeros(n_days)
    val_mean=np.zeros(n_days)
    val_stdev=np.zeros(n_days)
    
    # Calculate the mean and standard deviation for each day making sure the non-data days are removed
    for ii,dummy_day in enumerate (val_day):
        val_day[ii]=ii+1
        day_c_arr=c_arr[ii,:,:]
        day_c_arr_machine=c_arr_machine[ii,:,:]
        # Mask out the values where there is no data
        mask_c=find(day_c_arr.flat[:]>0.001)
        dummy_mean_arr=abs(day_c_arr.flat[mask_c]-day_c_arr_machine.flat[mask_c])/day_c_arr.flat[mask_c]
        val_mean[ii]=mean(dummy_mean_arr.flat[:])
        val_stdev[ii]=std(dummy_mean_arr.flat[:],ddof=1.0) 
    
    # Plot the relevant data
    ax2.errorbar(val_day,val_mean,yerr=val_stdev)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Average deviation with standard deviation')
    
# This controls the behavior of button_d
button_d.on_clicked(deviations)

#Generate the arrow color change buttons
rax = axes([0.25, 0.025, 0.1, 0.1], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    l_machine.set_color(label)
    l_diff.set_color(label)
    draw()
radio.on_clicked(colorfunc)

show()
