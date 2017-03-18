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

def get_weather():
    # Read in data 
    filename='catalogs/napa_model_wind.txt'
    #filename='catalogs/test_model_wind.txt'
    #filename='catalogs/frei_model_wind.txt'
    
    # Array of days
    time = np.genfromtxt(filename,dtype=np.int,usecols=[0],skiprows=2)

    # Arrays of coordinates,velocities and concentration 
    coor = np.genfromtxt(filename,dtype=np.int,usecols=[1,2],skiprows=2)
    vel_file = np.genfromtxt(filename,dtype=np.float,usecols=[3,4],skiprows=2)
    c = np.genfromtxt(filename,dtype=np.float,usecols=[5],skiprows=2)
    
    #If the wind data is in (U,V coordinates leave as is)
    vel=vel_file
    
    #If the wind data is in Direction/Speed coordinates uncomment to convert to U,V coordinates
    #vel[:,0]=-vel_file[:,1]*sin((pi/180.0)*vel_file[:,0])
    #vel[:,1]=-vel_file[:,1]*cos((pi/180.0)*vel_file[:,0])
  
    # Outputs
    return time,coor,vel,c 

def gen_arrays(days,coor,vel,c):
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
        
    #Outputs
    return coor_arr_x,coor_arr_y,vel_arr_x,vel_arr_y,c_arr,n_days,max_x,min_x,max_y,min_y


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
    
    # Need to re-establish the scaling
    v=sqrt(vx**2+vy**2)
    low_v=find(v.flat[:]<0.001)
    high_v=find(v.flat[:]>0.001)
    vx.flat[low_v]=nan
    vy.flat[low_v]=nan
    scale=1.8*v.flat[high_v].mean()*max(10,math.sqrt(v.flat[high_v].size))
    l.scale=scale
    im_l.set_data(ctration)
    l.set_UVC(vx,vy)
    draw()

# Define the plot window
ax = subplot(111)
subplots_adjust(left=0.25, bottom=0.25)

#Obtain the time and data from the files
time,coor,vel,c=get_weather()
coor_arr_x,coor_arr_y,vel_arr_x,vel_arr_y,c_arr,n_days,max_x,min_x,max_y,min_y=gen_arrays(time,coor,vel,c)

# Initialize the plot window starting on Day 1
x=coor_arr_x[0,:,:]
y=coor_arr_y[0,:,:]
vx=vel_arr_x[0,:,:]
vy=vel_arr_y[0,:,:]
ctration=c_arr[0,:,:]
ctration=ctration.T
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

# Plot the initial image with concentration 
im_l=pl.imshow(ctration,origin='lower',aspect=1.0,vmin=c_min,vmax=c_max)
pl.colorbar()
# Plot the initial velocity field
l=quiver(x, y, vx, vy, pivot='middle',scale=scale, headwidth=4, headlength=6)
# Zoom the plot the portion of the plot where there is relevant data
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

#Generate the arrow color change buttons
rax = axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    draw()
radio.on_clicked(colorfunc)

show()
