#!/usr/bin/env python

import csv
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import math
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import gc
from scipy import interpolate
from sys import getsizeof
import gc
from scipy import interpolate

def np_bivariate_normal_pdf(domain, mean, variance):
    X = np.arange(-domain+mean, domain+mean, variance)
    Y = np.arange(-domain+mean, domain+mean, variance)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = ((1. / np.sqrt(2 * np.pi)) * np.exp(-.5*R**2))
    return X+mean, Y+mean, Z



if __name__ == "__main__":

    name = "highres7red"
    name = "highresavg1red"
    filename = name+".nc"
    path = "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/simulations/"

    simdata = Dataset(path+filename, "r", format="NETCDF4")
    simtime = np.array(simdata.variables['time'])
    simaltitude1 = np.array(simdata.variables['altitude'])
    simaltitude = simaltitude1[0:1,:,:]
    print("altitude done")
    del simaltitude1
    gc.collect()
    simlevel = np.array(simdata.variables['level'])
    simheight = np.array(simdata.variables['height'])
    simlatitude = np.array(simdata.variables['latitude'])
    simlongitude = np.array(simdata.variables['longitude'])


    #higher resolution
    level2 = []
    level3 = []
    alt2 = []
    xx = np.linspace(0,100,61)
    yy = np.linspace(0,100,64)
    xxx = np.linspace(0,100,120)
    yyy = np.linspace(0,100,120)
    yyy = yyy[::-1]

    #simlevel = np.where(simheight<0, simaltitude, np.nan)


    print(simlevel.shape)
    for i in range(len(simlevel)):
        f = interpolate.interp2d(xx,yy,simheight[i])
        level2.append(f(xxx,yyy))
        f3 = interpolate.interp2d(xx,yy,simlevel[i])
        level3.append(f3(xxx,yyy))

    f2 = interpolate.interp2d(xx,yy,simaltitude)
    alt2.append(f2(xxx,yyy))

    for i in range(len(level2)):
        for j in range(len(level2[0])):
            for k in range(len(level2[0][0])):                
                if(alt2[0][j][k]<0):
                    level2[i][j][k] = level2[i][j][k] + alt2[0][j][k]
                else:
                    level2[i][j][k] = level2[i][j][k] 
                
                    

    simlevel = np.array(level2)
    simaltitude = np.array(alt2)

    timesteps = len(simlevel[:,0,0])
    x, y = np.meshgrid(simlongitude, simlatitude)
    x, y = np.meshgrid(xxx,yyy)
    level = np.where(simaltitude<simlevel, simlevel, np.nan)
    simaltitude = np.where(simaltitude>-200, simaltitude, np.nan)
    simaltitude = simaltitude*8
    level = level*30
    #i = 20
    #level = level[i:i+1]


    fig = plt.figure(figsize=(7,7))
    ax = fig.gca(projection='3d')
    #ax.set_aspect("auto")
    ax.pbaspect = [0.1,1.0,3.0]
    ax.relim()
    ax.autoscale_view()
        
    def update(frame,zarray,plot,plot2):
        print(frame)
        plot2[0].remove()
        plot2[0] = ax.plot_surface(x, y, level[frame,:,:], rstride=1, cstride=1, cmap='Blues_r', norm=normalize)

    normalize = matplotlib.colors.Normalize(vmin=200, vmax=300) 
    normalize2 = matplotlib.colors.Normalize(vmin=-3000, vmax=4000)
    plot = [ax.plot_surface(x, y, simaltitude[0,:,:], rstride=1, cstride=1, cmap='gist_earth', norm=normalize2)]
    plot2 = [ax.plot_surface(x, y, level[0,:,:], rstride=1, cstride=1, cmap='Blues_r', norm=normalize)]
    ax.view_init(30, 240) #perfect diagonal view
    #ax.view_init(50, 270) #more vertical 
    ax.set_zlim(-8000,8000)
    ani = animation.FuncAnimation(fig, update, fargs=(level, plot, plot2), interval=1, frames=len(level)-1)

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='finaldistance', artist='Matplotlib', comment='Animation')
    writer = FFMpegWriter(fps=10,  bitrate=1)
    #ani.save(path+name+".mp4",writer = writer)
    print("wrote")
    
    plt.show()
