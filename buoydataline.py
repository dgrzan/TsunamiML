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

def find(loclist, latarr, lonarr):
    
    indexlist = np.zeros((len(loclist),2))
    
    #find buoys that fall outside the simulation range
    xhigh = max(lonarr)
    xlow = min(lonarr)
    yhigh = max(latarr)
    ylow = min(latarr)
        
    for i in range(len(loclist)):
        if(loclist[i][0]>yhigh or loclist[i][0]<ylow):
            loclist = np.delete(loclist,i)
            print("deleted element "+str(i))
            continue
        if(loclist[i][1]>xhigh or loclist[i][1]<xlow):
            loclist = np.delete(loclist,i)
            print("deleted element "+str(i))
            continue

    #finds the index of each buoy
    for i in range(len(loclist)):
        lat = loclist[i][0]
        lon = loclist[i][1]
        lati = 0
        loni = 0
        for j in range(len(lonarr)):
            if(lonarr[j]>lon):
                loni = j
                break
        for j in range(len(latarr)):
            if(latarr[j]<lat):
                lati = j
                break

        indexlist[i][0] = int(loni)
        indexlist[i][1] = int(lati)
    
    return indexlist


def locations(filename):
    with open("/home/davidgrzan/Tsunami/cascadia/machinelearning/"+filename) as f:
        lines = f.readlines()
    finallocations = np.zeros((int(len(lines)/2),2))

    for i in range(len(lines)):
        lines[i] = float(lines[i])
        if(i%2==0):
            finallocations[int(i/2)][0] = lines[i]
        else:
            finallocations[int(i/2)][1] = -lines[i]

    return finallocations


def heights(locarr, uplift, timesteplimit):

    heightarr = np.zeros((len(locarr), timesteplimit))
    
    for i in range(len(locarr)):
        for j in range(timesteplimit):
            heightarr[i,j] = format(uplift[j,int(locarr[i][1]),int(locarr[i][0])], ".2f")

    return heightarr

def gridindex(spacing, alt, lat, lon):

    indexlist = []
    count = 0

    for i in range(len(lon)):
        for j in range(len(lat)):
            if(i%spacing==0 and j%spacing==0):
                if(alt[0,j,i]<-10 and lon[i]>-128):
                    indexlist.append([i,j])
                    count+=1

    return indexlist, count
            
        
if __name__ == "__main__":

    outputar = np.zeros((3000*4,50)) #every 16 seconds for all 50 entries (13 min)

    for i in range(3000):
        name = "lowres"+str(i+1)
        print(i+1)
        filename = name+".nc"
        path = "/media/davidgrzan/My Book/databasefullreduced/"
            
        simdata = Dataset(path+filename, "r", format="NETCDF4")
        upliftdata = np.array(simdata.variables['level'])
        altitude = np.array(simdata.variables['altitude'])
        simlatitude = np.array(simdata.variables['latitude'])
        simlongitude = np.array(simdata.variables['longitude'])
            
        buoyheights = upliftdata[:50,120,420]
        outputar[i*4+0] = buoyheights
        buoyheights = upliftdata[:50,120,405]
        outputar[i*4+1] = buoyheights
        buoyheights = upliftdata[:50,120,390]
        outputar[i*4+2] = buoyheights
        buoyheights = upliftdata[:50,120,375]
        outputar[i*4+3] = buoyheights
    print(outputar)

    np.savetxt("/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/input/buoyinputgrid"+str(1)+".txt", outputar, delimiter=",", fmt="%.3f")

    if(True):
        altitude[0,120,420] = 10000000
        altitude[0,120,405] = 10000000
        altitude[0,120,390] = 10000000
        altitude[0,120,375] = 10000000

        out_dataset = Dataset("/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/input/buoyinputgrid"+str(1)+".nc", 'w', format='NETCDF4')
        
        out_dataset.createDimension('latitude', np.size(simlatitude))
        out_dataset.createDimension('longitude', np.size(simlongitude))
        
        lats_data   = out_dataset.createVariable('latitude', 'f4', ('latitude',))
        lons_data   = out_dataset.createVariable('longitude', 'f4', ('longitude',))
        altitude_data = out_dataset.createVariable('buoy_locations', 'f4', ('latitude','longitude'))
        
        lats_data[:]       = simlatitude
        lons_data[:]       = simlongitude
        altitude_data[:,:] = altitude[0]
        
        out_dataset.close()
