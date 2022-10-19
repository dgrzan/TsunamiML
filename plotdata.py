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
from statistics import variance
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap

#loads outputdata of a machine learning trial. returns: expected, predicted 
def loaddata(name):
    name = str(name)
    filename = "output"+str(name)+".nc"
    path = "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/output/rawdata/"
    simdata = Dataset(path+filename, "r", format="NETCDF4")
    inundationdata = np.array(simdata.variables['output'])

    expected = inundationdata[::3,:,:]
    predicted = inundationdata[1::3,:,:]

    return expected, predicted

#loads the nc files with the buoy locations in them
def loadbuoys(name):
    name = str(name)
    filename = "buoyinputgrid"+str(name)+".nc"
    path = "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/input/"
    simdata = Dataset(path+filename, "r", format="NETCDF4")
    buoylocations = np.array(simdata.variables['buoy_locations'])
    
    return buoylocations

#finds the closest buoy and calculates the distance in km
def findclosest(data):
    buoyindex = []

    for i in range(len(data)):
        for j in range(len(data[0])):
            if(data[i,j]>1000000):
                buoyindex.append([i,j])

    minlength = 999999999999999999
    for i in range(len(buoyindex)):
        if(minlength>((buoyindex[i][1]-425)**2+(buoyindex[i][0]-120)**2)):
            minlength=((buoyindex[i][1]-425)**2+(buoyindex[i][0]-120)**2)

    return math.sqrt(minlength)*1.8

#takes in a single 2d array and plots it as a colormesh with levels ranging from 0m to 15m
def plotcolormesh(data):
    x = np.linspace(-1,1,len(exp[0,0,:]))
    y = np.linspace(-1,1,len(exp[0,:,0]))
    x, y = np.meshgrid(x, y)
    z = data

    fig = plt.figure()
    ax = plt.axes()
    cmap = cm.get_cmap("jet",lut=10)
    im = ax.pcolormesh(x, y, z, vmin=0, vmax=10, cmap=cmap)
    fig.colorbar(im, ax=ax)

#returns an array of the average custom errors of each testing event
def getcustomerrorevent(pred, exp, cutoff=0.15):
    output = np.zeros(len(pred))

    for i in range(len(pred)):
        count = 0
        eventerror = 0
        for j in range(len(pred[0])):
            for k in range(len(pred[0,0])):
                if(pred[i,j,k]>cutoff or exp[i,j,k]>cutoff):
                    eventerror += abs(pred[i,j,k]-exp[i,j,k])
                    count+=1
        if(count==0): count=1
        output[i] = float(eventerror/count)
                    
    return output

#returns an array of the error for each pixel
def geterrortotal(pred, exp, cutoff=0.15):
    output = []

    for i in range(len(pred)):
        for j in range(len(pred[0])):
            for k in range(len(pred[0,0])):
                if(pred[i,j,k]>cutoff or exp[i,j,k]>cutoff):
                    output.append(pred[i,j,k]-exp[i,j,k])
                    
    return output

#returns an array of the error percentage for each pixel
def geterrortotalpercent(pred, exp, cutoff=0.15):
    output = []

    for i in range(len(pred)):
        for j in range(len(pred[0])):
            for k in range(len(pred[0,0])):
                if(exp[i,j,k]>cutoff and pred[i,j,k]>cutoff):
                    output.append(float(100*(pred[i,j,k]-exp[i,j,k])/exp[i,j,k]))
                    
    return output

#returns an array of the error percentage for each event
def geterrorpercentevent(pred, exp, cutoff=0.15):
    output = np.zeros(len(pred))

    for i in range(len(pred)):
        count = 0
        eventerror = 0
        for j in range(len(pred[0])):
            for k in range(len(pred[0,0])):
                if(pred[i,j,k]>cutoff and exp[i,j,k]>cutoff):
                    eventerror += float(abs(pred[i,j,k]-exp[i,j,k])/exp[i,j,k])
                    count+=1
        if(count==0): count=1
        output[i] = float(100*eventerror/count)
                    
    return output

#returns an array for the percent difference in squares covered
def getdiff(pred, exp, cutoff=0.15):
    output = np.zeros(len(pred))

    for i in range(len(pred)):
        countpred = 0
        countexp = 0
        for j in range(len(pred[0])):
            for k in range(len(pred[0,0])):
                if(pred[i,j,k]>cutoff):
                    countpred+=1
                if(exp[i,j,k]>cutoff):
                    countexp+=1
        if(countexp==0):
            countexp=1
        output[i] = float(100*abs(countpred-countexp)/countexp)
                    
    return output

#creates a mask showing the water
def createmask():
    filename = "highres1.nc"
    path = "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/"
    simdata = Dataset(path+filename, "r", format="NETCDF4")
    alt = np.array(simdata.variables['altitude'])

    alt = alt[0,:,200:]
    alt = alt[::-1,:]
    for i in range(len(alt)-1):
        for j in range(len(alt[0])):
            if(alt[i+1,j]<=0):
                alt[i,j] = np.nan
    return alt

#applies mask to existing array
def applymask(data):
    mask = createmask()
    output = ma.masked_where(np.isnan(mask),data)

    return output

#creates an array that is NaN for all values not land
def land(data,cutoff=0.15):
    output = np.zeros(data.shape)
    
    for i in range(len(data)):
        for j in range(len(data[0])):
            if(data[i,j]<cutoff):
                output[i,j]=1
            else:
                output[i,j]=np.nan

    return ma.masked_where(np.isnan(output), output)
    

#plots 6 examples of test events to show accuracy
def plotexamples(whichone):
    shift = 93
    number = 6
    
    whichone = str(whichone)
    exp, pred = loaddata(whichone)
    fig = plt.figure(figsize=(7,float(10/6)*number))
    cmap = cm.get_cmap("turbo",lut=15)
    cmap.set_bad(color="white")
    cmap2 = cm.get_cmap("gray")
    #cmap2.set_bad(color="gray")
    gs = fig.add_gridspec(number, 3, hspace=0, wspace=0)
    ax = gs.subplots(sharex='col', sharey='row')
    for i in range(number):
        for j in range(3):
            x = np.linspace(-1,1,len(exp[0,0,:]))
            y = np.linspace(-1,1,len(exp[0,:,0]))
            x, y = np.meshgrid(x, y)
            if(j==0):
                z1 = exp[i+shift]
                z1 = applymask(z1)
                landz = land(z1)
                z = z1.copy()
            elif(j==1):
                z2 = pred[i+shift]
                z2 = applymask(z2)
                landz = land(z2)
                z = z2.copy()
            else:
                z = list(map(abs,z2-z1))
                z = applymask(z)
                landz = land(z1+z2,cutoff=0.15)
            if(number==1):
                im = ax[j].pcolormesh(x, y, z, vmin=0, vmax=15, cmap=cmap)
                ax[j].pcolormesh(x, y, landz, vmin=0, vmax=1.5, cmap=cmap2, alpha=1)
                ax[j].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
            else:
                im = ax[i,j].pcolormesh(x, y, z, vmin=0, vmax=15, cmap=cmap)
                ax[i,j].pcolormesh(x, y, landz, vmin=0, vmax=1.5, cmap=cmap2, alpha=1)
                ax[i,j].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(range(0,16,1))
    plt.show()

#plots buoy locations on a map
def plotlocations(whichone):
    name = str(whichone)
    filename = "buoyinputgrid"+str(name)+".nc"
    path = "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/input/"
    simdata = Dataset(path+filename, "r", format="NETCDF4")
    buoylocations = np.array(simdata.variables['buoy_locations'])

    filename = "/media/davidgrzan/My Book/databasefullreduced/lowres1.nc"
    simdata = Dataset(filename, "r", format="NETCDF4")
    upliftdata = np.array(simdata.variables['level'])
    alt = np.array(simdata.variables['altitude'])
    lat = np.array(simdata.variables['latitude'])
    lon = np.array(simdata.variables['longitude'])

    buoyindex = []

    for i in range(len(buoylocations)):
        for j in range(len(buoylocations[0])):
            if(buoylocations[i,j]>1000000):
                buoyindex.append([i,j])
    np.array(buoyindex)

    latlon = []
    for i in range(len(buoyindex)):
        longitude = lon[buoyindex[i][1]]
        latitude = lat[buoyindex[i][0]]
        latlon.append([longitude,latitude])
    np.array(latlon)


    m = Basemap(projection='mill',llcrnrlat=40,urcrnrlat=50, llcrnrlon=-132 ,urcrnrlon=-118, resolution="f")
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray')

    lons = [row[0] for row in latlon]
    lats = [row[1] for row in latlon]

    lons, lats = m(lons, lats)
    
    m.scatter(lons,lats,marker=".",color="navy",s=100)
    x, y = m(-123.9+0.2,45.9-0.2)
    plt.text(x,y,"Seaside")
    x, y = m(-123.9226+0.05,45.9932-0.05)
    plt.plot(x,y,"o",markersize=4,color="red")

    return 0

#returns the average event custom error of the absolute value
def avgcustomerrorevent(data):
    
    return np.mean(data)

#returns the average custom error of the aboslute value
def avgerrortotal(data):
    
    return np.mean(list(map(abs,data)))

#plots a histogram of the custom error of each event
def plotcustomerrorevent(data):
    fig, ax1 = plt.subplots()
    ax1.hist(data,20)

#plots a histogram of all of the error from all pixels
def ploterrortotal(data):
    fig, ax2 = plt.subplots()
    ax2.hist(list(map(abs,data)),400,range=[-7,7])
    
#plots a histogram of all of the error percentages from all pixels
def ploterrortotalpercent(data):
    fig, ax = plt.subplots()
    ax.hist(data,400,range=[-400,400])



if __name__ == "__main__":

    """
    #get the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    exp, pred = loaddata("10")
    error = abs(pred-exp)

    #plot the data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    customerrorarray = getcustomerrorevent(exp,pred,cutoff=0.2)
    plotcustomerrorevent(customerrorarray)

    errortotalarray = geterrortotal(pred,exp,cutoff=0.2)
    ploterrortotal(errortotalarray)

    errortotalpercentarray = geterrortotalpercent(exp,pred,cutoff=0.2)
    ploterrortotalpercent(errortotalpercentarray)

    print(avgcustomerrorevent(customerrorarray))
    print(avgerrortotal(errortotalarray))
    
    """
    #plotexamples(10)
    plotlocations(1)
    plt.show()

    """
    exp, pred = loaddata(10)
    plotcolormesh(pred[5])
    plt.show()
    """
    
    errorevent = []
    eventvariance = []
    percenterrorevent = []
    percenteventvariance = []
    maxdist = []
    percentiles = []
    percentiles2 = []
    percentilesabs = []
    percentilesdiff = []
    
    x = range(5,100,5)
    x = [1.8*y for y in x]
    
    fig, ax3 = plt.subplots()

    for i in range(5,100,5):
        if(i==5):
            i=1
        print(i)
        exp, pred = loaddata(str(i))
        customerrorarray = getcustomerrorevent(exp,pred,cutoff=0.15)
        percenterroreventarray = geterrorpercentevent(pred,exp,cutoff=0.15)
        #print(max(range(len(percenterroreventarray)), key=percenterroreventarray.__getitem__))
        percenterroreventarray2 = geterrorpercentevent(pred,exp,cutoff=1.0)
        diffarray = getdiff(exp,pred,cutoff=0.15)
        errorevent.append(avgcustomerrorevent(customerrorarray))
        eventvariance.append(variance(customerrorarray))
        percenterrorevent.append(np.mean(percenterroreventarray))
        percenteventvariance.append(variance(percenterroreventarray))
        percentiles.append([np.percentile(percenterroreventarray,50),np.percentile(percenterroreventarray,10),np.percentile(percenterroreventarray,90)])
        percentiles2.append([np.percentile(percenterroreventarray2,50),np.percentile(percenterroreventarray2,10),np.percentile(percenterroreventarray2,90)])
        percentilesabs.append([np.percentile(customerrorarray,50),np.percentile(customerrorarray,10),np.percentile(customerrorarray,90)])
        percentilesdiff.append([np.percentile(diffarray,50),np.percentile(diffarray,10),np.percentile(diffarray,90)])
        if(i==10 or i==95):
            errortotalpercentarray = geterrortotalpercent(exp,pred,cutoff=0.15)
            if(i==10):
                color = "b"
            else:
                color = "r"
            ax3.hist(errortotalpercentarray,400,range=[-400,400],color=color,alpha=0.5)

        maxdist.append(findclosest(loadbuoys(str(i))))


    fig, ax = plt.subplots()
    #ax.errorbar(x,errorevent,yerr=eventvariance, capsize=3, capthick=1, marker=".",color="blue")
    ax.plot(x,errorevent,color="blue",linewidth=1, marker=".")
    ax.fill_between(x, np.add(errorevent, [-x for x in eventvariance]), np.add(errorevent,eventvariance), alpha=0.25)
    ax.set_ylim((0,2))
    ax.set_xlabel("Buoy Separation Distance (km)")
    ax.set_ylabel("Average Error (m)")
    
    fig, ax2 = plt.subplots()
    #ax2.errorbar(x,percenterrorevent,yerr=percenteventvariance, capsize=3, capthick=1, marker=".",color="blue")
    ax2.plot(x,percenterrorevent,color="Blue",linewidth=1,marker=".")
    #ax2.fill_between(x, np.add(percenterrorevent, [-x for x in percenteventvariance]), np.add(percenterrorevent,percenteventvariance), alpha=0.25)
    #ax2.set_ylim((0,2))
    ax2.set_xlabel("Buoy Separation Distance (km)")
    ax2.set_ylabel("Average Percent Error (%)")

    fig, ax4 = plt.subplots()
    ax4.plot(x[1:],maxdist[1:],marker=".",color="black")
    ax4.set_xlabel("Buoy Separation Distance (km)")
    ax4.set_ylabel("Distance of Closest Buoy to Seaside, OR (km)")

    percentiles = np.array(percentiles)
    percentiles2 = np.array(percentiles2)
    fig, ax5 = plt.subplots()
    #ax5.errorbar(x,percentiles[:,0],yerr=np.array([percentiles[:,0]-percentiles[:,1],percentiles[:,2]-percentiles[:,0]]), capsize=3, capthick=1, marker=".",color="blue")
    ax5.plot(x[1:],percentiles[1:,0],color="darkgreen",linewidth=1,marker=".")
    ax5.fill_between(x[1:], percentiles[1:,1], percentiles[1:,2], alpha=0.25, color="darkgreen")
    #ax5.errorbar(x,percentiles2[:,0],yerr=np.array([percentiles2[:,0]-percentiles2[:,1],percentiles2[:,2]-percentiles2[:,0]]), capsize=3, capthick=1, marker=".",color="red")
    #ax5.plot(x,percentiles2[:,0],color="red",linewidth=1,marker=".")
    #ax5.fill_between(x, percentiles2[:,1], percentiles2[:,2], alpha=0.25, color="red")
    #ax5.set_ylim((0,2))
    ax5.set_xlabel("Buoy Separation Distance (km)")
    ax5.set_ylabel("Error Percentage (%)")
    ax5.hlines(y=percentiles[0,0], xmin=18, xmax=95*1.8, linewidth=1, color="black")
    ax5.hlines(y=percentiles[0,1], xmin=18, xmax=95*1.8, linewidth=1, color="black", linestyle="--")
    ax5.hlines(y=percentiles[0,2], xmin=18, xmax=95*1.8, linewidth=1, color="black", linestyle="--")

    percentilesabs = np.array(percentilesabs)
    fig, ax6 = plt.subplots()
    #ax6.errorbar(x,percentilesabs[:,0],yerr=np.array([percentilesabs[:,0]-percentilesabs[:,1],percentilesabs[:,2]-percentilesabs[:,0]]), capsize=3, capthick=1, marker=".",color="blue")
    ax6.plot(x[1:],percentilesabs[1:,0],color="navy",linewidth=1,marker=".")
    ax6.fill_between(x[1:], percentilesabs[1:,1], percentilesabs[1:,2], alpha=0.25, color="navy")
    #ax6.set_ylim((0,2))
    ax6.set_xlabel("Buoy Separation Distance (km)")
    ax6.set_ylabel("Absolute Error (m)")
    ax6.hlines(y=percentilesabs[0,0], xmin=18, xmax=95*1.8, linewidth=1, color='black')
    ax6.hlines(y=percentilesabs[0,1], xmin=18, xmax=95*1.8, linewidth=1, color="black", linestyle="--")
    ax6.hlines(y=percentilesabs[0,2], xmin=18, xmax=95*1.8, linewidth=1, color="black", linestyle="--")

    percentilesdiff = np.array(percentilesdiff)
    fig, ax7 = plt.subplots()
    #ax7.errorbar(x,percentilesdiff[:,0],yerr=np.array([percentilesdiff[:,0]-percentilesdiff[:,1],percentilesdiff[:,2]-percentilesdiff[:,0]]), capsize=3, capthick=1, marker=".",color="blue")
    ax7.plot(x[1:],percentilesdiff[1:,0],color="maroon",linewidth=1,marker=".")
    ax7.fill_between(x[1:], percentilesdiff[1:,1], percentilesdiff[1:,2], alpha=0.25, color="maroon")
    #ax7.set_ylim((0,2))
    ax7.set_xlabel("Buoy Separation Distance (km)")
    ax7.set_ylabel("Percentage Difference in Coverage(%)")
    ax7.hlines(y=percentilesdiff[0,0], xmin=18, xmax=95*1.8, linewidth=1, color='black')
    ax7.hlines(y=percentilesdiff[0,1], xmin=18, xmax=95*1.8, linewidth=1, color="black", linestyle="--")
    ax7.hlines(y=percentilesdiff[0,2], xmin=18, xmax=95*1.8, linewidth=1, color="black", linestyle="--")





    plt.show()

