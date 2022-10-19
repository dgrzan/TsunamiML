#!/usr/bin/env python

import numpy as np
import csv
from batch1 import *
import random
import matplotlib.pyplot as plt
import torch
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap


def loaddata(spacing):

    spacing = str(spacing)
    
    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/input/buoyinputgrid'+spacing+'.txt', "r") as csvfile:
        datax = list(csv.reader(csvfile))
    with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/maxinundation3.txt', "r") as csvfile:
        datay = list(csv.reader(csvfile))

    x = np.asarray(datax)
    y = np.asarray(datay)
    x = x.astype(np.float)
    y = y.astype(np.float)

    x = addsarrayschannels(x)
    maskarray = createmask(y)
    y = maskinput(y,maskarray)

    testnumber = 300
    xtest = torch.tensor(x[len(x[:,0])-testnumber:,:])
    ytest = torch.tensor(y[len(y[:,0])-testnumber:,:])

    mlp = torch.load("/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/output/weights/weights"+spacing+".pt")

    #pred = mlp(xtest.squeeze(1).float())
    #pred = pred.cpu().detach().numpy()
    ytest = ytest.cpu().detach().numpy()

    return mlp, xtest, ytest

def analysis(whichone):

    sens = []
    
    mlp, x, y = loaddata(whichone)
    pred = mlp(x.squeeze(1).float())
    pred = pred.cpu().detach().numpy()
    baseerror = geterror(pred, y)
    #print(baseerror)

    for i in range(len(x[0])):
        print(i)
        modx = modify(x,i)
        pred = mlp(modx.squeeze(1).float())
        pred = pred.cpu().detach().numpy()
        error = geterror(pred, y)
        #print(float(error/baseerror))
        sens.append(abs(float(error/baseerror)-1))

    maxsens = max(sens)
    sens = [float(x/maxsens) for x in sens]
    sens = [x+1 for x in sens]
    


    sens = np.array(sens)
    np.save("/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/sens/sens"+str(whichone)+".npy",sens)

    return sens

def modify(x,n):
    modx = torch.clone(x)
    for i in range(len(x)):
            for k in range(len(x[0,0])):
                modx[i,n,k] = 0
        
    return modx

def indexlatlon(whichone):
    filename = "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/simulations/lowres1.nc"
            
    simdata = Dataset(filename, "r", format="NETCDF4")
    upliftdata = np.array(simdata.variables['level'])
    alt = np.array(simdata.variables['altitude'])
    lat = np.array(simdata.variables['latitude'])
    lon = np.array(simdata.variables['longitude'])

    spacing = whichone
    
    latlonlist = []
    count = 0

    for i in range(len(lon)):
        for j in range(len(lat)):
            if(i%spacing==0 and j%spacing==0):
                if(alt[0,j,i]<-10 and lon[i]>-128):
                    latlonlist.append([lon[i],lat[j]])
                    count+=1

    return latlonlist

#returns an array of the average custom errors of each testing event
def geterror(pred, exp, cutoff=0.15):
    output = np.zeros(len(pred))

    for i in range(len(pred)):
        count = 0
        eventerror = 0
        for j in range(len(pred[0])):
                if(pred[i,j]>cutoff or exp[i,j]>cutoff):
                    eventerror += abs(pred[i,j]-exp[i,j])
                    count+=1
        if(count==0): count=1
        output[i] = float(eventerror/count)
                    
    return mean(output)

def avgwave(m):
    filename = "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/simulations/avg.nc"
    simdata = Dataset(filename, "r", format="NETCDF4")
    level = np.array(simdata.variables['level'])
    alt = np.array(simdata.variables['altitude'])
    lat = np.array(simdata.variables['latitude'])
    lon = np.array(simdata.variables['longitude'])

    
    z = level[50]
    sens = np.zeros(np.array(z).shape)
    print(z.shape)
    x = lon
    y = lat
    xform = np.zeros(len(x))
    yform = np.zeros(len(y))
    for i in range(len(x)):
        for j in range(len(y)):
            xx, yy = m(x[i],y[j])
            xform[i] = xx
            yform[j] = yy


    col = [colors.to_rgb("lightblue"),colors.to_rgb("darkblue")]
    cmap = LinearSegmentedColormap.from_list("Custom", col, N=20)
    plt.pcolormesh(xform,yform,z,cmap=cmap,vmin=1,vmax=2.0,zorder=1,alpha=1.0)
    cbar = plt.colorbar(location="left", pad=-0.05)

    


if __name__ == "__main__":
    
    whichone = 95
    loadbool = False

    if(loadbool==True):
        sensitivity = np.load("/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/sens/sens"+str(whichone)+".npy")
    else:
        sensitivity = analysis(whichone)
    latlon = indexlatlon(whichone)
    lons = [row[0] for row in latlon]
    lats = [row[1] for row in latlon]


    m = Basemap(projection='mill',llcrnrlat=40,urcrnrlat=50, llcrnrlon=-132 ,urcrnrlon=-118, resolution="f")
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='lightgray')

    avgwave(m)
    
    lons, lats = m(lons, lats)
    alphasens = [(x-1) for x in sensitivity]
    m.scatter(lons,lats,marker=".",c=sensitivity,cmap="Reds",s=100)
    x, y = m(-123.9+0.2,45.9-0.2)
    plt.text(x,y,"Seaside")
    x, y = m(-123.9226+0.05,45.9932-0.05)
    plt.plot(x,y,"o",markersize=4,color="red")
    cbar = plt.colorbar(fraction=0.040, pad=0.01)
    
    plt.show()
