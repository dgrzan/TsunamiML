#!/usr/bin/env python

import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import transforms
import numpy as np
import csv
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from statistics import mean
import math
from scipy.stats import norm
import pickle

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self, layersize, resize, channels):
        super().__init__()
        self.layers = nn.Sequential(
            
            nn.Conv1d(channels, 5, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(5, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=5, padding=0),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=5, padding=0),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(2*256, layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.ReLU(),
            nn.Linear(layersize, layersize),
            nn.Dropout(p=0.001),
            nn.ReLU(),
            nn.Linear(layersize, 732)

        )    

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

#creates a mask labeling the points of interest for a max inundation map as 1 and the points to ignore as 0
def createmask(maxinundation1d):
    samplemax = maxinundation1d[0]
    maskarray = np.zeros(len(samplemax))
                
    for i in range(len(samplemax)):
        if(samplemax[i]==-1):
            maskarray[i]=0
        else:
            maskarray[i]=1
            
    return maskarray

#uses the mask to cut down the max inundation input array
def maskinput(maxinundation1d, maskarray):
    count = 0
    for i in range(len(maskarray)):
        if(maskarray[i]==1):
            count+=1
    maskedinput = np.zeros((len(maxinundation1d),count))

    for i in range(len(maxinundation1d)):
        count = 0
        for j in range(len(maskarray)):
            if(maskarray[j]==1):
                maskedinput[i,count] = maxinundation1d[i,j]
                count+=1

    return maskedinput

#defining error as the absolute difference in squares that had above 0.1m inundation
def customloss(outputs, targets):
    outputs = outputs.cpu().detach().numpy()[0]
    targets = targets.cpu().detach().numpy()[0]
    loss = 0
    count = 0

    for i in range(len(outputs)):
        if(outputs[i]>0.1 or targets[i]>0.1):
            loss+=abs(outputs[i]-targets[i])
            count+=1

    loss = float(loss/count)
            
    return loss

#downsamples initial earthquake file to a size of your choice
def downsample(inputarray,resize):
    inputsize = len(inputarray[0])
    ratio = float(inputsize/resize)
    outputarray = np.zeros((3000,resize))

    for k in range(3000):
        for i in range(resize):
            index = int(np.rint(float(ratio*i)))
            if(index<resize):
                outputarray[k,i] = inputarray[k,index]

    return outputarray

#converts inundation database into wet or dry (1 or 0) if above a certain threshold
def weandrdry(ydata):
    shape = np.shape(y)
    print(shape)

    yout = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            if(ydata[i,j]>1.0):
                yout[i,j]=1
            else:
                yout[i,j]=0

    return yout

#takes all of the buoy data (all 14 buoys) and puts them into one line per event
def flattendata(xdata):
    outputx = np.zeros((3000,50*60))
    j = 0
    jj = 0
    for i in range(len(xdata)):
        if(j!=i%60):
            jj=0
        j = i%60
        jj += 1
        outputx[j,jj]

    return outputx

#adds a certain number n of buoys to a list
def addsarrays(x, n):
    xout = np.zeros((3000,n*50))

    buoynumber = int(len(x)/3000)
    
    for i in range(3000):
        for j in range(n):
            np.put(xout[i], range(j*50,(j+1)*50), x[i*buoynumber+j,0:50-1]) #change +0 to +j to return it to normal

    return xout

#adds arrays so that the number of channels are the number of observation points
def addsarrayschannels(x):
    n = int(len(x)/3000)
    xout = np.zeros((3000,n,50))

    for i in range(3000):
        for j in range(n):
            np.put(xout[i,j], range(0,50), x[i*n+j])
    
    return xout

#manually calculated std
def calcstd(x):
    meann = mean(x)
    m = 0

    for i in range(len(x)):
        m += abs(x[i]-meann)**2

    return math.sqrt(float(m/len(x)))

#given a histogram with n and bins, calculates the range around the mean where x% of the data falls
def errorrange(n, bins, percent):

    meanindex = mode(n)
            
    totalvolume = sum(n)
    target = float(percent*totalvolume)

    currentvolume = n[meanindex]
    leftindex = meanindex
    rightindex = meanindex
    while(currentvolume<target):
        leftindex-=1
        rightindex+=1
        currentvolume+=n[leftindex]+n[rightindex]

    errorrangee = bins[rightindex-1]-bins[meanindex]
    return errorrangee

#for an array n for a histogram, finds the index of the most often occuring bin
def mode(n):
    meanindex = 0
    maxn = 0
    for i in range(len(n)):
        if(n[i]>maxn):
            maxn = n[i]
            meanindex = i
    return meanindex

#plots a histogram of percent error for a given test event
def ploterror(pred, exp):
    E = []
    RPE = []
    RPD = []
    for i in range(len(pred)):
        if(exp[i]>0.1):
            RPE.append(float(100*(abs(pred[i]-exp[i])/exp[i])))
            RPD.append(float(2*abs(pred[i]-exp[i])/(pred[i]+exp[i])))
            E.append(float(pred[i]-exp[i]))

    fig, ax1 = plt.subplots(figsize =(10, 6))
    ax1.hist(RPE,50,range=[0,100])
    #fig, ax2 = plt.subplots(figsize =(10, 6))
    #ax2.hist(RPD,30)
    fig, ax3 = plt.subplots(figsize =(10, 6))
    ax3.hist(E,30,range=[-2,2])

    plt.show()

#plots a histogram of percent error for the whole test set
def plottotalerror(pred, exp):
    E = []
    RPE = []
    RPD = []
    for k in range(len(pred)):
        for i in range(len(pred[0])):
            if(exp[k,i]>0.1):
                RPE.append(float(100*((pred[k,i]-exp[k,i])/exp[k,i])))
                RPD.append(float(2*abs(pred[k,i]-exp[k,i])/(pred[k,i]+exp[k,i])))
                E.append(float(pred[k,i]-exp[k,i]))

    #fig, ax = plt.subplots(figsize =(7, 5))
    #ax1.hist(list(map(abs,RPE)),100,range=[0,100])
    fig, ax1 = plt.subplots(figsize =(7, 5))
    n1, bins1,bb = ax1.hist(E,200,range=[-2,2])
    vol1 = sum(n1)*(bins1[1]-bins1[0])
    fig, ax2 = plt.subplots(figsize =(7, 5))
    n2, bins2,bb = ax2.hist(RPE,200,range=[-100,100])
    vol2 = sum(n2)*(bins2[1]-bins2[0])
    print(vol2)

    """
    m1, std1 = norm.fit(E)
    m2, std2 = norm.fit(RPE)
    print(m1,std1,m2,std2)

    x1 = np.linspace(-3,3,200)
    p1 = norm.pdf(x1,m1,std1)
    ax1.plot(x1,vol1*p1,'k')
    title = "Fit results: mean = %.2f,  std = %.2f" % (m1, std1)
    ax1.set_title(title)

    x2 = np.linspace(-300,300,600)
    p2 = norm.pdf(x2,m2,std2)
    ax2.plot(x2,vol2*p2,'k')
    title = "Fit results: mean = %.2f,  std = %.2f" % (m2, std2)
    ax2.set_title(title)
    """

    rangeE = errorrange(n1,bins1,0.8)
    rangeRPE = errorrange(n2,bins2,0.8)
    ax1.axvline(x=bins1[mode(n1)]+rangeE,color='k')
    ax1.axvline(x=bins1[mode(n1)]-rangeE,color='k')
    ax2.axvline(x=bins2[mode(n2)]+rangeRPE,color='k')
    ax2.axvline(x=bins2[mode(n2)]-rangeRPE,color='k')

    print("mean of absolute error: "+str(mean(map(abs,E))))
    print("mean of absolute percent error: "+str(mean(map(abs,RPE))))
    print("mean of error: "+str(mean(E)))
    print("std of error: "+str(np.std(E)))
    print("mean of percent error: "+str(mean(RPE)))
    print("std of percent error: "+str(np.std(RPE)))
    print("80% of the error E falls within "+str(rangeE)+" of the mean")
    print("80% of the percent error RPE falls within "+str(rangeRPE)+" of the mean")

    plt.show()

    ################################################################################################################################################
    ################################################################################################################################################
    ################################################################################################################################################
if __name__ == '__main__':

    for jj in reversed(range(10,11,5)):

        spacing = 1
        spacing = str(spacing)

        #parameters
        testnumber = 300
        validationnumber = 100
        epochs = 800
        resize = 50*60
        learningrate = 0.0005
        layersize = 800
        batchsize = 1

        # Set fixed random number seed
        torch.manual_seed(42)
        torch.set_num_threads(1)

        #prepare the dataset
        with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/input/buoyinputgrid'+spacing+'.txt', "r") as csvfile:
            datax = list(csv.reader(csvfile))
        with open('/home/davidgrzan/Tsunami/cascadia/machinelearning/maxinundation3.txt', "r") as csvfile:
            datay = list(csv.reader(csvfile))

        #change everything to numpy arrays for ease of use
        x = np.asarray(datax)
        y = np.asarray(datay)
        x = x.astype(np.float)
        y = y.astype(np.float)
        print(np.shape(x))
        channels = int(len(x)/3000)

        #puts all of the data in a row for each entry
        #x = flattendata(x)
        print(len(x)/3000)
        #x = addsarrays(x,int(len(x)/3000))
        #x = addsarrays(x,3)
        x = addsarrayschannels(x)
        print(x)
        print(np.shape(x))


        #mask the extra squares that are in the ocean or too high on land
        maskarray = createmask(y)
        y = maskinput(y,maskarray)
        print(np.shape(y))

        #convert inundation data into wet or dry (1 or 0)
        #y = weandrdry(y)
        print(np.shape(y))

        xtrain = torch.tensor(x[0:len(x[:,0])-testnumber-validationnumber,:])
        ytrain = torch.tensor(y[0:len(y[:,0])-testnumber-validationnumber,:])
        xvalid = torch.tensor(x[len(x[:,0])-testnumber-validationnumber:len(x[:,0])-testnumber,:])
        yvalid = torch.tensor(y[len(y[:,0])-testnumber-validationnumber:len(y[:,0])-testnumber,:])
        xtest = torch.tensor(x[len(x[:,0])-testnumber:,:])
        ytest = torch.tensor(y[len(y[:,0])-testnumber:,:])

        train = data_utils.TensorDataset(xtrain, ytrain)
        trainloader = data_utils.DataLoader(train, batch_size=batchsize, shuffle=False)
        xtest = xtest.unsqueeze(1)
        xvalid = xvalid.unsqueeze(1)
        print("shape of validation: "+str(xvalid.size()))

        # Initialize the MLP
        mlp = MLP(layersize, resize, channels)

        # Define the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        loss_function = nn.L1Loss()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(mlp.parameters(), lr=learningrate)

        epochlossarray = np.zeros(epochs)
        validationlossarray = np.zeros(epochs)

        lowesterror = 1000000
        lowesterrorepoch = 0
        lowesterrormodel = mlp

        # Run the training loop
        for epoch in range(0, epochs): 

            epochloss = 0
            # Print epoch
            print('Starting epoch '+str(epoch))

            # Set current loss value
            current_loss = 0.0
            epochloss = 0
            myloss = 0
            validationlossavg = 0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader, 0):

                # Get inputs
                inputs, targets = data
                #inputs = inputs.unsqueeze(1)

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = mlp(inputs.float())

                # Compute loss
                loss = loss_function(outputs.float(), targets.float())
                if(epoch==epochs-1):
                    myloss += customloss(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                epochloss += loss.item()

                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0
            if(epoch==epochs-1):
                myloss = float(myloss/(3000-testnumber-validationnumber))
                print("custom loss: "+str(myloss))

            #finds error of validation set for that epoch
            for k in range(validationnumber):
                validationoutputs = mlp(xvalid[k].float())
                validationloss = loss_function(validationoutputs, yvalid[k].float())
                validationlossavg += validationloss.item()
            validationlossavg = float(validationlossavg/validationnumber)
            validationlossarray[epoch] = validationlossavg
            print("validation loss: "+str(validationlossavg))
            if(validationlossavg<lowesterror):
                lowesterror = validationlossavg
                lowesterrorepoch = epoch
                lowesterrormodel = mlp

            epochloss = float(batchsize*epochloss/len(list(train)))
            print("epoch loss: "+str(epochloss))
            epochlossarray[epoch] = epochloss

        print("lowest error: "+str(lowesterror))
        print("epoch of lowest error: "+str(lowesterrorepoch))

        ax = plt.subplot(111)
        ax.plot(epochlossarray,"blue")
        ax.plot(validationlossarray,"red")
        #plt.show()
        pickle.dump(ax, open("/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/output/lossplots/loss"+spacing+".pickle", "wb"))

        # Process is complete. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('Training process has finished.')

        pred = lowesterrormodel(xtest.squeeze(1).float())
        pred = pred.cpu().detach().numpy()
        ytest = ytest.cpu().detach().numpy()

        prediction = np.zeros((64,62))
        actual = np.zeros((64,62))
        error = np.zeros((64,62))
        total = np.zeros((testnumber*3,64,62))

        totalavgerror = 0
        totalavgdiff = 0
        totalavgL1error = 0
        totalavgmiss = 0
        totalavgfalse = 0
        totalavgsuccess = 0

        L1sd = np.zeros(testnumber)
        confinedsd = np.zeros(testnumber)
        L1sdnum = 0
        confinedsdnum = 0

        for k in range(testnumber):
            print("saving "+str(k)+" out of "+str(testnumber)+" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            avgerror = 0
            countactual = 0
            countpred = 0
            countsuccess = 0
            countmask = -1
            realisticcount = 0
            L1loss = 0 
            L1count = 0
            for i in range(int(64*62)):

                xx = 62-int(i/62)
                yy = i%62

                if(maskarray[i]==1):
                    countmask+=1
                    #write prediciton, actual, and error to one file
                    actual[xx,yy] = ytest[k,countmask]
                    prediction[xx,yy] = pred[k,countmask]
                    error[xx,yy] = pred[k,countmask] - ytest[k,countmask]

                    L1loss+=abs(pred[k,countmask]-ytest[k,countmask])
                    L1count+=1

                    if(pred[k,countmask]>0.1 or ytest[k,countmask]>0.1):
                        avgerror+=abs(pred[k,countmask]-ytest[k,countmask])
                        realisticcount+=1 #only count a square if either the test or prediciton has an appreciable height
                    if(pred[k,countmask]>0.1):
                        countpred+=1
                    if(ytest[k,countmask]>0.1):
                        countactual+=1
                    if(pred[k,countmask]>0.1 and ytest[k,countmask]>0.1):
                        countsuccess+=1 #counting squares that were both predicted and correct

            avgerror=float(avgerror/realisticcount)
            avgdiff = abs(countpred-countactual)
            avgL1error = float(L1loss/L1count)

            L1sd[k] = avgL1error
            confinedsd[k] = avgerror

            total[k*3+0] = actual
            total[k*3+1] = prediction
            total[k*3+2] = error

            totalavgmiss += countpred-countsuccess
            totalavgfalse += countactual-countsuccess
            totalavgsuccess += countsuccess
            totalavgerror += avgerror
            totalavgdiff += avgdiff
            totalavgL1error += avgL1error

            print("average confined error: "+str(avgerror))
            print("difference in inundated squares: "+str(countpred-countactual))

            #ploterror(pred[k],ytest[k])
        L1sdnum = np.std(L1sd)
        confinedsdnum = np.std(confinedsd)

        print(L1sdnum, confinedsdnum, float(totalavgsuccess/testnumber), float(totalavgfalse/testnumber), float(totalavgmiss/testnumber))

        print("TOTAL average diff: "+str(float(totalavgdiff/testnumber)))
        print("TOTAL average confined error: "+str(float(totalavgerror/testnumber)))
        print("TOTAL average L1 error: "+str(float(totalavgL1error/testnumber)))
        #plottotalerror(pred,ytest)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        out_dataset = Dataset("/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/output/rawdata/output"+spacing+".nc", 'w', format='NETCDF4')

        out_dataset.createDimension('type', testnumber*3)
        out_dataset.createDimension('latitude', 64)
        out_dataset.createDimension('longitude', 62)
        lats_data   = out_dataset.createVariable('latitude', 'f4', ('latitude',))
        lons_data   = out_dataset.createVariable('longitude', 'f4', ('longitude',))

        #lats_data[:]       = range(0,1,64)
        #lons_data[:]       = range(0,1,62)
        height_data = out_dataset.createVariable('output', 'f4', ('type','latitude','longitude'))

        height_data[:,:,:] = total

        out_dataset.close()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        torch.save(mlp, "/home/davidgrzan/Tsunami/cascadia/machinelearning/batch/output/weights/weights"+spacing+".pt")
