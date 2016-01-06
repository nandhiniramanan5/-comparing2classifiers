from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import sys
import classalgorithms as algs
from scipy import stats
i=0
j=0
k=0
L1=[]
L2=[]
lamb=[]
DS = np.genfromtxt('C:\Users\Nandini\Documents\Textbooks\ML\\a3barebones\\susysubset.csv', delimiter=',')

def cvsplit(Xtrain,ytrain,k,i,numinputs):
#split function to create folds
    limit=len(Xtrain)/k
    start=i*limit
    end=(i+1)*limit
    if start==0:
           train=dataset[end:,0:numinputs]
           tlabel=dataset[end:,numinputs]
    if end==len(dataset):
           train=dataset[:start,0:numinputs]
           tlabel=dataset[:start,numinputs]
    if start!=0 and end !=len(dataset):
           train=dataset[:start,0:numinputs]
           train=np.vstack((train,dataset[end:,0:numinputs]))
           #print Xtrain.shape
           tlabel=dataset[:start,numinputs]
           tlabel=np.append(tlabel,dataset[end:,numinputs])
    test=dataset[start:end,0:numinputs]
    label=dataset[start:end,numinputs] 
    #print Xtrain.shape,ytrain.shape,Xtest.shape,ytest.shape
    return train,tlabel,test,label
 
def splitdataset(dataset, trainsize=500, testsize=300, testfile=None):
    numinputs = dataset.shape[1]-1
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]
    
    if testfile is not None:
        testdataset = loadcsv(testfile)
        Xtest = dataset[:,0:numinputs]
        ytest = dataset[:,numinputs]        
    k=10
   
    train,tlabel,test,label=cvsplit(Xtrain,ytrain,k,i,numinputs)
    # Add a column of ones; done after to avoid modifying entire dataset
    '''Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))'''
    
    #print Xtrain.shape, ytrain.shape,Xtest.shape,ytest.shape
    return ((train,tlabel), (test,label),(Xtrain,ytrain),(Xtest,ytest))
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy(dataset):
    trainset, testset,ortrainset, ortestset = splitdataset(dataset)   
    return trainset,testset,ortrainset, ortestset


if __name__ == '__main__':
    #linearregwgtlabel = ["lamda1", "lamda2", "lamda3", "lamda4", "lamda5","lamda6","lamda7","lamda8","lamda9","lamda10"]
    linearregwgtval=[0.1,0.01,0.05,.005,.0005,.001,.0001,.00001,.000001,1] 
    haccuracy = {}
    h2accuracy={}
    for j in range(30):
        size = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
        indices = np.random.randint(0,DS.shape[0],size)
        dataset = DS[indices]
        for i in range(10):
            trainset, testset, ortrainset, ortestset = loadsusy(dataset)
            classalgs = {
                    'Linear Regression': algs.LinearRegressionClass(),
                    'Logistic Regression': algs.LogitRegL2(),
                    }
            for learnername, learner in classalgs.iteritems():
                print 'Running learner = ' + learnername
                learner.learn(trainset[0], trainset[1],linearregwgtval[i])
                predictions = learner.predict(testset[0])
                accuracy = getaccuracy(testset[1], predictions)
                if learnername=='Linear Regression':
                    if linearregwgtval[i] not in haccuracy:
                        haccuracy[linearregwgtval[i]]=0
                    haccuracy[linearregwgtval[i]]+=accuracy
                elif learnername=='Logistic Regression':
                    if linearregwgtval[i] not in h2accuracy:
                        h2accuracy[linearregwgtval[i]]=0
                    h2accuracy[linearregwgtval[i]]+=accuracy
            #print haccuracy,h2accuracy
            efflamda=max(haccuracy,key=haccuracy.get)
            efflamda2=max(h2accuracy,key=h2accuracy.get)
            #print efflamda,efflamda2
            haccuracy={}
            h2accuracy={}
        for learnername, learner in classalgs.iteritems():    
            if learnername=='Linear Regression':
                learner.learn(ortrainset[0], ortrainset[1],efflamda)
                predictions = learner.predict(ortestset[0])
                Faccuracy = getaccuracy(ortestset[1], predictions)
                #print Faccuracy
                L1.append(Faccuracy)
            elif learnername=='Logistic Regression':
                learner.learn(ortrainset[0], ortrainset[1],efflamda2)
                predictions = learner.predict(ortestset[0])
                Faccuracy = getaccuracy(ortestset[1], predictions)
                #print Faccuracy
                L2.append(Faccuracy)
    #t-test                    
    pvalue=stats.ttest_ind(L1, L2)
    print 'pvalue is :'
    print pvalue
    Hypothesis=False
    for val in pvalue:
        if val>.05:
            Hypothesis=True
    
    if Hypothesis==False:
        #false null hypothesis
        print 'false null hypothesis, take mean and compare'
        if(np.mean(L1)>np.mean(L2)):
            print 'Linear regression is performing better as its mean is better'
        else:
            print 'Logistics regression is performing better as its mean is better'
    