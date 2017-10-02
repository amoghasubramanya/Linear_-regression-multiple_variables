# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 22:40:43 2017

@author: Amogha Subramanya

Linear regression with multiple variables to predict the prices of houses. 
Suppose you are selling your house and you want to know what a go o d market 
price would be. One way to do this is to first collect information on 
recent houses sold and make a model of housing prices.
The file ex1data2.txt contains a training set of housing prices in Portland, Oregon.
The first column is the size of the house (in square feet), the
second column is the number of bedrooms, and the third column is the price
of the house.
"""
import csv
import matplotlib.pyplot as plt
import numpy as np

def featureNormalize(X):
    means=np.mean(X,axis=0)
    stdev=np.std(X,axis=0)
    X = (X - means)/stdev
    return means,stdev,X

def computeCost(X,y,theta):
    J=0
    m=len(X)
    hx=list()
    #Compute Hypothesis
    hx=theta*X
    hofx=np.array(sum(hx,axis=1))
    #Compute sum of squared error
    sqrerror=hofx-y
    sqrerror=np.square(sqrerror)
    J=(1/(2*m))*np.sum(sqrerror)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m=len(y)
    J_history=list()
    J_history.append(computeCost(X,y,theta))
    for i in range(iterations):
        hx=theta*X
        hofx=np.array(sum(hx,axis=1))             #h(x)=theta0*x0+theta1*x1+theta2*x
        err=hofx-y                                #predicted-actual
        t=X*err[:, np.newaxis]                    #X*err
        t1=np.sum(t,axis=0)                       #Sum of(X*err)
        t1=(alpha/m)*t1                           
        theta-=t1                                 #Simulataneously update theta
        J_history.append(computeCost(X,y,theta))
    return J_history,theta

def plotCost(cost):
        iters = list()
        for i in range(len(cost)):
            iters.append(i+1)
        plt.figure("Cost v/s Iterations")
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.plot(iters, cost, 'b')
        
def predict(theta,X,means,stddev):
    X = (X - means)/stddev
    x1=np.array([1])
    X=np.concatenate((x1, X))
    return np.sum(X*theta)
    
        
#Reading from the input file.
print('Linear regression with multiple variables to predict the prices of houses')
dataset=list()
fp=open('ex1data2.txt','r')
reader = csv.reader(fp, delimiter=',')
for row in reader:
    dataset.append(row)

data=np.array(dataset, dtype=int)
m=len(data)
X=data[:,0:2]
y=data[:,2]
means,stddev,X=featureNormalize(X)


#Adding x0=1
x1=(np.ones((m,1), dtype=int))
X=np.concatenate((x1, X), axis=1)
#Features 

#Initial cost with theta=[0,0,0]
theta=[0,0,0]
J=computeCost(X,y,theta)
    
print('With Theta=[0,0,0]')
print('Cost=', J)

alpha=0.01
iterations=1500


print('Gradient Descent:')
print('Number of iterations= ', iterations)
print('Alpha= ',alpha)

#Run GradientDescent
cost,theta= gradientDescent(X, y, theta, alpha, iterations);
print('Theta found by running gradient Descent', theta)
plotCost(cost)
#Predicting price for house with size=1650 and 3 bedrooms
xi=[1650,3]
p=predict(theta,xi,means,stddev)
print('House price with 1650sq.ft and 3 bedrooms = ',p)