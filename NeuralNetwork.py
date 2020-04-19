# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 01:49:17 2020

@author: LENOVO
"""
import numpy as np
import pandas as pd
from sklearn import model_selection

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],5) 
        self.weights2   = np.random.rand(5,1)                       
        self.y          = y.reshape(len(y),1)
        self.output     = np.zeros(self.y.shape)
    
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
        self.layer1=(self.layer1-np.mean(self.layer1))/np.std(self.layer1)

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, ((self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot((self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        d_weights2=(d_weights2-np.mean(d_weights2))/np.std(d_weights2)
        d_weights1=(d_weights1-np.mean(d_weights1))/np.std(d_weights1)
        self.weights1 += 0.05*d_weights1
        self.weights2 += 0.05*d_weights2


dataset=pd.read_csv('housepricedata.csv')
x=(1.0*dataset.iloc[:,:-1].values)
y=(1.0*dataset.iloc[:,-1].values)
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(x,y,test_size=0.2)
'''X_train=(1.0*dataset.iloc[:int(0.8*len(dataset)),:-1].values)
Y_train=1.0*dataset.iloc[:int(0.8*len(dataset)),-1].values
Y_test=1.0*dataset.iloc[int(0.8*len(dataset)):,-1].values
X_test=1.0*dataset.iloc[int(0.8*len(dataset)):,:-1].values'''
i=0
for i in range(X_train.shape[1]):
    X_train[:,i]=((X_train[:,i]-X_train[:,i].min())/((X_train[:,i]).max()-(X_train[:,i]).min()))
    X_test[:,i]=((X_test[:,i]-X_test[:,i].min())/((X_test[:,i]).max()-(X_test[:,i]).min()))

NN=NeuralNetwork(X_train, Y_train)
err=np.sum(pow((NN.y-NN.output),2))
NN.feedforward()
print(err)
i=0
while i in range(10000):
    NN.backprop()
    NN.feedforward()
    err=np.sum(pow((NN.y-NN.output),2))
    #print(i)
    i+=1
'''Y_train_pred=np.zeros(len(NN.y))
i=0
for i in range(len(NN.y)):
    if NN.output[i]>0.5:
        Y_train_pred[i]=1
    else:
        Y_train_pred[i]=0
i=0
count=0
for i in range(len(NN.y)):
    if Y_train_pred[i]==Y_train[i]:
        count+=1
accuracy=count/len(X_train)'''

layer1=sigmoid(np.dot(X_test,NN.weights1))
output_arr=sigmoid(np.dot(layer1,NN.weights2))
Y_pred=np.zeros(len(X_test))
i=0
for i in range(len(X_test)):
    if output_arr[i]>0.5:
        Y_pred[i]=1
    else:
        Y_pred[i]=0
i=0
count=0
for i in range(len(X_test)):
    if Y_pred[i]==Y_test[i]:
        count+=1
accuracy=count/len(X_test)
print(accuracy)               
        

    

