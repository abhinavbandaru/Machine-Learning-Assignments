import pandas as pd
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt

data = pd.read_csv('housepricedata.csv')

def normalize(dataset):
    dataNorm=((dataset-dataset.mean())/(dataset.std()))
    dataNorm["AboveMedianPrice"]=dataset["AboveMedianPrice"]
    return dataNorm
    
df = normalize(data)
df.insert(0,'Nothing',1)
x = np.array(df.drop(['AboveMedianPrice'],1))
y = np.array(df['AboveMedianPrice'])
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1.0 - np.tanh(z)**2

#initializing weights for first layer
w1 = np.zeros((11,9))

#initializing weights for 2nd layer
w2 = np.random.rand(9,1)

#learning rate
global learnRate
learnRate = 0.1

def feedforward(w1, x, y, w2):
    n=len(y)
    y=y[:,np.newaxis]
    z1 = tanh(x @ w1)
    y_pred = sigmoid(z1 @ w2)
    error = 0
    for i in range(x.shape[0]):
        error += pow((y[i][0]-y_pred[i]),2)
    cost = (1/2)* (1/n) * error
    return cost,y_pred,z1

def backprop(w1,x,y,w2):
    n=len(y)
    y=y[:,np.newaxis]
    it=0
    Error_history =[]
    while True:
        cost,y_pred,z1 = feedforward(w1,x,y,w2)
        grad2 = 1/n * ((z1.transpose()) @ ((y_pred - y)*(sigmoid_derivative(y_pred))))
        grad1 = 1/n * ((x.transpose()) @ (tanh_derivative(z1)*((y_pred - y) @ (w2.transpose()))))
        w1 = w1 - (learnRate * grad1)
        w2 = w2 - (learnRate * grad2)
        if it>2 and Error_history[-1]-cost<=pow(10,-6):
            print("No of iterations:",it)    
            return w1,Error_history,w2
        it = it + 1
        Error_history.append(cost)

def testaccuracy(w1,w2,y):
    global truepositive,truenegative
    truepositive = 0
    truenegative = 0
    y=y[:,np.newaxis]
    cost,y_pred,z1 = feedforward(w1, x_test, y, w2)
    for i in range(y_pred.shape[0]):
        if y_pred[i]>=0.5 and y[i]==1:
            truepositive = truepositive + 1
        if y_pred[i]<0.5 and y[i]==0:
            truenegative = truenegative + 1
    return (truepositive + truenegative)/y_pred.shape[0]

def fscore(w1,w2,y):
    global truepositive,falsepositive,falsenegative
    truepositive = 0
    falsepositive = 0
    falsenegative = 0
    y=y[:,np.newaxis]
    cost,y_pred,z1 = feedforward(w1, x_test, y, w2)
    for i in range(y_pred.shape[0]):
        if y_pred[i]>=0.5 and y[i]==1:
            truepositive = truepositive + 1
        if y_pred[i]>=0.5 and y[i]==0:
            falsepositive = falsepositive + 1
        if y_pred[i]<0.5 and y[i]==1:
            falsenegative = falsenegative + 1
    return (2*truepositive)/(2*truepositive + falsepositive + falsenegative)

w1,Error_history,w2 = backprop(w1,x_train,y_train,w2)
print("The final weights for 1st layer:\n",w1)
print("The final weights for 2nd layer:\n",w2)
print("Testing Accuracy:",testaccuracy(w1,w2,y_test)*100,"%")
print("F-Score:",fscore(w1,w2,y_test)*100,"%")
plt.plot(Error_history)
plt.xlabel("No of Iterations")
plt.ylabel("$Error(\Theta)$")
plt.title("Cost function using 2 layered Neural Network")
plt.show()


