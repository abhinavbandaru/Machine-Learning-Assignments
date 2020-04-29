import pandas as pd
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt

data = pd.read_csv('data_banknote_authentication.txt')
data.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

def normalize(dataset):
    dataNorm=((dataset-dataset.mean())/(dataset.std()))
    dataNorm["class"]=dataset["class"]
    return dataNorm
    
df = normalize(data)
df.insert(0,'Nothing',1)
x = np.array(df.drop(['class'],1))
y = np.array(df['class'])
x_train,x_test,y_train,y_test = model_selection.train_test_split(x,y,test_size=0.2)

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

#initializing weights
initial_w = np.random.rand(5,1)
#uniform sample points from uniform distribution over [0,1)

#regularization parameter(l)
global l
l = 0
#learning rate
global learnRate
learnRate = 0.1

def L2RegularizedCostFunction(w, x, y):
    n=len(y)
    y=y[:,np.newaxis]
    y_pred = sigmoid(x @ w)
    logerror = (-y * np.log(y_pred)) - ((1-y)*np.log(1-y_pred))
    cost = 1/n * sum(logerror)
    regularizedCost= cost + l/(2*n)*sum(w**2)
    grad = 1/n * (x.transpose() @ (y_pred - y))[0:] + (l/n)* w[0:]
    return regularizedCost, grad

def L2gradientDescent(w,x,y):
    n=len(y)
    it=0
    Error_history =[]
    while True:
        cost, grad = L2RegularizedCostFunction(w,x,y)
        w = w - (learnRate * grad)
        if it>2 and Error_history[-1]-cost<=pow(10,-6):
            print("No of iterations:",it)    
            return w,Error_history
        it = it + 1
        Error_history.append(cost)

def L1RegularizedCostFunction(w, x, y):
    n=len(y)
    y=y[:,np.newaxis]
    y_pred = sigmoid(x @ w)
    logerror = (-y * np.log(y_pred)) - ((1-y)*np.log(1-y_pred))
    cost = 1/n * sum(logerror)
    regularizedCost= cost + l/(n)*sum(abs(w))
    grad = 1/n * (x.transpose() @ (y_pred - y))[0:] + (l/n)
    return regularizedCost, grad

def L1gradientDescent(w,x,y):
    n=len(y)
    it=0
    Error_history =[]
    while True:
        cost, grad = L1RegularizedCostFunction(w,x,y)
        w = w - (learnRate * grad)
        if it>2 and Error_history[-1]-cost<=pow(10,-6):
            print("No of iterations:",it)    
            return w,Error_history
        it = it + 1
        Error_history.append(cost)


def testaccuracy(w,y):
    global truepositive,truenegative
    truepositive = 0
    truenegative = 0
    y=y[:,np.newaxis]
    y_pred = sigmoid(x_test @ w)
    for i in range(y_pred.shape[0]):
        if y_pred[i]>=0.5 and y[i]==1:
            truepositive = truepositive + 1
        if y_pred[i]<0.5 and y[i]==0:
            truenegative = truenegative + 1
    return (truepositive + truenegative)/y_pred.shape[0]

def fscore(w,y):
    global truepositive,falsepositive,falsenegative
    truepositive = 0
    falsepositive = 0
    falsenegative = 0
    y=y[:,np.newaxis]
    y_pred = sigmoid(x_test @ w)
    for i in range(y_pred.shape[0]):
        if y_pred[i]>=0.5 and y[i]==1:
            truepositive = truepositive + 1
        if y_pred[i]>=0.5 and y[i]==0:
            falsepositive = falsepositive + 1
        if y_pred[i]<0.5 and y[i]==1:
            falsenegative = falsenegative + 1
    return (2*truepositive)/(2*truepositive + falsepositive + falsenegative)

#WITHOUT REGULARZATION
l=0
print("1.WITHOUT REGULARIZATION")
w,Error_history = L2gradientDescent(initial_w,x_train,y_train)
print("The final w using Gradient Descent without Regularization:\n",w)
print("Testing Accuracy:",testaccuracy(w,y_test)*100,"%")
print("F-Score:",fscore(w,y_test)*100,"%")
plt.plot(Error_history)
plt.xlabel("No of Iterations")
plt.ylabel("$Error(\Theta)$")
plt.title("Cost function using Gradient Descent without Regularization")
plt.show()

#L1 REGULARZATION
l=1
print("2.L1 REGULARIZATION")
w,Error_history = L1gradientDescent(initial_w,x_train,y_train)
print("The final w using using Gradient Descent with L1 Regularization:\n",w)
print("Testing Accuracy:",testaccuracy(w,y_test)*100,"%")
print("F-Score:",fscore(w,y_test)*100,"%")
plt.plot(Error_history)
plt.xlabel("No of Iterations")
plt.ylabel("$Error(\Theta)$")
plt.title("Cost function using Gradient Descent with L1 Regularization")
plt.show()

#L2 REGULARZATION
l=1
print("3.L2 REGULARIZATION")
w,Error_history = L2gradientDescent(initial_w,x_train,y_train)
print("The final w using Gradient Descent with L2 Regularization:\n",w)
print("Testing Accuracy:",testaccuracy(w,y_test)*100,"%")
print("F-Score:",fscore(w,y_test)*100,"%")
plt.plot(Error_history)
plt.xlabel("No of Iterations")
plt.ylabel("$Error(\Theta)$")
plt.title("Cost function using Gradient Descent with L2 Regularization")
plt.show()

