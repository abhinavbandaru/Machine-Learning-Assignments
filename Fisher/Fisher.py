import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#getting the dataset ready
dataset=pd.read_csv("a1_d1.csv",header=None)
ind_1=np.where(dataset.iloc[:,-1].values==1)
ind_0=np.where(dataset.iloc[:,-1].values==0)

#calculating m1 and m2
m1=sum(dataset.iloc[ind_1].values[:,:-1])/len(dataset.iloc[ind_1])
m2=sum(dataset.iloc[ind_0].values[:,:-1])/len(dataset.iloc[ind_0])
m1=m1.reshape(len(m1),1)
m2=m2.reshape(len(m2),1)

#calculating Sw and w
mat_m1=np.zeros((len(m1),len(m1)))
mat_m2=np.zeros((len(m2),len(m2)))
for i in range(len(dataset)):
    if dataset.iloc[i,-1:].values==1:
        mat_m1+=np.dot(np.transpose(dataset.iloc[i,:-1].values-m1),np.asarray(dataset.iloc[i,:-1].values-m1))
    else:
        mat_m2+=np.dot(np.transpose(dataset.iloc[i,:-1].values-m2),np.asarray(dataset.iloc[i,:-1].values-m2))
mat_m1=mat_m1/len(dataset.iloc[ind_1])    
mat_m2=mat_m2/len(dataset.iloc[ind_0])
Sw=mat_m1+mat_m2
w=np.dot(np.linalg.inv(Sw),(m1-m2))

#transforming into 1D
i=0
trans_vec=np.zeros(len(dataset))
for i in range(len(dataset)):
    trans_vec[i]=np.dot(np.transpose(w),dataset.iloc[i,:-1].values)

#plotting normal distribution
pnt_m1=trans_vec[ind_1]
pnt_m2=trans_vec[ind_0]
std_m1=np.std(pnt_m1)
std_m2=np.std(pnt_m2)
mean_m1=np.mean(pnt_m1)
mean_m2=np.mean(pnt_m2)
pnt_m1.sort()
pnt_m2.sort()
plt.plot(pnt_m1,stats.norm.pdf(pnt_m1,mean_m1,std_m1))
plt.plot(pnt_m2,stats.norm.pdf(pnt_m2,mean_m2,std_m2))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Normal Distribution")
plt.show()

#plotting the points in 1D
plt.plot(pnt_m1,len(pnt_m1)*[1],'ro')
plt.plot(pnt_m2,len(pnt_m2)*[1],'bs')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Plot of transformed points in 1 Dimension")
plt.show()

#finding the intersection pt
a = 1/(2*std_m1**2) - 1/(2*std_m2**2)
b = mean_m2/(std_m2**2) - mean_m1/(std_m1**2)
c = mean_m1**2 /(2*std_m1**2) - mean_m2**2 / (2*std_m2**2) - np.log(std_m2/std_m1)
inter_mat=np.roots([a,b,c])
if inter_mat[0]<=max(mean_m2,mean_m1) and min(mean_m2,mean_m1)<=inter_mat[0]:
    inter_pt=inter_mat[0]
else:
    inter_pt=inter_mat[1]

#calculating accuracy
if mean_m1<inter_pt:
    less=1
    more=0
else:
    less=0
    more=1
pred=np.zeros((len(dataset),1))
test=dataset.iloc[:,-1].values
for i in range(len(dataset)):
    if trans_vec[i]<=inter_pt:
        pred[i]=less    
    else:
        pred[i]=more
count=0
i=0
for i in range(len(dataset)):
    if pred[i]==test[i]:
        count+=1
accuracy=count/len(dataset)
print("Testing Accuracy for dataset 1 :",accuracy*100,"%")

truepositive = 0
falsepositive = 0
falsenegative = 0
for i in range(len(dataset)):
    if pred[i]==1 and test[i]==1:
        truepositive = truepositive + 1
    if pred[i]==1 and test[i]==0:
        falsepositive = falsepositive + 1
    if pred[i]==0 and test[i]==1:
        falsenegative = falsenegative + 1
fscore = (2*truepositive)/(2*truepositive + falsepositive + falsenegative)
print("F-Score for dataset 1:",fscore*100,"%")

#getting the 2nd dataset ready
dataset=pd.read_csv("a1_d2.csv",header=None)
ind_1=np.where(dataset.iloc[:,-1].values==1)
ind_0=np.where(dataset.iloc[:,-1].values==0)

#calculating m1 and m2
m1=sum(dataset.iloc[ind_1].values[:,:-1])/len(dataset.iloc[ind_1])
m2=sum(dataset.iloc[ind_0].values[:,:-1])/len(dataset.iloc[ind_0])
m1=m1.reshape(len(m1),1)
m2=m2.reshape(len(m2),1)

#calculating Sw and w
mat_m1=np.zeros((len(m1),len(m1)))
mat_m2=np.zeros((len(m2),len(m2)))
for i in range(len(dataset)):
    if dataset.iloc[i,-1:].values==1:
        mat_m1+=np.dot(np.transpose(dataset.iloc[i,:-1].values-m1),np.asarray(dataset.iloc[i,:-1].values-m1))
    else:
        mat_m2+=np.dot(np.transpose(dataset.iloc[i,:-1].values-m2),np.asarray(dataset.iloc[i,:-1].values-m2))
mat_m1=mat_m1/len(dataset.iloc[ind_1])    
mat_m2=mat_m2/len(dataset.iloc[ind_0])
Sw=mat_m1+mat_m2
w=np.dot(np.linalg.inv(Sw),(m1-m2))

#transforming into 1D
i=0
trans_vec=np.zeros(len(dataset))
for i in range(len(dataset)):
    trans_vec[i]=np.dot(np.transpose(w),dataset.iloc[i,:-1].values)

#plotting normal distribution
pnt_m1=trans_vec[ind_1]
pnt_m2=trans_vec[ind_0]
std_m1=np.std(pnt_m1)
std_m2=np.std(pnt_m2)
mean_m1=np.mean(pnt_m1)
mean_m2=np.mean(pnt_m2)
pnt_m1.sort()
pnt_m2.sort()
plt.plot(pnt_m1,stats.norm.pdf(pnt_m1,mean_m1,std_m1))
plt.plot(pnt_m2,stats.norm.pdf(pnt_m2,mean_m2,std_m2))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Normal Distribution")
plt.show()

#plotting the points in 1D
plt.plot(pnt_m1,len(pnt_m1)*[1],'ro')
plt.plot(pnt_m2,len(pnt_m2)*[1],'bs')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Plot of transformed points in 1 Dimension")
plt.show()

#finding the intersection pt
a = 1/(2*std_m1**2) - 1/(2*std_m2**2)
b = mean_m2/(std_m2**2) - mean_m1/(std_m1**2)
c = mean_m1**2 /(2*std_m1**2) - mean_m2**2 / (2*std_m2**2) - np.log(std_m2/std_m1)
inter_mat=np.roots([a,b,c])
if inter_mat[0]<=max(mean_m2,mean_m1) and min(mean_m2,mean_m1)<=inter_mat[0]:
    inter_pt=inter_mat[0]
else:
    inter_pt=inter_mat[1]

#calculating accuracy
if mean_m1<inter_pt:
    less=1
    more=0
else:
    less=0
    more=1
pred=np.zeros((len(dataset),1))
test=dataset.iloc[:,-1].values
for i in range(len(dataset)):
    if trans_vec[i]<=inter_pt:
        pred[i]=less    
    else:
        pred[i]=more
count=0
i=0
for i in range(len(dataset)):
    if pred[i]==test[i]:
        count+=1
accuracy=count/len(dataset)
print("Testing Accuracy for dataset 2 :",accuracy*100,"%")

truepositive = 0
falsepositive = 0
falsenegative = 0
for i in range(len(dataset)):
    if pred[i]==1 and test[i]==1:
        truepositive = truepositive + 1
    if pred[i]==1 and test[i]==0:
        falsepositive = falsepositive + 1
    if pred[i]==0 and test[i]==1:
        falsenegative = falsenegative + 1
fscore = (2*truepositive)/(2*truepositive + falsepositive + falsenegative)
print("F-Score for dataset 2:",fscore*100,"%")
