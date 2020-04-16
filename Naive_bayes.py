# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import re
import nltk.tokenize
import math
from nltk.stem import PorterStemmer
porter=PorterStemmer()
#from sklearn.model_selection import train_test_split

dataset=pd.read_csv('a1_d3.txt',sep="\n",header=None,delimiter='\t')
for i in range(len(dataset)):
    X=dataset.iloc[i,0]
    dataset.iloc[i,0] =X.lower()
    dataset.iloc[i,0] = re.sub(r'\W'," ", dataset.iloc[i,0]) 
    dataset.iloc[i,0] = re.sub(r'\s+'," ", dataset.iloc[i,0])
    stem_sentence=[]
    words=nltk.word_tokenize(dataset.iloc[i,0])
    for var in words:
        stem_sentence.append(porter.stem(var))
        stem_sentence.append(" ")
    stem_sentence="".join(stem_sentence)
    dataset.iloc[i,0]=stem_sentence
        
'''
X=dataset.iloc[:,0]
y=dataset.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
matrix = CountVectorizer(max_features=1000)
X_train = matrix.fit_transform(X_train).toarray()
X_test=matrix.transform(X_test).toarray()

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)


y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
'''
import operator
av_accuracy=0
min_accuracy=1
max_accuracy=0
for t in range(1,6):
    split_data=int(t*len(dataset)/5)
    split_data_begin=int((t-1)*len(dataset)/5)
    X_test=dataset.iloc[split_data_begin:split_data,:]
    X_train=dataset.iloc[:split_data_begin,:]
    X_train=pd.concat([X_train,dataset.iloc[split_data:,:]])   
    
    uniqueword={}
    for sent in X_train.iloc[:,0]:
        word=nltk.word_tokenize(sent)
        for var in word:
            if var not in uniqueword:
                uniqueword[var]=1
            else:
                uniqueword[var]+=1

    accuracy_max=0
    best_size=50
    listofdict=sorted(uniqueword.items(),key=operator.itemgetter(1),reverse=True)
    listofdict_var=listofdict[:750]
    listofdicti=[i[0] for i in listofdict_var]
    X_t=[]
    for sent in X_train.iloc[:,0]:
        word=nltk.word_tokenize(sent)
        vector=np.zeros(len(listofdicti))
        for var in word:
            if var in listofdicti:
                vector[listofdicti.index(var)]=1
        X_t.append(vector)
    X_t=np.asarray(X_t)
    tot_yes=X_train[1].sum(axis=0)
    tot_no=len(X_train)-tot_yes
    prob_arr_yes=np.zeros((tot_yes,len(listofdicti)))
    prob_arr_no=np.zeros((tot_no,len(listofdicti)))
    count_yes=0
    count_no=0
    for i in range(len(X_train)):
        if X_train.iloc[i,1]==1:
            prob_arr_yes[count_yes]=X_t[i,:]
            count_yes+=1
        else:
            prob_arr_no[count_no]=X_t[i,:]
            count_no+=1
    
    prob_vec_yes=((prob_arr_yes.sum(axis=0)+1)/len(prob_arr_yes))
    prob_vec_no=((prob_arr_no.sum(axis=0)+1)/len(prob_arr_no))
    
    prob_yes=(tot_yes/len(X_train))
    prob_no=(tot_no/len(X_train))
    
    Y_pred=np.zeros(len(X_test))
    Y_test=np.asarray(X_test.iloc[:,1])
    
    i=0
    for sent in X_test.iloc[:,0]:
        word=nltk.word_tokenize(sent)
        prob=1
        for var in word:
            if var in listofdicti:
                prob=prob*prob_vec_yes[listofdicti.index(var)]
        prob_temp_yes=prob*prob_yes
        prob=1
        for var in word:
            if var in listofdicti:
                prob=prob*(prob_vec_no[listofdicti.index(var)])
        prob_temp_no=prob*prob_no
        if prob_temp_yes>prob_temp_no:
            Y_pred[i]=1
        else:
            Y_pred[i]=0
        i+=1
    
    count=0;
    for i in range(len(X_test)):
        if Y_pred[i]==Y_test[i]:
            count+=1
    
    accuracy=count/len(X_test)
    print(accuracy)
    av_accuracy=accuracy+av_accuracy
    if(accuracy<min_accuracy):
        min_accuracy=accuracy
    if(accuracy>max_accuracy):
        max_accuracy=accuracy
av_accuracy/=5
diff=max(abs(max_accuracy-av_accuracy),abs(min_accuracy-av_accuracy))
print(av_accuracy,"+-",round(diff,2))    
    
                        
