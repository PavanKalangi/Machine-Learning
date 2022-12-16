# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:13:03 2022

@author: HP
"""

import pandas as pd
df1=pd.read_csv("D:\\Assignments DS\\Naive bayes\\SalaryData_Train.csv")

df_con = df1[df1.columns[[0,3,9,10,11]]]
df_cat = df1[df1.columns[[1,2,4,5,6,7,8,12]]]
df_cat
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,8):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])
df_new = pd.concat([df_con,df_cat],axis=1,ignore_index=True)

X_train=df_new.iloc[:,0:13]
Y_train=df1.iloc[:,13]

df2=pd.read_csv("D:\\Assignments DS\\Naive bayes\\SalaryData_Test.csv")

df_con = df2[df2.columns[[0,3,9,10,11]]]
df_cat = df2[df2.columns[[1,2,4,5,6,7,8,12]]]
df_cat
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for i in range(0,8):
    df_cat.iloc[:,i] = LE.fit_transform(df_cat.iloc[:,i])

df_new1 = pd.concat([df_con,df_cat],axis=1,ignore_index=True)

X_test=df_new1.iloc[:,0:13]
Y_test=df2.iloc[:,13]
    

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

Y_pred_train = MNB.predict(X_train)
Y_pred_test = MNB.predict(X_test)

from sklearn.metrics import accuracy_score

Training_score = accuracy_score(Y_train,Y_pred_train)
Test_score = accuracy_score(Y_test,Y_pred_test)

print("Training score",Training_score.round(3))
print("Test score",Test_score.round(3))