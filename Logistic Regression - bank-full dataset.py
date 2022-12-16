# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:12:55 2022

@author: HP
"""

import pandas as pd
df = pd.read_csv("D:\\Assignments DS\\loistic reg\\bank-full1.csv")

X=df.iloc[:,:16]
Y=df.iloc[:,16]
df.dtypes

#X=X.drop(X.columns[[8,9,10,15]],axis=1,inplace=True)

df_cont = df[df.columns[[0,5,11,12,13,14]]]
df_cont

from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_cont = MM.fit_transform(df_cont)
MM_cont =  pd.DataFrame(MM_cont)
MM_cont.shape


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["job"]=LE.fit_transform(df["job"])
df["marital"]=LE.fit_transform(df["marital"])
df["education"]=LE.fit_transform(df["education"])
df["default"]=LE.fit_transform(df["default"])
df["housing"]=LE.fit_transform(df["housing"])
df["loan"]=LE.fit_transform(df["loan"])

X_scale = pd.concat([MM_cont,df["job"],df["marital"],df["education"],df["default"],df["housing"],df["loan"]], axis=1,ignore_index=True)


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_scale,Y)

Y_pred = LR.predict(X_scale)

from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score

ac = accuracy_score(Y,Y_pred)
print("Accuracy",accuracy_score(Y,Y_pred).round(2))
print("Sensitivity",recall_score(Y,Y_pred).round(2))
print("f1_score",f1_score(Y,Y_pred).round(2))
print("precision",precision_score(Y,Y_pred).round(2))
