# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:03:48 2022

@author: HP
"""

import pandas as pd
df = pd.read_csv("D:\\Assignments DS\\KNN\\Zoo.csv")
df.head()

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["animal name"] =LE.fit_transform(df["animal name"])

X=df.iloc[:,0:17]
Y=df.iloc[:,17]

from sklearn.preprocessing import StandardScaler  # Minmaxscaler
SS = StandardScaler()
X_scale = SS.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test  = train_test_split(X_scale,Y, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=9, p=2)

KNN.fit(X_train,Y_train)
Y_pred_train = KNN.predict(X_train)
Y_pred_test =  KNN.predict(X_test)

from sklearn.metrics import accuracy_score
Train_acc= accuracy_score(Y_train,Y_pred_train)
print("Accuracy score for Training data:", Train_acc.round(2))
Test_acc = accuracy_score(Y_test,Y_pred_test)
print("Accuracy score for Test data:", Test_acc.round(2))
