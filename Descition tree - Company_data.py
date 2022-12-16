# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:54:52 2022

@author: HP
"""
import pandas as pd
df= pd.read_csv("D:\\Assignments DS\\descition tree\\Company_Data.csv")

df.shape
list(df)
df.head()
df.dtypes


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["ShelveLoc"] = LE.fit_transform(df["ShelveLoc"])
df["Urban"] = LE.fit_transform(df["Urban"])
df["US"] = LE.fit_transform(df["US"])

X=df.iloc[:,1:]
Y=df["Sales"]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3)

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=(7)) 
dt.fit(X_train, Y_train)

Y_pred_train = dt.predict(X_train) 
Y_pred_test = dt.predict(X_test) 

from sklearn.metrics import mean_squared_error
print("Training Error: ",mean_squared_error(Y_train,Y_pred_train).round(2))
print("Test Error: ",mean_squared_error(Y_test,Y_pred_test).round(2))

from sklearn.ensemble import BaggingRegressor
bag = BaggingRegressor(base_estimator=(dt),n_estimators=350,max_samples=0.5,max_features=0.8) 
bag.fit(X_train, Y_train)

Y_pred_train = bag.predict(X_train) 
Y_pred_test = bag.predict(X_test) 

from sklearn.metrics import mean_squared_error,r2_score
print("Training Error: ",mean_squared_error(Y_train,Y_pred_train).round(2))
print("Test Error: ",mean_squared_error(Y_test,Y_pred_test).round(2))

train_error = []
test_error = []


for i in range(1,500,1):
    bag = BaggingRegressor(base_estimator=(dt),n_estimators=350,max_samples=0.5,max_features=0.8,random_state=i) 
    bag.fit(X_train, Y_train)
    Y_pred_train = bag.predict(X_train) 
    Y_pred_test = bag.predict(X_test) 
    train_error.append(mean_squared_error(Y_train,Y_pred_train).round(2))
    test_error.append(mean_squared_error(Y_test,Y_pred_test).round(2))

import numpy as np
np.mean(train_error).round(2) 
np.mean(test_error).round(2) 

#even though bagging is done we are not able to get good accuracy i.e we are not able to get less variance between training and test data hence this data needs any other ML model