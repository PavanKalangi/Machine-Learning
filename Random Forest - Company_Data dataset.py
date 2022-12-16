# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:35:11 2022

@author: HP
"""

import pandas as pd
df = pd.read_csv("D:\\Assignments DS\\random forest\\Company_Data.csv")

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

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=200,max_samples=0.9,max_features=0.6,max_depth=(4)) 
RFR.fit(X_train, Y_train)

Y_pred_train = RFR.predict(X_train) 
Y_pred_test = RFR.predict(X_test)

from sklearn.metrics import mean_squared_error
print("Training Error: ",mean_squared_error(Y_train,Y_pred_train).round(2))
print("Test Error: ",mean_squared_error(Y_test,Y_pred_test).round(2))