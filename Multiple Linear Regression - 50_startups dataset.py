# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:23:51 2022

@author: HP
"""
#step 1: import data
import pandas as pd
df = pd.read_csv("D:\\Assignments DS\\5\\50_Startups.csv")
df.head()

#step 2: Data visualisationv
df.plot.scatter(x="R&D Spend" , y="Profit")
df.plot.scatter(x="Administration" , y="Profit")
df.plot.scatter(x="Marketing Spend" , y="Profit")
df.corr()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["State"] = LE.fit_transform(df["State"])

#step 3:splitting the variables
#model 1 #r2 score= 95% and multicollinearity issues exist
X=df.iloc[:,0:4] 
Y=df["Profit"]

#model 2 #r2 score= 95% and multicollinearity issues doesnot exist
X=df[["R&D Spend"]] 
Y=df["Profit"]

#model 3 #r2 score=95% and multicollinearity issues exist
X=df[["R&D Spend","Marketing Spend"]] 
Y=df["Profit"]

#model 4 #r2 score= 95% and multicollinearity issues exist
X=df[["R&D Spend","Administration"]] 
Y=df["Profit"]

#model 5 #r2 score= 61% and multicollinearity issues exist
X=df[["Marketing Spend","Administration"]]
Y=df["Profit"]

#model 6 #r2 score= 95% and multicollinearity issues exist
X=df[["R&D Spend","State"]] 
Y=df["Profit"]

#step 4:model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)

#step5:metrics
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred).round(2)
r2 = r2_score(Y,Y_pred).round(2)
print(mse)
print(r2)

#stats models

import statsmodels.api as sma
X_new = sma.add_constant(X)
lm2 = sma.OLS(Y,X_new).fit()
lm2.summary()
