# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:05:15 2022

@author: HP
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
df = pd.read_csv("D:\\Assignments DS\\5\\ToyotaCorolla.csv",encoding='latin1')

df.drop(df.columns[[0,1,4,5,7,9,10,11,14,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]],axis=1,inplace=True)
df.dtypes
df.corr()

#model 1 r2score 86.4% and multicollinearity issues exist
X= df.iloc[:,1:]
Y=df["Price"]
#model 2 r2score 80.5% and  no multicollinearity issues exist
X = df[["Age_08_04","Weight"]]
Y=df["Price"]

#model 3 r2score 84.8% and multicollinearity issues exist
X = df[["Age_08_04","Weight","KM"]]
Y=df["Price"]

#model 4 r2score 83.6% and multicollinearity issues exist
X = df[["Age_08_04","Weight","HP"]]
Y=df["Price"]

#model 5 r2score 80.5% and multicollinearity issues exist
X = df[["Age_08_04","Weight","Doors"]]
Y=df["Price"]

#model 6 r2score 80.6% and multicollinearity issues exist
X = df[["Age_08_04","Weight","cc"]]
Y=df["Price"]

#model 7 r2score 86.4% and no multicollinearity issues exist
X = df[["Age_08_04","Weight","KM","HP","Gears","Quarterly_Tax"]]
Y=df["Price"]


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)

from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred).round(2)
r2 = r2_score(Y,Y_pred).round(2)
print(mse)
print(r2)

#using stats models
import statsmodels.api as sma
X_new = sma.add_constant(X)
lm2 = sma.OLS(Y,X_new).fit()
lm2.summary()