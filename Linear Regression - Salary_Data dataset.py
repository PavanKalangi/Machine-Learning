# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 15:56:01 2022

@author: HP
"""

#2) Salary_hike -> Build a prediction model for Salary_hike
#step 1
import pandas as pd
import numpy as np

#step 1
df = pd.read_csv("D:\\Assignments DS\\4\\Salary_Data.csv")
df.head

#step 2 Data visualization
df.plot.scatter("YearsExperience","Salary")

#step 3 splitting the variables into 2 parts
X = df[["YearsExperience"]]
Y = df["Salary"]

X1=np.sqrt(X)
X2=np.log(X)
X3=X**2

#step 4 model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X1,Y)

LR.intercept_
LR.coef_
t1 = np.array([[1.5],[5.9]])
LR.predict(t1)

#step5 predicted values
Y_pred = LR.predict(X1)
import matplotlib.pyplot as plt
plt.scatter(X1,Y,color="red")
plt.plot(X1["YearsExperience"],Y_pred,color="black")
plt.show()

#step 6 metrics
from sklearn.metrics import mean_squared_error
mse =  mean_squared_error(Y,Y_pred)
print("rmse",np.sqrt(mse))

#RMSE values after doing transformations
#rmse for X and Y is 5592.04
#rmse for X1 and Y is 7080.09
#rmse for X2 and Y is 10302.89
#rmse for X3 and Y is 7843.47
#from the above the transformation X is having low Rmse