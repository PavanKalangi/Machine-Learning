# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 23:19:45 2022

@author: HP
"""
#1) Delivery_time -> Predict delivery time using sorting time 
#step 1
import pandas as pd 
import numpy as np
df = pd.read_csv("D:\\Assignments DS\\4\\delivery_time.csv")
df.head

#step 2: Data visualisation
df.plot.scatter(x="Sorting Time" , y="Delivery Time")
df.corr()

#step 3: split the variables into 2 parts
X=df[["Sorting Time"]]
Y=df["Delivery Time"]

#Transformations
#X1=np.sqrt(X["Sorting Time"])
X1=np.sqrt(X)
X2=np.log(X)
X3=X**2

#step 4 : model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X3,Y)

LR.intercept_
LR.coef_

t1 = np.array([[10],[43],[24]])
LR.predict(t1)

#step 5 predicted values
Y_pred = LR.predict(X3)

import matplotlib.pyplot as plt
plt.scatter(X3,Y,color="red")
plt.plot(X3["Sorting Time"],Y_pred,color="black")
plt.show()

#step 6 metrics
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y,Y_pred)
print("rmse",np.sqrt(mse))

#RMSE values after doing transformations
#rmse for X and Y is 2.791
#rmse for X1 and Y is 2.731
#rmse for X2 and Y is 2.733
#rmse for X3 and Y is 3.011
#from the above the transformation X1 is having low Rmse


