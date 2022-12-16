# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:12:28 2022

@author: HP
"""

import pandas as pd
df = pd.read_csv("D:\\Assignments DS\\random forest\\Fraud_check.csv")
new_Y=[]
for i in df["Taxable.Income"]:
    if i<=30000:
        new_Y.append("Risky")
    else:
        new_Y.append("Good")

new_Y = pd.DataFrame(new_Y)
new_df = pd.concat([df,new_Y],axis=1)
#new_df = new_df.drop(new_df["Taxable.Income"],axis=1,inplace=True)
new_df.drop(new_df.columns[[2]],axis=1,inplace=True)
new_df

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
new_df["Undergrad"] = LE.fit_transform(new_df["Undergrad"])
new_df["Marital.Status"] = LE.fit_transform(new_df["Marital.Status"])
new_df["Urban"] = LE.fit_transform(new_df["Urban"])
new_df

X = new_df.iloc[:,0:4]
Y = new_Y

from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X = SS.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3)


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=(4)) 
dt.fit(X_train, Y_train)

Y_pred_train = dt.predict(X_train) 
Y_pred_test = dt.predict(X_test) 

from sklearn.metrics import accuracy_score
print("Training Accuracy: ",accuracy_score(Y_train,Y_pred_train).round(2))
print("Test Accuracy: ",accuracy_score(Y_test,Y_pred_test).round(2))

from sklearn.ensemble import RandomForestClassifier
RFR = RandomForestClassifier(n_estimators=150,max_samples=0.8,max_features=0.4,max_depth=(5)) 
RFR.fit(X_train, Y_train)

Y_pred_train = RFR.predict(X_train) 
Y_pred_test = RFR.predict(X_test) 

from sklearn.metrics import accuracy_score
print("Training accuracy: ",accuracy_score(Y_train,Y_pred_train).round(2))
print("Test accuracy: ",accuracy_score(Y_test,Y_pred_test).round(2))
