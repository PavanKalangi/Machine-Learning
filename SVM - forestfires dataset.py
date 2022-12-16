# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:40:41 2022

@author: HP
"""

import pandas as pd
df = pd.read_csv("D:\\Assignments DS\\SVM\\forestfires.csv")



from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["month"] = LE.fit_transform(df["month"])
df["day"] = LE.fit_transform(df["day"])

X=df.iloc[:,0:30]
Y=df["size_category"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=2.0)
#svm = SVC(kernel='linear',C=3.0)
#svm = SVC(kernel='poly',degree=3)
#svm = SVC(kernel='rbf', gamma=3)

svm.fit(X_train, Y_train)
Y_pred_train = svm.predict(X_train)
Y_pred_test  = svm.predict(X_test)

# import the metrics class
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test, Y_pred_test)
print(cm)

print("Training Accuracy :",accuracy_score(Y_train, Y_pred_train).round(2))
print("Testing Accuracy:",accuracy_score(Y_test, Y_pred_test).round(2))

#we are achieving very good accuracy with kernal as linear and cost parameter 2.0
#we are achieving training accuracy as 100% and test accuracy around 99%