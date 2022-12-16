# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 17:33:37 2022

@author: HP
"""

#####hierarchial clustering
import pandas as pd
df = pd.read_excel("D:\\Assignments DS\\clustering\\EastWestAirlines.xlsx",sheet_name = "data", index_col = 0)
df.shape

X = df.iloc[:,1:]
X.shape

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scale = ss.fit_transform(X)

#dendrogram construction
import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(X_scale,method="single"))
dend = shc.dendrogram(shc.linkage(X_scale,method='complete'))

from sklearn.cluster import AgglomerativeClustering
#cluster = AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="single")
cluster = AgglomerativeClustering(n_clusters=2,affinity="euclidean",linkage="complete")
Y = cluster.fit_predict(X_scale)
Y_new = pd.DataFrame(Y)
Y_new.value_counts()
pd.concat([df,Y_new],axis=1)


##############################################################################
#KMeans
X = df.iloc[:,1:]
X.shape

from sklearn.cluster import KMeans
kint_list=[]
for i in range(2,20):
    km = KMeans(n_clusters=i,n_init=20)
    km.fit_predict(X_scale)
    kint_list.append(km.inertia_)
print(kint_list)

km = KMeans(n_clusters=6,n_init=20)
km.fit(X_scale)
y = km.predict(X_scale)
Y = pd.DataFrame(y)
Y.value_counts()

import matplotlib.pyplot as plt
plt.scatter(range(2,20),kint_list)
plt.plot(range(2,20),kint_list)
plt.show()

#################################################################################
#DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=4, min_samples=3) #eps = epsalon = radius
dbscan.fit(X_scale)
y=dbscan.labels_
Y=pd.DataFrame(y)
Y.value_counts()


###########conclusion
#according to agglomerative clustering , kmeans and dbscan we have got good no of clusters with good number of samples using kmeans and that is with 6 clusters