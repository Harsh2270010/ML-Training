# Hierarchical Clustring
#Agglomerative Clustring

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data=pd.read_csv("driver-data.csv")
#print(data.head())

X=data.iloc[:,[1,2]].values

# import scipy.cluster.hierarchy as sch 

# dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))

from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=5,linkage='ward')
y_hc=hc.fit_predict(X)
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='r',label='cluster-1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='b',label='cluster-2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='g',label='cluster-3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='m',label='cluster-4')
plt.title("Agglomerative Clustering")
plt.xlabel("mean_dist_day")
plt.ylabel("mean_over_speed_perc")
plt.show()



