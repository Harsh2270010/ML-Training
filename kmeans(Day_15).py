import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data=pd.read_csv("driver-data.csv")
#print(data.head)


X=data.iloc[:,[1,2]].values

from sklearn.cluster import KMeans
coll=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmeans.fit(X)
    
coll.append(kmeans.inertia_)
print("coll :",coll)

kmeans=KMeans(n_clusters=10,init="k-means++",random_state=0)
y_kmeans=kmeans.fit_predict(X)


plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='r',label="cluster 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='b',label="cluster 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='g',label="cluster 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='m',label="cluster 4")
plt.legend()
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=70,c="y")
plt.show()
