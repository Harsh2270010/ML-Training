import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 
import seaborn as sns 
from sklearn.neighbors import KNeighborsClassifier

data=pd.read_csv("apples_and_oranges_Classification.csv")
print(data)
x=data.iloc[:,0:2].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_test,y_train=train_test_split(x,y,test_size=3,random_state=0)
print(x_test)
knn=KNeighborsClassifier()
knn.fit(x,y)
predict =knn.predict([[20,2.7]])
#data visulistion
xPlot=data.Weight
yPlot=data.Size
sns.scatterplot(x=xPlot,y=yPlot,hue=data.Class)
plt.xlabel("Weight")
plt.xlabel("size")
plt.show()
