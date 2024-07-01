import numpy as np 
import pandas as pd 
from statsmodels.multivariate.manova import MANOVA
data= pd.read_csv("Iris.csv")
df=pd.DataFrame(data=data)
y=data.iloc[:,-1]
y=y.map({0:'Iris-setosa',1:'Iris-vesicolor',2:'Iris-virginica'})
manova= MANOVA.from_formula('sepal_length+sepal_width+petal_length+petal_width',data)
result=manova.mv_test()
print(result)


# print (median)

from sklearn.preprocessing import StandardScaler

data=[(10,20),(20,30),(40,50)]
sc=StandardScaler()
model=sc.fit(data)
e=model.transform(data)
print(e)