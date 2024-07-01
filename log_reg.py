import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("suv_data (1).csv")
print(data.head(10))

#changing the original data
edata=data.drop(["User ID","Gender"],inplace=True,axis=1)
# print(edata)

#independent data
X=data.drop(["Purchased"],axis=1)
print(X)

#dependent data
y=data['Purchased']
print(y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit(X_test)

from sklearn.linear_model import LinearRegression
log=LinearRegression()
log.fit(X_train,y_train)

predict=log.predict([[45,34000]])
print("Suv car purchesed or not::",predict)

