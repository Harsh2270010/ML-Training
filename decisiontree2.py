import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Salary Data.csv")
d=data.dropna(inplace=True)
# print(d)
# print(data)

X=data.iloc[:,[0,5]].values
# print(X)
y=data.iloc[:,-1].values
# print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor(n_estimators=10)
forest.fit(X_train,y_train)
predict=forest.predict([[24,1]])
# predict=forest.predict(X)
print(predict)