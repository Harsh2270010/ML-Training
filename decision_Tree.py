import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv("Decisiondata.csv")
X = data.iloc[:,1:2]
y = data.iloc[:,2]
print(y)

from sklearn.tree import DecisionTreeRegressor

des = DecisionTreeRegressor()
des.fit(X,y)
plt.scatter(X,y)
plt.plot(X, des.predict(X), color="red")
plt.title("Descision Tree Regression")
plt.xlabel("exprerience ")
plt.ylabel("salaried data")
plt.show()
