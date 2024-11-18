import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
data=pd.read_csv("iris.csv")
y=data.iloc[:,-1]
y=y.map({0:'iris.setosa',1:'iris.versicolor',2:'iris.virgincia'})
manova=MANOVA.from_formula('sepal_length+sepal_width+petal_length+petal_width~Species',data)
result=manova.mv_test()
print(result)
