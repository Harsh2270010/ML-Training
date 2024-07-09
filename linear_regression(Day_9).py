import pandas as pd 
import numpy as np

#data =pd.read_csv("tvmarketing.csv")
#print(data.head(10))
# x=data.iloc[:,:-1]
# y=data.iloc[:,1]

# from sklearn.model_selection import train_test_split

# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=3,random_state=0)

# from sklearn.linear_model import LinearRegression

# leg=LinearRegression()
# leg.fit(x_train,y_train)
# predict=leg.predict(x_train)
# print("prediction::",predict)

# #data visualization
# import matplotlib.pyplot as plt 
# plt.scatter(x_train,y_train,color='blue')
# plt.plot(x_train,predict,color='red')
# plt.title("TV Market Data")
# plt.xlabel("No. of Product")
# plt.ylabel("Sales data")
# plt.show()

#salary prediction

data=pd.read_csv("salary_prediction_data.csv")
print(data.head(10))

salaryData=data.iloc[:,[4,5]]
multiDimARRAY=np.array(salaryData)['Experience']
# data.isnull.sum
dropNULLSalary=salaryData.dropna()

x=dropNULLSalary.iloc[:,0:1].values
y=dropNULLSalary.salary.values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=0)
from sklearn.linear_model import LinearRegression
leg=LinearRegression()
leg.fit(x,y)
predict=leg.predict([[2.3]])
print(predict)


#data visualization

from sklearn.linear_model import LinearRegression

