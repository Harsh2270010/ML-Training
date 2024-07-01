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

from sklearn.naive_bayes import GaussianNB

gaus=GaussianNB()
gaus.fit(X_train,y_train)

#predict
predict=gaus.predict([[34,67000]])
print(predict)

#matrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

score=accuracy_score(y_test ,predict)
print("Score::",score) #score::0.891666666667

con=confusion_matrix(y_test,predict)
print(con)

report=classification_report(y_test,predict)
print(report)

from matplotlib.colors import ListedColormap

x_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=.01),np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=.01))


plt.contour(X1,X2,gaus.predict( np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap('red','green'))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X1.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1], c=ListedColormap(('red','green'))(i),label=j)

plt.title("Gauusian NB Algorithm salaried SUV data")
plt.xlabel("salaried data for suv ")
plt.ylabel("purchased suv car")
plt.show()