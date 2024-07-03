import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

data=pd.read_csv("suv_data (1).csv")
# print(data.head(5))

data.drop(['User ID','Gender'],inplace=True,axis=1)

X=data.drop('Purchased',axis=1)
y=data['Purchased']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#preprocessing
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#svm
from sklearn.svm import SVC
svc=SVC(kernel='linear',random_state=0)
svc.fit(X_train,y_train)

predict=svc.predict(X)
print("predict", predict)

from sklearn.metrics import accuracy_score
score=accuracy_score(y,predict)
print("score",score)

from matplotlib.colors import ListedColormap
X_set,y_set=X_train,y_train
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max()+1,step=.01),
                  np.arange(start=X_set[:,0].min()-1,
                            stop=X_set[:,0].max()+1,step=.01)         )

plt.contour(X1,X2,svc.predict(
    np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,cmap=ListedColormap('red','green'))

plt.xlim(X1.min(),X2.max())
plt.ylim(X1.min(),X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],
                c=ListedColormap(('red','green'))(i),label=j)
    
plt.title("Gauusian NB Algorithm salaried SUV data")
plt.xlabel("salaried data for suv ")
plt.ylabel("purchased suv car")
plt.show()
