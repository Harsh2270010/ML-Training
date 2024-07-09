import numpy as np
import pandas as pd 

data=pd.read_csv("iris.csv")

x=data.iloc[:,:-1]
#print(x)

y=data.iloc[:,-1]
#print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=100,criterion="gini")
forest.fit(x_train,y_train)
#predict=forest.predict([[2.3,3.3,3.2,1.7]])
predict=forest.predict(x)
print(predict)


from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

score=accuracy_score(y,predict)
print("Accuracy Score ::",score)
con=confusion_matrix(y,predict)
print(con)
