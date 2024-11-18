import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("suv_data (1).csv")
print(data.head(10))

# Drop unnecessary columns
data.drop(['User ID', 'Gender'], inplace=True, axis=1)

# Independent data
X = data.drop(['Purchased'], axis=1)
print(X)

# Dependent data
y = data['Purchased']
print(y)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # Use transform instead of fit

# Model training
from sklearn.naive_bayes import GaussianNB
gaus = GaussianNB()
gaus.fit(X_train, y_train)

# Predicting on the test set
y_pred = gaus.predict(X_test)  # Predict on the test data

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

score = accuracy_score(y_test, y_pred)
print("Score::", score)  # Print accuracy score

con = confusion_matrix(y_test, y_pred)
print(con)

report = classification_report(y_test, y_pred)
print(report)

# Predict for a new sample (make sure to scale the input)
new_sample = sc.transform([[34, 67000]])  # Scale the new input
predict = gaus.predict(new_sample)
print("Prediction for new sample:", predict)

# Visualization
from matplotlib.colors import ListedColormap

x_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 1, stop=x_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=x_set[:, 1].min() - 1, stop=x_set[:, 1].max() + 1, step=0.01))

plt.contour(X1, X2, gaus.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
           alpha=0.75, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c=ListedColormap(('red', 'green'))(i), label=j)

plt.title("Gaussian NB Algorithm for SUV Purchase Prediction")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()