import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,accuracy_score

# Step 1: Read the dataset and store it in pandas dataframe
data = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# Step 2: Check the structure of the dataset
data.info()
print(data.head())

# Step 3: Handling missing values
data = data.dropna()

# Step 4: Convert the datatypes only if necessary
# data['dateCrawled'] = pd.to_datetime(data['dateCrawled'])
# data['dateCreated'] = pd.to_datetime(data['dateCreated'])

# Step 5: Data Cleaning
columns_to_drop = ['name', 'seller_type', 'transmission', 'owner']
data = data.drop(columns=columns_to_drop)

# Step 6: Analyzing Selling Price Using Plots
# Histogram of Selling Price
plt.figure(figsize=(10, 6))
sns.histplot(data['selling_price'], bins=20, kde=True)
plt.xlabel('Price')
plt.ylabel('Count')
plt.title('Histogram of Selling Price')
plt.show()

# Boxplot of Selling Price by Vehicle Type
# plt.figure(figsize=(10, 6))
# #sns.boxplot(x='vehicleType', y='selling_price', data=data)
# plt.xlabel('Vehicle Type')
# plt.ylabel('Price')
# plt.title('Boxplot of Selling Price by Vehicle Type')
# plt.show()

# Scatter plot of Year of Registration vs. Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='yearOfRegistration', y='selling_price', data=data)
plt.xlabel('Year of Registration')
plt.ylabel('Price')
plt.title('Scatter plot of Year of Registration vs. Price')
plt.show()

# Step 7: Predicting Selling Price using Linear Regression
features = ['year', 'km_driven', 'fuel']
target = 'selling_price'
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
print("Y prediction :",y_pred)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')

#score=accuracy_score(y,y_pred)
#print(score)