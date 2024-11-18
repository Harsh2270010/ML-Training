import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'age': [25, 30, 45, 50, 23, 34, 44, 29, 39, 42],
    'salary': [50000, 60000, 80000, 100000, 45000, 52000, 71000, 49000, 64000, 78000],
    'education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'Bachelors', 'PhD', 'Bachelors', 'Masters', 'PhD'],
    'target': [1, 0, 1, 0, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print(df)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
X = df.drop('target', axis=1)
y = df['target']

# Define numerical and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define preprocessing for numerical data (scaling and imputation)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define preprocessing for categorical data (imputation and one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

# Display the preprocessed data
print("Preprocessed Training Data:")
print(X_train)

print("\nPreprocessed Test Data:")
print(X_test)
