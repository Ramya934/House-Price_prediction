# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Set pandas options to display all columns in one line
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) 

# Sample data 
data = {
    'square_footage': [1000, 1500, 2000, 2500, 3000, 1800, 1600, 2300, 1200, 1100],
    'num_bedrooms': [2, 3, 3, 4, 4, 3, 2, 4, 2, 1],
    'num_bathrooms': [1, 2, 2, 3, 3, 2, 1, 3, 1, 1],
    'age_of_house': [10, 5, 20, 15, 8, 12, 30, 7, 25, 40],
    'location_quality': [3, 4, 5, 3, 4, 3, 2, 5, 2, 1],  # Scale of 1 (worst) to 5 (best)
    'price': [200000, 250000, 300000, 350000, 400000, 280000, 220000, 340000, 180000, 160000]
}
df = pd.DataFrame(data)

# Display first few rows of the dataset
print("Data preview:\n", df.head())


# Separate features and target variable
X = df[['square_footage', 'num_bedrooms', 'num_bathrooms', 'age_of_house', 'location_quality']]
y = df['price']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train the model using Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Step 4: Predict and evaluate using Linear Regression
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print("Linear Regression Mean Squared Error:", mse_lin)

# Step 5: Train the model using Decision Tree Regression
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Step 6: Predict and evaluate using Decision Tree Regression
y_pred_tree = tree_reg.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print("Decision Tree Regression Mean Squared Error:", mse_tree)

# Choose the best model based on MSE
if mse_lin < mse_tree:
    print("Linear Regression performs better with MSE:", mse_lin)
else:
    print("Decision Tree Regression performs better with MSE:", mse_tree)
