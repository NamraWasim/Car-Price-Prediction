import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Get the absolute path of the data file
data_path = os.path.join(os.path.dirname(__file__), '../data/car_prices.csv')

# Load the dataset
df = pd.read_csv(data_path)

# Display the first few rows of the dataframe
print(df.head())

# Convert categorical columns to numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'])

# Define features (X) and target (y)
X = df.drop(columns=['name', 'selling_price'])
y = df['selling_price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plotting the actual vs predicted prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()
