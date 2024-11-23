import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


Load the House Prices Dataset:


url = '(link unavailable)'
df = pd.read_csv(url)


Step 2: Preprocess the data


# Drop unnecessary columns
df.drop('ocean_proximity', axis=1, inplace=True)

# Convert categorical variables to numerical variables
df['housing_median_age'] = pd.cut(df['housing_median_age'], bins=[0, 10, 20, 30, 40, 50], labels=[1, 2, 3, 4, 5, 6])

# Scale numerical variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']] = scaler.fit_transform(df[['housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']])

# Split data into features (X) and target (y)
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


Step 3: Create and train the model


# Create the model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))


Step 4: Evaluate the model


# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate MAE and MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Print MAE and MSE
print(f'MAE: {mae:.2f}')
print(f'MSE: {mse:.2f}')

# Plot predictions
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()
