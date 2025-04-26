import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("All Countries and Economies GDP (US) 1960-2023.csv")

# Extract rows for India
india_data = df[df['Country Name'] == 'Finland']

# Drop non-year columns (keeping only years)
gdp_data = india_data.loc[:, '1960':'2023'].T  # Transpose years to rows
gdp_data.columns = ['GDP']
gdp_data['Year'] = gdp_data.index.astype(int)

# Drop missing values if any
gdp_data = gdp_data.dropna()

# Prepare X and y
X = gdp_data[['Year']]  # feature
y = gdp_data['GDP']     # target

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict GDP values
y_pred = model.predict(X)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual GDP')
plt.plot(X, y_pred, color='red', label='Predicted GDP (Linear Fit)')
plt.xlabel('Year')
plt.ylabel('GDP (US Dollars)')
plt.title('Finland GDP over Time - Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

# Print model coefficients
print(f"Coefficient (Growth Rate): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
