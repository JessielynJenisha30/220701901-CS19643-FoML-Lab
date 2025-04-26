mport pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load dataset
df = pd.read_csv("All Countries and Economies GDP (US) 1960-2023.csv")

# Filter for India and transpose GDP values (columns: 1960-2023)
india_data = df[df['Country Name'] == 'Finland']
gdp_data = india_data.loc[:, '1960':'2023'].T
gdp_data.columns = ['GDP']
gdp_data['Year'] = gdp_data.index.astype(int)

# Remove missing values
gdp_data.dropna(inplace=True)

# Define X and y
X = gdp_data[['Year']]
y = gdp_data['GDP']

# Polynomial Features (degree 2 or 3)
degree = 3  # You can try 2 or 4 as well
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# Train the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual GDP')
plt.plot(X, y_pred, color='green', linewidth=2, label=f'Polynomial Regression (Degree {degree})')
plt.xlabel('Year')
plt.ylabel('GDP (US Dollars)')
plt.title(f'Finland GDP over Time - Polynomial Regression (Degree {degree})')
plt.legend()
plt.grid(True)
plt.show()

# Print coefficients
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
