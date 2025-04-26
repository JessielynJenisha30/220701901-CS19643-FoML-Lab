import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv("All Countries and Economies GDP (US) 1960-2023.csv")
india = df[df['Country Name'] == 'India'].loc[:, '1960':'2023'].T
india.columns = ['GDP']
india['Year'] = india.index.astype(int)
india.dropna(inplace=True)

# Create binary label: 1 if GDP above average, else 0
avg_gdp = india['GDP'].mean()
india['Label'] = (india['GDP'] > avg_gdp).astype(int)

# Prepare data
X = india[['Year']]
y = india['Label']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
