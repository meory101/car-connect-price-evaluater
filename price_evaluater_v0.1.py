import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Sample dataset (Replace with actual data)
data = pd.DataFrame({
    'brand': ['Toyota', 'BMW', 'Honda', 'Ford', 'Toyota'],
    'model': ['Corolla', 'X5', 'Civic', 'Focus', 'Camry'],
    'year': [2015, 2018, 2017, 2016, 2019],
    'mileage': [60000, 40000, 50000, 70000, 30000],
    'color': ['Red', 'Black', 'White', 'Blue', 'Gray'],
    'gear': ['Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic'],
    'fuel': ['Gasoline', 'Diesel', 'Gasoline', 'Gasoline', 'Hybrid'],
    'owners': [1, 2, 1, 3, 1],
    'price': [12000, 30000, 15000, 10000, 22000]
})

label_encoders = {}
categorical_columns = ['brand', 'model', 'color', 'gear', 'fuel']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

X = data.drop(columns=['price'])
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R^2 Score: {r2}')
