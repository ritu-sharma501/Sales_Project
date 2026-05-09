import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample Sales Data
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales': [100, 120, 130, 150, 170, 180, 200, 210, 230, 250]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Labels
X = df[['Month']]
y = df['Sales']

# Train Model
model = LinearRegression()
model.fit(X, y)

# Predict Future Sales
future_months = [[11], [12], [13]]
predictions = model.predict(future_months)

# Print Predictions
print("Future Sales Prediction:")
for i, pred in enumerate(predictions, start=11):
    print(f"Month {i}: {pred:.2f}")

# Plot Graph
plt.scatter(df['Month'], df['Sales'], color='blue', label='Actual Sales')
plt.plot(df['Month'], model.predict(X), color='red', label='Prediction Line')

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecasting System")
plt.legend()

plt.show()