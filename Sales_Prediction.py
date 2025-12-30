import numpy as np
from sklearn.linear_model import LinearRegression

# Example trained model (dummy training for demonstration)
# In real projects, this model is trained using historical data
X_train = np.array([
    [230, 37, 69],
    [44, 39, 45],
    [17, 45, 69],
    [151, 41, 58],
    [180, 10, 40]
])

y_train = np.array([22.1, 10.4, 9.3, 18.5, 17.9])

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- USER INPUT ----------------
tv = float(input("Enter TV Advertising Spend: "))
radio = float(input("Enter Radio Advertising Spend: "))
newspaper = float(input("Enter Newspaper Advertising Spend: "))

# Prediction
input_data = np.array([[tv, radio, newspaper]])
predicted_sales = model.predict(input_data)

print("\n Predicted Sales:", round(predicted_sales[0], 2))
