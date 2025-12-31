import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# ---------------- SAMPLE TRAINING DATA ----------------
# Features: [sepal_length, sepal_width, petal_length, petal_width]
X_train = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Setosa
    [4.9, 3.0, 1.4, 0.2],   # Setosa
    [7.0, 3.2, 4.7, 1.4],   # Versicolor
    [6.4, 3.2, 4.5, 1.5],   # Versicolor
    [6.3, 3.3, 6.0, 2.5],   # Virginica
    [5.8, 2.7, 5.1, 1.9]    # Virginica
])

y_train = np.array([
    "setosa",
    "setosa",
    "versicolor",
    "versicolor",
    "virginica",
    "virginica"
])

# ---------------- TRAIN MODEL ----------------
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ---------------- USER INPUT ----------------
sl = float(input("Enter Sepal Length: "))
sw = float(input("Enter Sepal Width: "))
pl = float(input("Enter Petal Length: "))
pw = float(input("Enter Petal Width: "))

# Prediction
input_data = np.array([[sl, sw, pl, pw]])
prediction = model.predict(input_data)

print("\n Predicted Iris Species:", prediction[0])
