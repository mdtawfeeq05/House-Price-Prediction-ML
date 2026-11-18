import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Step 1: Create sample housing data
data = {
    "area": [800, 950, 1100, 1250, 1400,
             1550, 1700, 1850, 2000, 2150,
             2300, 2450, 2600, 2750, 2900,
             3050, 3200, 3350, 3500, 3650],

    "bedrooms": [2, 2, 2, 2, 3,
                 3, 3, 3, 3, 4,
                 4, 4, 4, 4, 5,
                 5, 5, 5, 5, 5],

    "age": [15, 12, 10, 9, 8,
            7, 6, 5, 4, 3,
            8, 7, 6, 5, 4,
            3, 2, 6, 4, 1]
}

df = pd.DataFrame(data)

# Prices in lakh rupees (synthetic)
df["price"] = (
    df["area"] * 0.04
    + df["bedrooms"] * 2
    - df["age"] * 0.5
    + np.array([
        2, -1, 3, -2, 1,
        0, -3, 2, -1, 3,
        -2, 1, 0, -3, 2,
        -1, 3, -2, 1, 0
    ])
)

# Step 2: Features and target
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")

# Step 5: Test predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("MAE:", round(mae, 2), "lakh")
print("MSE:", round(mse, 2), "lakh squared")
print("RMSE:", round(rmse, 2), "lakh")
print("R2 Score:", round(r2, 3))

# Step 7: Predict a new house
new_house = pd.DataFrame({
    "area": [1750],
    "bedrooms": [3],
    "age": [4]
})

prediction = model.predict(new_house)

print(
    "Predicted House Price: ₹",
    round(prediction[0], 2),
    "lakh"
)
