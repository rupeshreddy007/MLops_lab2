import pandas as pd
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/winequality-red.csv", sep=';')

# 2. Features & target
X = df.drop("quality", axis=1)
y = df["quality"]

# 3. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Model 
model = LinearRegression()

# 6. Train
model.fit(X_train, y_train)

# 7. Predict
y_pred = model.predict(X_test)

# 8. Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 9. Save model
with open("output/model.pkl", "wb") as f:
    pickle.dump(model, f)

# 10. Save metrics
results = {"MSE": mse, "R2": r2}

with open("output/results.json", "w") as f:
    json.dump(results, f, indent=4)

# 11. Print metrics (IMPORTANT for GitHub Actions)
print(f"MSE: {mse}")
print(f"R2: {r2}")