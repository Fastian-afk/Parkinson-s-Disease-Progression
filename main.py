import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("parkinsons_updrs.data")

# Initial exploration
print("Shape:", df.shape)
print("Columns:\n", df.columns)
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical Summary:\n", df.describe())

# -------------------- Visualizations --------------------

# 1. Distribution of UPDRS Scores
plt.figure(figsize=(12, 5))
sns.histplot(df["total_UPDRS"], kde=True, color='red', bins=30)
plt.title("Distribution of total_UPDRS")
plt.xlabel("total_UPDRS Score")
plt.ylabel("Count")
plt.grid(True)
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

# 3. Boxplots of Key Features vs UPDRS
features = ["Jitter(%)", "Shimmer", "HNR", "PPE"]
for feat in features:
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x=feat, y="total_UPDRS", hue="sex")
    plt.title(f"{feat} vs total_UPDRS")
    plt.show()

# 4. Interactive Trends with Time
fig = px.line(df[df["subject#"] == 1], x="test_time", y="total_UPDRS",
              title="UPDRS Progression Over Time (Subject 1)")
fig.show()

# -------------------- Feature Engineering --------------------

# Drop identifiers
df_clean = df.drop(columns=["subject#", "sex", "test_time"])

# Define X and y
X = df_clean.drop(columns=["motor_UPDRS", "total_UPDRS"])
y = df_clean["total_UPDRS"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- Model Training & Evaluation --------------------

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n{name}:")
    print("MSE:", mse)
    print("R² Score:", r2)

    # Plot predictions vs actual
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, preds, alpha=0.6, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f"{name}: Actual vs Predicted")
    plt.xlabel("Actual total_UPDRS")
    plt.ylabel("Predicted total_UPDRS")
    plt.grid(True)
    plt.show()

# -------------------- Future Prediction (Example) --------------------

# Predict future progression for a new sample (you can automate this)
sample = X_test.iloc[0:1]
scaled_sample = scaler.transform(sample)
future_pred = models["XGBoost"].predict(scaled_sample)
print("\nFuture Prediction for a sample:")
print("Predicted total_UPDRS:", future_pred[0])