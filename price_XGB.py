import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# ==================================== Load clean dataset ====================================
df = pd.read_csv("./dataset/clean_dataset.csv")
df = df[df["class"] == "Economy"].reset_index(drop=True)
df["price"] = df["price"].astype(float)
df["duration_hr"] = df["duration"].astype(float)
stop_map = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3
}
df["stop_count"] = df["stops"].map(stop_map).fillna(2).astype(int)
df["route"] = df["source_city"] + "_" + df["destination_city"]


# =============================== Encode categorical variables ===============================
cat_cols = [
    "airline",
    "flight",
    "source_city",
    "destination_city",
    "departure_time",
    "arrival_time",
    "route"
]

encoders = {}
for col in cat_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# ==================================== Feature selection ====================================
feature_cols = [
    "airline",
    "flight",
    "route",
    "departure_time",
    "arrival_time",
    "duration_hr",
    "stop_count",
    "days_left"
]

X = df[feature_cols]
# y = df["price"]
y = np.log1p(df["price"]) 


# ================================ Train-test split (time-aware) ================================
df_sorted = df.sort_values("days_left")

X_train, X_test, y_train, y_test = train_test_split(
    df_sorted[feature_cols],
    np.log1p(df_sorted["price"]),
    # df_sorted["price"],
    test_size=0.2,
    shuffle=False
)

# ===================================== Train XGBoost model =====================================
model = XGBRegressor(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=11,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric="rmse"
)

model.fit(X_train, y_train)

# ====================================== Feature importance ======================================
importances = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(feature_cols, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (XGBoost – Clean Dataset)")
plt.tight_layout()
plt.savefig('feature_importance_XGB_clean.png')

# ========================================= Evaluation =========================================

y_train_pred_log = model.predict(X_train)
y_train_pred = np.expm1(y_train_pred_log)
y_train_original = np.expm1(y_train)

train_rmse = mean_squared_error(y_train_original, y_train_pred, squared=False)
train_mae = mean_absolute_error(y_train_original, y_train_pred)

y_test_pred_log = model.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)
y_test_original = np.expm1(y_test)

test_rmse = mean_squared_error(y_test_original, y_test_pred, squared=False)
test_mae = mean_absolute_error(y_test_original, y_test_pred)

print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
print(f"Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}")