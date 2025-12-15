import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.calibration import calibration_curve

# ==========================================
# STEP 1: LOAD & PREPARE DATA
# ==========================================
print("Loading data...")
df = pd.read_csv('./dataset/Clean_Dataset.csv')

# We need to create a "Ground Truth" for the model.
# Logic: We calculate the average price for every specific flight route per day.
print("Engineering features...")
trend = df.groupby(['source_city', 'destination_city', 'airline', 'class', 'days_left'])['price'].mean().reset_index()

# Create the Target: Compare "Price Today" vs "Price 7 Days Later"
future_gap = 7
trend['join_key'] = trend['days_left'] - future_gap

# Self-merge to bring future prices into the current row
merged = pd.merge(
    trend, trend,
    left_on=['source_city', 'destination_city', 'airline', 'class', 'join_key'],
    right_on=['source_city', 'destination_city', 'airline', 'class', 'days_left'],
    suffixes=('', '_future')
)

# Target = 1 if Future Price is > 5% higher than Today's Price (Buy Signal)
merged['target_buy_now'] = (merged['price_future'] > merged['price'] * 1.05).astype(int)

# ==========================================
# STEP 2: TRAIN THE MODEL (To get y_test, y_pred)
# ==========================================
print("Training model...")
features = ['days_left', 'class', 'airline', 'source_city', 'destination_city']
X = pd.get_dummies(merged[features], drop_first=True)
y = merged['target_buy_now']

# Split the data (This is where y_test comes from)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Generate Predictions (This is where y_pred and y_prob come from)
y_pred = clf.predict(X_test)              # The hard decision (0 or 1)
y_prob = clf.predict_proba(X_test)[:, 1]  # The probability score (0.0 to 1.0)

# ==========================================
# STEP 3: EVALUATE PERFORMANCE
# ==========================================
print("\n--- RESULTS ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==========================================
# STEP 4: VISUALIZE
# ==========================================
plt.figure(figsize=(15, 5))

# Plot 1: Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix\n(Did we buy at the right time?)')
plt.xlabel('Predicted (1=Buy, 0=Wait)')
plt.ylabel('Actual (1=Price Rose, 0=Flat/Drop)')

# Plot 2: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, color='orange', lw=2, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve\n(Discrimination Ability)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Calibration Curve
prob_true, prob_pred_bins = calibration_curve(y_test, y_prob, n_bins=10)
plt.subplot(1, 3, 3)
plt.plot(prob_pred_bins, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
plt.title('Calibration Curve\n(Can we trust the % confidence?)')
plt.xlabel('Predicted Probability')
plt.ylabel('Actual Fraction of Price Hikes')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print("Done! Charts generated.")