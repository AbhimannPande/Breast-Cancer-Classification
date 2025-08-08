import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import joblib

# Load raw preprocessed train/test data (unscaled)
X_train = pd.read_csv("X_train_scaled.csv")
X_test = pd.read_csv("X_test_scaled.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# 1. Scale features with RobustScaler (fit on train, transform both)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train logistic regression model on scaled data
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# 3. Predict probabilities and default classes
y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred_default = model.predict(X_test_scaled)

# 4. Evaluate default threshold=0.5
print("Confusion Matrix (threshold=0.5):")
print(confusion_matrix(y_test, y_pred_default))
print("Precision:", precision_score(y_test, y_pred_default))
print("Recall:", recall_score(y_test, y_pred_default))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# 5. Tune threshold to balance precision and recall
thresholds = np.arange(0, 1, 0.01)
precisions, recalls = [], []
for t in thresholds:
    preds = (y_proba >= t).astype(int)
    precisions.append(precision_score(y_test, preds))
    recalls.append(recall_score(y_test, preds))

# Plot precision-recall vs threshold
plt.plot(thresholds, precisions, label='Precision')
plt.plot(thresholds, recalls, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall for Different Thresholds')
plt.legend()
plt.savefig("precision_recall_threshold.png")
plt.close()

# 6. Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.3f})'.format(roc_auc_score(y_test, y_proba)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

# 7. Sigmoid function plot (example on first feature coef)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

coef = model.coef_[0][0]
intercept = model.intercept_[0]
x_vals = np.linspace(-10, 10, 200)
y_vals = sigmoid(coef * x_vals + intercept)

plt.plot(x_vals, y_vals)
plt.title('Sigmoid Function (Example)')
plt.xlabel('Feature 1 scaled value')
plt.ylabel('Predicted Probability')
plt.savefig("sigmoid_curve.png")
plt.close()

# 8. Custom threshold evaluation
custom_threshold = 0.4  # adjust as needed
y_pred_custom = (y_proba >= custom_threshold).astype(int)

print(f"\nConfusion Matrix (threshold={custom_threshold}):")
print(confusion_matrix(y_test, y_pred_custom))
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

# 9. Save model and scaler for deployment or later use
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Logistic regression trained, evaluated, scaler saved, and plots generated.")
