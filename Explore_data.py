import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# 1. Load Dataset
df = pd.read_csv("data.csv")

# 2. Initial Cleanup
df.drop(columns=['Unnamed: 32', 'id'], inplace=True)
df['target'] = df['diagnosis'].map({'M': 1, 'B': 0})
df.drop(columns='diagnosis', inplace=True)

# 3. Basic Info
print("🔍 Shape:", df.shape)
print("🧾 Data Types:\n", df.dtypes)
print("🧼 Missing Values:\n", df.isnull().sum())
print("📊 Class Distribution:\n", df['target'].value_counts())

# 4. Summary Stats
print("\n📈 Summary Stats:")
print(df.describe())

# 5. Visualizations BEFORE Outlier Removal
sns.countplot(x='target', data=df)
plt.title("Target Class Distribution (1 = Malignant, 0 = Benign)")
plt.savefig("class_distribution.png")
plt.close()

plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("📌 Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

df.hist(figsize=(18, 14), bins=20)
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.savefig("feature_histograms.png")
plt.close()

selected_cols = ['radius_mean', 'texture_mean', 'area_mean']
for col in selected_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot (Before) - {col}")
    plt.savefig(f"boxplot_before_{col}.png")
    plt.close()

# 6. Feature + Target Split
X = df.drop(columns='target')
y = df['target']

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --- Outlier Removal Methods ---

# Method 1: Ultra-strict IQR (zero tolerance, multiplier 0.5)
def remove_outliers_iqr_strict(df, iqr_multiplier=0.5):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = (df < (Q1 - iqr_multiplier * IQR)) | (df > (Q3 + iqr_multiplier * IQR))
    filtered_df = df[~outlier_mask.any(axis=1)]  # zero outlier tolerance
    return filtered_df

# Method 2: Isolation Forest (ML-based outlier detection)
def remove_outliers_isolation_forest(X_train, contamination=0.05):
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X_train)
    mask = preds == 1  # 1 = normal, -1 = outlier
    return X_train[mask], mask

# Choose your outlier removal method:

# --- Uncomment one of these ---

# 1. Ultra-strict IQR removal
X_train_no_outliers = remove_outliers_iqr_strict(pd.DataFrame(X_train, columns=X.columns))
y_train_no_outliers = y_train.loc[X_train_no_outliers.index]

# 2. Isolation Forest removal
# X_train_no_outliers, mask = remove_outliers_isolation_forest(pd.DataFrame(X_train, columns=X.columns))
# y_train_no_outliers = y_train.loc[mask]

print(f"🧹 Outlier Removal dropped {len(X_train) - len(X_train_no_outliers)} rows from training data.")
print("📊 New class distribution after outlier removal:")
print(y_train_no_outliers.value_counts())

# 9. Visualize boxplots AFTER outlier removal
for col in selected_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=X_train_no_outliers[col])
    plt.title(f"Boxplot (After) - {col}")
    plt.savefig(f"boxplot_after_{col}.png")
    plt.close()

# 10. Feature Scaling with RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_no_outliers)
X_test_scaled = scaler.transform(X_test)

# 11. Save cleaned data for modeling
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("X_test_scaled.csv", index=False)
pd.DataFrame(y_train_no_outliers).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("y_test.csv", index=False)

print("\n✅ EDA + Outlier Removal complete. Scaled data and new plots saved.")
