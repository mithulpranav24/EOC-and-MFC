import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = "dataset.csv"
data = pd.read_csv(data_path)

# Debug: Inspect the dataset
print(data.info())
print(data.head())

# Drop unnecessary columns if they exist
data = data.drop(columns=[col for col in ['Unnamed: 0', 'Time'] if col in data.columns])

# Handle missing values: fill numerical columns with mean and categorical columns with mode
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

if not categorical_cols.empty:
    mode_values = {col: data[col].mode().iloc[0] for col in categorical_cols if not data[col].isnull().all()}
    data = data.fillna(value=mode_values)

# Encode categorical variables
for col in categorical_cols:
    if not data[col].isnull().all():
        data[col] = LabelEncoder().fit_transform(data[col])

# Ensure the target column exists
if 'snort_alert' not in data.columns:
    raise ValueError("The target column 'snort_alert' is missing in the dataset.")

# Identify features and target variable
X = data.drop(columns=['snort_alert'], errors='ignore')  # Features
y = data['snort_alert']  # Target

# Drop columns with all NaN values
X = X.dropna(axis=1, how="all")

# Drop constant columns (zero variance)
zero_variance_cols = X.loc[:, X.nunique() <= 1].columns
X = X.drop(columns=zero_variance_cols)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Debug: Check for NaN or infinite values before scaling
print("NaN in X_train:", np.isnan(X_train).sum().sum())
print("NaN in X_test:", np.isnan(X_test).sum().sum())
print("Infinite in X_train:", np.isinf(X_train).sum().sum())
print("Infinite in X_test:", np.isinf(X_test).sum().sum())

# Fill remaining NaN values with column means (as a last resort)
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_train.mean())  # Use train mean to avoid data leakage

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)



# Visualize the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1], cbar=False, annot_kws={"size": 20})

# Labels with increased font size
plt.xlabel("Predicted Labels", fontsize=20)
plt.ylabel("True Labels", fontsize=20)

# X and Y ticks
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig("BC-RandomForestConfusion.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()