import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import re

# Function to validate IP address format
def is_valid_ip(ip):
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    return isinstance(ip, str) and bool(re.match(ip_pattern, ip))

# Load the dataset
file_path = "attack.csv"  # Update if needed
df = pd.read_csv(file_path)

# Display dataset shape and columns
print("Initial dataset shape:", df.shape)
print("Columns in dataset:", df.columns)

# Check for missing required columns
required_columns = ["ip_src", "ip_dst", "snort_alert"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns in dataset: {missing_columns}")
    exit()

# Remove rows where key features are missing
df = df.dropna(subset=["ip_src", "ip_dst", "snort_alert"])
print("Dataset shape after removing NaN:", df.shape)

# Define features and target (including IP subnet information)
df["ip_src_subnet"] = df["ip_src"].apply(lambda x: ".".join(x.split(".")[:3]) if is_valid_ip(x) else "invalid")
df["ip_dst_subnet"] = df["ip_dst"].apply(lambda x: ".".join(x.split(".")[:3]) if is_valid_ip(x) else "invalid")

# Filter out rows with invalid IPs
df = df[df["ip_src_subnet"] != "invalid"]
df = df[df["ip_dst_subnet"] != "invalid"]
print("Dataset shape after filtering invalid IPs:", df.shape)

features = ["ip_src", "ip_dst", "ip_src_subnet", "ip_dst_subnet"]
target = "snort_alert"

# Encode categorical features
encoder = LabelEncoder()
for col in features:
    df[col] = encoder.fit_transform(df[col])

# Define X and y
X = df[features]
y = df[target]

# Check if data exists after processing
if X.empty or y.empty:
    print("Error: No valid data available for training.")
    exit()

print("Final dataset shape:", X.shape)

# Balance classes using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Dataset shape after SMOTE:", X_resampled.shape)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestClassifier(class_weight="balanced", random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Train a Random Forest model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions
df["predicted_alert"] = best_model.predict(X)  # Predict on the full dataset

# Evaluate the model
accuracy = accuracy_score(y_test, best_model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, best_model.predict(X_test)))

# Reload the dataset for the final plot (to use original IP addresses)
df = pd.read_csv(file_path)

# Replace with actual column name
destination_col = "ip_dst"  # Destination IP column

# Filter for valid IP addresses in destination column
df = df[df[destination_col].apply(is_valid_ip)]

# Count attacks by destination IP (top 10)
destination_counts = df[destination_col].value_counts().head(10)

# Plot top attacked destinations
plt.figure(figsize=(10, 5))
destination_counts.plot(kind='bar', color='red')
plt.xlabel("Destination IP")
plt.ylabel("Number of Attacks")
plt.title("Attacked Destinations")
plt.xticks(rotation=45)
plt.show()
