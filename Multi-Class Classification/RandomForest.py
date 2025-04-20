import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------
# 1. Data Loading
# ---------------------------------------
def load_data(file_path):
    """Load dataset from CSV file."""
    data = pd.read_csv(file_path)
    return data

# ---------------------------------------
# 2. Data Preprocessing
# ---------------------------------------
def preprocess_data(data):
    """Clean and preprocess the dataset."""
    # Drop unnecessary columns if they exist
    columns_to_drop = ['Unnamed: 0', 'Time']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # Handle missing values
    numeric_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    # Fill missing values in numeric columns with mean
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    # Fill missing values in categorical columns with mode
    for col in categorical_cols:
        if not data[col].isnull().all():
            data[col] = data[col].fillna(data[col].mode()[0])

    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    return data, label_encoders, numeric_cols

# ---------------------------------------
# 3. Feature Engineering and Scaling
# ---------------------------------------
def prepare_features(data, numeric_cols, target_column='attack_type'):
    """Prepare features and target, and scale numeric features."""
    # Define features (X) and target (y)
    X = data.drop(columns=[target_column], errors='ignore')
    y = data[target_column]

    # Update numeric_cols to only include columns present in X
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Replace any remaining NaNs in numeric features with 0
    X[numeric_cols] = X[numeric_cols].fillna(0)

    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y, scaler

# ---------------------------------------
# 4. Model Training
# ---------------------------------------
def train_model(X_train, y_train):
    """Train Random Forest classifier."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# ---------------------------------------
# 5. Model Evaluation and Visualization
# ---------------------------------------
def evaluate_model(clf, X_test, y_test):
    """Evaluate model and visualize results."""
    # Make predictions
    y_pred = clf.predict(X_test)

    # Compute confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm.diagonal().sum()) / cm.sum()
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    target_names = ['UC1', 'UC2', 'UC3', 'UC4']
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# ---------------------------------------
# Main Execution
# ---------------------------------------
def main():
    # Load data
    data_path = "dataset.csv"
    data = load_data(data_path)

    # Check if multiclass classification is applicable
    if 'attack_type' in data.columns:
        print("\n### Multiclass Classification ###")

        # Preprocess data
        data, label_encoders, numeric_cols = preprocess_data(data)

        # Prepare features and target
        X, y, scaler = prepare_features(data, numeric_cols)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        clf = train_model(X_train, y_train)

        # Evaluate model
        evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    main()
