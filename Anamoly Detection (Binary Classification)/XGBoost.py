import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    # Clean column names by replacing special characters with underscores
    data.columns = data.columns.str.replace(r'[\W]', '_', regex=True)

    # Drop unnecessary columns if they exist
    columns_to_drop = ['Unnamed_0', 'Time']
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
def prepare_features(data, numeric_cols, target_column='snort_alert'):
    """Prepare features and target, and scale numeric features."""
    # Ensure target column exists
    if target_column not in data.columns:
        raise ValueError(f"The target column '{target_column}' is missing in the dataset.")

    # Define features (X) and target (y)
    X = data.drop(columns=[target_column], errors='ignore')
    y = data[target_column]

    # Update numeric_cols to only include columns present in X
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Replace any remaining NaNs in numeric features with 0
    X[numeric_cols] = X[numeric_cols].fillna(0)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Preserve column names

    return X_scaled, y, scaler

# ---------------------------------------
# 4. Model Training
# ---------------------------------------
def train_model(X_train, y_train):
    """Train XGBoost classifier."""
    clf = xgb.XGBClassifier(
        max_depth=1,
        learning_rate=0.05,
        n_estimators=50,
        subsample=0.5,
        colsample_bytree=0.8,
        random_state=35,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)
    return clf

# ---------------------------------------
# 5. Model Evaluation and Visualization
# ---------------------------------------
def evaluate_model(clf, X_test, y_test):
    """Evaluate model, print accuracy and classification report, and visualize confusion matrix."""
    # Make predictions
    y_pred = clf.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Print classification report
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Generate and visualize confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
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

    # Preprocess data
    data, label_encoders, numeric_cols = preprocess_data(data)

    # Prepare features and target
    X, y, scaler = prepare_features(data, numeric_cols)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

    # Train model
    clf = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    main()
