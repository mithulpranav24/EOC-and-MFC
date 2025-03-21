import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = "dataset.csv"  #file path
data = pd.read_csv(data_path)  # Read the CSV file into a DataFrame

"""
# Step 1: Inspect the dataset
print("Dataset Shape:", data.shape)  # Print the shape (rows, columns) of the dataset
print(data.info())  # Display basic information about the dataset, including column types and missing values
print(data.head())  # Display the first few rows of the dataset to inspect the data
"""

# Step 2: Drop unnecessary columns
# Specify columns to drop, such as 'Unnamed: 0' (index column) and 'Time' (if not needed)
columns_to_drop = ['Unnamed: 0', 'Time']  # Modify this list based on your dataset's structure
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])  # Drop specified columns

# Step 3: Handle missing values
# Separate numeric and categorical columns for easier handling of missing data
numeric_cols = data.select_dtypes(include=['number']).columns  # Select numeric columns
categorical_cols = data.select_dtypes(include=['object']).columns  # Select categorical columns

# Fill missing values for numeric columns with the mean of each column
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values for categorical columns with the most frequent value (mode) of each column
for col in categorical_cols:
    if not data[col].isnull().all():  # Only process columns with missing values
        data[col] = data[col].fillna(data[col].mode()[0])  # Replace NaN with the mode (most frequent value)

# Step 4: Encode categorical variables
# Create a dictionary to store LabelEncoders for each categorical column
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()  # Initialize the LabelEncoder
    data[col] = le.fit_transform(data[col])  # Encode the categorical column
    label_encoders[col] = le  # Store the encoder for potential inverse transformation later

# Step 5: Multiclass Classification
# Check if the target column 'attack_type' exists for classification
if 'attack_type' in data.columns:
    print("\n### Multiclass Classification ###")

    # Features (X) and target (y)
    X = data.drop(columns=['attack_type'], errors='ignore')  # Drop the target column (if it exists)
    y = data['attack_type']  # Set the target variable (attack_type)

    # Redefine numeric_cols based on updated X
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Ensure there are no NaN values in X before scaling
    X[numeric_cols] = X[numeric_cols].fillna(0)  # Replace any remaining NaNs with 0 (or another appropriate value)

    # Scale numeric features to normalize the data
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Split the data into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a classification model (XGBoost classifier)
    clf = xgb.XGBClassifier(n_estimators=100, random_state=42)  # Initialize XGBoost model
    clf.fit(X_train, y_train)  # Fit the model on the training data

    # Predict using the trained model on the test data
    y_pred = clf.predict(X_test)

    # Calculate the confusion matrix and accuracy
    cm = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix
    accuracy = (cm.diagonal().sum()) / cm.sum()  # Calculate accuracy based on diagonal values of the matrix
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy as a percentage

    # Define custom target labels for the classification report
    target_names = ['UC1', 'UC2', 'UC3', 'UC4']

    # Print the classification report (precision, recall, F1-score) with custom labels
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(8, 6))  # Set the size of the figure
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')  # Set the title of the plot
    plt.xlabel('Predicted')  # Label the x-axis
    plt.ylabel('True')  # Label the y-axis
    plt.show()  # Display the plot

"""
# Step 6: Visualize the prediction distribution as a pie chart
# Flatten y_pred if it is a 2D array (in case)
if len(y_pred.shape) > 1:
    y_pred = y_pred.ravel()

# Count occurrences of each classification in predictions
prediction_counts = pd.Series(y_pred).value_counts(sort=False)  # Count unique predictions
prediction_counts.index = target_names  # Assign target names as the index

# Plot the pie chart
plt.figure(figsize=(8, 8))  # Set the figure size
plt.pie(
    prediction_counts,
    labels=target_names,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette("pastel")
)
plt.title('Prediction Distribution')  # Set the title for the pie chart
plt.show()  # Display the pie chart
"""