import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest  # Added IsolationForest import
from catboost import CatBoostClassifier  # Import CatBoost
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_path = "dataset.csv"  #file path
data = pd.read_csv(data_path)  # Read the CSV dataset into a DataFrame
"""
# Step 1: Inspect the dataset
# Print basic information about the dataset such as shape, data types, and the first few rows
print("Dataset Shape:", data.shape)
print(data.info())  # Shows a concise summary of the DataFrame
print(data.head())  # Displays the first 5 rows of the dataset
"""
# Step 2: Drop unnecessary columns
# Drop columns that are not needed for the analysis
columns_to_drop = ['Unnamed: 0', 'Time']  # Update based on your dataset's structure
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])  # Drop only the existing columns

# Step 3: Handle missing values
# Separate numeric and categorical columns to handle missing values appropriately
numeric_cols = data.select_dtypes(include=['number']).columns  # Identify numeric columns
categorical_cols = data.select_dtypes(include=['object']).columns  # Identify categorical columns

# Fill missing values for numeric columns with the mean value of each column
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Fill missing values for categorical columns with the most frequent value (mode)
for col in categorical_cols:
    if not data[col].isnull().all():  # Check if the column has any missing values
        data[col] = data[col].fillna(data[col].mode()[0])  # Fill missing values with the mode

# Step 4: Encode categorical variables
# Apply label encoding to transform categorical columns into numeric values
label_encoders = {}  # Dictionary to store label encoders for each categorical column
for col in categorical_cols:
    le = LabelEncoder()  # Initialize LabelEncoder
    data[col] = le.fit_transform(data[col])  # Apply label encoding
    label_encoders[col] = le  # Store the encoder for future use

# Step 5: Multiclass Classification
if 'attack_type' in data.columns:  # Check if the target column 'attack_type' exists
    print("\n### Multiclass Classification ###")

    # Define features (X) and target variable (y)
    X = data.drop(columns=['attack_type'], errors='ignore')  # Features: all columns except 'attack_type'
    y = data['attack_type']  # Target variable: 'attack_type'

    # Redefine numeric_cols based on updated features (X)
    numeric_cols = X.select_dtypes(include=['number']).columns

    # Ensure there are no NaN values in the features before scaling
    X[numeric_cols] = X[numeric_cols].fillna(0)  # Replace NaNs with 0 or other suitable value

    # Scale numeric features to standardize them
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Step 6: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=42)  # 70% training, 30% testing

    # Step 7: Train the model using CatBoostClassifier
    clf = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)  # Initialize CatBoostClassifier
    clf.fit(X_train, y_train)  # Train the model on the training data

    # Step 8: Make predictions on the test set
    y_pred = clf.predict(X_test)  # Get the model's predictions on the test set

    # Step 9: Evaluate the model's performance
    cm = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix
    accuracy = (cm.diagonal().sum()) / cm.sum()  # Calculate the accuracy from the confusion matrix
    print(f"Accuracy: {accuracy * 100:.2f}%")  # Display accuracy as a percentage

    # Define custom target labels for classification report
    target_names = ['UC1', 'UC2', 'UC3', 'UC4']  # Replace with your actual target names

    # Print detailed classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Step 10: Visualize the confusion matrix
    plt.figure(figsize=(8, 6))  # Set figure size
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names,
                yticklabels=target_names)  # Plot heatmap
    plt.title('Confusion Matrix')  # Set plot title
    plt.xlabel('Predicted')  # X-axis label
    plt.ylabel('True')  # Y-axis label
    plt.show()  # Display the plot


# Step 11: Visualize the distribution of predictions
# Flatten y_pred if it's a 2D array
y_pred_flat = y_pred.ravel()  # Convert (n, 1) to (n,)

# Count occurrences of each classification in predictions
prediction_counts = pd.Series(y_pred_flat).value_counts(sort=False)  # Count values in y_pred
prediction_counts.index = target_names  # Set the index to target names

# Plot the pie chart
plt.figure(figsize=(8, 8))  # Set figure size
plt.pie(prediction_counts, labels=target_names, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
plt.title('Classification Distribution')  # Set title
plt.show()  # Display the pie chart

