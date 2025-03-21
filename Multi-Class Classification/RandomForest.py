import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data_path = "dataset.csv"  # File path
data = pd.read_csv(data_path)

# Drop unnecessary columns
columns_to_drop = ['Unnamed: 0', 'Time']
data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Handle missing values
numeric_cols = data.select_dtypes(include=['number']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
for col in categorical_cols:
    if not data[col].isnull().all():
        data[col] = data[col].fillna(data[col].mode()[0])

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Multiclass Classification
if 'attack_type' in data.columns:
    X = data.drop(columns=['attack_type'], errors='ignore')
    y = data['attack_type']

    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(0)

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    accuracy = (cm.diagonal().sum()) / cm.sum()
    print(f"Accuracy: {accuracy * 100:.2f}%")

    target_names = ['UC1', 'UC2', 'UC3', 'UC4']

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Bar Graph Plot
    prediction_counts = pd.Series(y_pred).value_counts(sort=False)
    prediction_counts.index = target_names

    plt.figure(figsize=(10, 6))
    sns.barplot(x=target_names, y=prediction_counts, hue=target_names, palette="pastel", dodge=False, legend=False)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Classification Distribution')
    plt.show()


# Bar Graph Plot
prediction_counts = pd.Series(y_pred).value_counts(sort=False)
prediction_counts.index = target_names

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=target_names, y=prediction_counts, hue=target_names, palette="pastel", dodge=False, legend=False)

# Set tick positions before modifying labels
ax.set_xticks(range(len(target_names)))  # Set fixed tick positions for x-axis
ax.set_xticklabels(target_names, fontsize=20, rotation=0)  # Now, set labels safely

ax.set_yticks(ax.get_yticks())  # Ensure fixed tick positions for y-axis
ax.set_yticklabels([int(tick) for tick in ax.get_yticks()], fontsize=16, rotation=0)  # Format as integers

# Add labels and title with increased font size
plt.xlabel('Classes', fontsize=20)
plt.ylabel('Count', fontsize=20)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('classification_distribution.png', dpi=300, bbox_inches='tight')
plt.show()



