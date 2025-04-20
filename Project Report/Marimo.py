import marimo

__generated_with = "0.12.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 2.5em; font-weight: bold;">22MAT122 - Mathematics for Computing 2</h1>
            <h2 style="font-size: 2em; font-weight: bold;">Machine Learning-Based Detection of MiTM Attacks in SCADA-Controlled Power Systems</h2>
            <p style="font-size: 1.5em;">
                <strong>M Mithul Pranav CB.SC.U4AIE24331</strong><br>
                <strong>Rithan S CB.SC.U4AIE24348</strong>
            </p>
        </div>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Abstract
        This study presents a machine learning-based approach for detecting Man-in-the-Middle (MiTM) attacks in Supervisory Control and Data Acquisition (SCADA)-controlled power systems. By leveraging supervised learning algorithms, such as Random Forest and Support Vector Machines, the proposed model analyzes network traffic patterns to identify anomalies indicative of MiTM attacks. The system was trained and tested on a dataset simulating SCADA communication protocols under various attack scenarios. Results demonstrate a detection accuracy of over 95%, with low false-positive rates, offering a robust solution for enhancing the cybersecurity of critical power infrastructure.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Introduction
        The increasing reliance on SCADA systems for managing power grids has made them prime targets for cyberattacks, particularly Man-in-the-Middle (MiTM) attacks, which compromise data integrity and system reliability. Traditional intrusion detection methods often struggle to keep pace with the sophistication of modern cyber threats, necessitating advanced solutions. This work is motivated by the critical need to safeguard power systems, which underpin societal and economic stability. By employing machine learning to detect MiTM attacks, this research addresses the challenge of real-time anomaly detection in SCADA networks, offering a scalable and adaptive approach to bolster the resilience of power infrastructure against evolving cyber threats.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Dataset Description
        Focusing on Man-in-the-Middle (MiTM) attacks against DNP3-based substations, the dataset is designed to record cyber-physical interactions inside a power grid setting. Comprising of raw network recordings, processed cyber-physical variables, and tagged attack scenarios, the dataset is appropriate for security research and machine learning-based intrusion detection. Among the many attack scenarios the dataset records are False Command Injection (FCI) and False Data Injection (FDI). FCI assaults can cause power outages or line overloading by altering binary control commands such as circuit breaker statuses. FDI attacks create false control decisions by means of sensor data acquired by the master controller. Certain assaults mix FDI and FCI, which complicates detection. Recorded under several use cases, these assault scenarios each denote a different infiltration technique. Based on DNP3 polling rates, the number of impacted outstations, and attack execution techniques, the dataset classifies these situations.

        As indicated in Table 1, the dataset offers a thorough feature set for both cyber and physical system characteristics. Cyber features help to detect network irregularities; the DNP3-specific features included in Table 2 offer information on the operational effect of attacks on power grid communications. The dataset guarantees network events and power system reactions are coupled by including time-aligned cyber-physical variables, enabling thorough intrusion detection studies. Especially in the area of intrusion detection for smart grids and industrial control systems, this dataset is a valuable tool for cybersecurity research. It helps to create machine learning and deep learning models for spotting MiTM threats by offering both raw and processed cyber-physical data. The dataset's organized organization and annotated attack scenarios make it perfect for training, testing, and assessing AI-driven security solutions in critical infrastructure settings.

        The labels UC1, UC2, UC3, and UC4 are custom identifiers used to classify different types of MiTM attack scenarios in this study. Their corresponding attack types are detailed in Table 3.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Table 1: Use Cases and Attack Types
        <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
            <thead>
                <tr style="background-color: #f2f2f2; color: black !important;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;"><strong>Use Case</strong></th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;"><strong>Attack Type</strong></th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">UC1</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">False Command Injection (FCI)</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">UC2</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">False Command Injection (FCI) + Analog Command Manipulation</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">UC3</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">False Data Injection (FDI) + False Command Injection (FCI)</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">UC4</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Three-Stage Attack</td>
                </tr>
            </tbody>
        </table>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Table 2: Cyber Features and Descriptions
        <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
            <thead>
                <tr style="background-color: #f2f2f2; color: black !important;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;"><strong>Feature</strong></th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;"><strong>Description</strong></th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Frame Length</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Size of the network frame in bytes.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Protocols</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">List of encapsulated protocols (TCP, ARP, DNP3, etc.).</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Ethernet Src/Dst</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Unique MAC addresses (used in ARP spoofing detection).</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">IP Src/Dst</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Unique IP addresses of sender and receiver.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">IP Length</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Total IP packet size (correlates with DNP3 payload size).</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">TCP Length</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Size of TCP segments (useful for identifying payload modifications).</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">TCP Flags</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Indicators for SYN, ACK, FIN states in TCP connections.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Retransmission</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Detects packet loss due to attacks or congestion.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">RTT</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">High RTT may indicate MiTM attack.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Flow Count</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Number of TCP flows (indicates connection attempts and disruptions).</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Packet Count</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Number of packets transmitted in a specific time interval.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Snort Alert</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Boolean flag indicating IDS detection.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Alert Type</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Attack type (DNP3 manipulation, ARP spoof, ICMP flood, etc.).</td>
                </tr>
            </tbody>
        </table>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ### Table 3: DNP3-Specific Features and Descriptions
        <table style="width: 100%; border-collapse: collapse; margin: 0 auto;">
            <thead>
                <tr style="background-color: #f2f2f2; color: black !important;">
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;"><strong>Feature</strong></th>
                    <th style="border: 1px solid #ddd; padding: 8px; text-align: center;"><strong>Description</strong></th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">DNP3 Link Layer Src/Dst</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">ID of the DNP3 master or outstation involved in communication.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">DNP3 Link Layer Length</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Payload size (correlates with function type).</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">DNP3 Control Flags</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Indicators of message state (e.g., whether it’s a READ, WRITE, or OPERATE command).</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">DNP3 Object Count</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Number of Binary Inputs (BI), Binary Outputs (BO), Analog Inputs (AI), Analog Outputs (AO) in a message.</td>
                </tr>
                <tr>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">DNP3 Payload</td>
                    <td style="border: 1px solid #ddd; padding: 8px; text-align: center;">Extracted physical values such as breaker status, power flow, and bus injections.</td>
                </tr>
            </tbody>
        </table>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Methodology
        ### Overview
        This study employs a structured machine learning pipeline to classify network attacks, focusing on binary classification (e.g., presence or absence of an attack indicated by `snort_alert`) and multiclass classification (e.g., specific attack types such as UC1, UC2, UC3, UC4). The methodology leverages diverse algorithms—LightGBM, Support Vector Machine (SVM), Random Forest, XGBoost, and CatBoost—to compare their performance on a preprocessed network attack dataset. The pipeline includes data loading, preprocessing, feature engineering, model training, and evaluation, with visualizations to assess model effectiveness.

        ### 1. Data Loading
        The process begins with loading the dataset from a CSV file (`dataset.csv`) using the `pandas` library. This step ensures the raw data, containing network traffic features and attack labels, is accessible for subsequent analysis. The function `load_data` handles the file input, providing a `pandas` DataFrame for further processing.

        ### 2. Data Preprocessing
        Data preprocessing is critical to ensure quality input for the models. The following steps are applied:
        - **Column Cleaning**: Special characters in column names are replaced with underscores using regular expressions to ensure compatibility with machine learning libraries.
        - **Column Dropping**: Unnecessary columns such as `Unnamed_0` and `Time` are removed if present.
        - **Missing Value Handling**: Numeric columns are filled with their mean values, while categorical columns are filled with their mode to address missing data. For SVM, categorical columns entirely filled with NaN are dropped before encoding.
        - **Categorical Encoding**: Categorical variables are encoded using `LabelEncoder` to convert them into numerical format, with a dictionary of encoders (`label_encoders`) stored for potential inverse transformations.

        ### 3. Feature Engineering and Scaling
        Feature preparation involves isolating features and the target variable, followed by scaling to standardize the data:
        - **Feature and Target Separation**: Features (X) are defined by dropping the target column (`snort_alert` for binary or `attack_type` for multiclass), while the target variable (y) is extracted.
        - **Numeric Column Update**: Only numeric columns present in the feature set are considered.
        - **Missing Value Imputation**: Remaining NaNs in numeric features are replaced with 0.
        - **Additional Cleaning (Random Forest)**: Columns with zero variance or all NaN values are dropped to improve model stability.
        - **Scaling**: Numeric features are standardized using `StandardScaler` to ensure consistent scales across features, preserving column names in a `pandas` DataFrame.

        ### 4. Model Training
        Five machine learning models are trained on the preprocessed data:
        - **LightGBM**: Configured with `gbdt` boosting, a maximum depth of 1, learning rate of 0.07, 50 estimators, subsample of 0.5, and colsample_bytree of 0.8, with verbosity suppressed.
        - **SVM**: Utilizes an RBF kernel with a default regularization parameter (C=1.0) and gamma set to 'scale'.
        - **Random Forest**: Employs 100 estimators with a fixed random state for reproducibility.
        - **XGBoost**: Set with a maximum depth of 1, learning rate of 0.05, 50 estimators, subsample of 0.5, colsample_bytree of 0.8, and `logloss` as the evaluation metric.
        - **CatBoost**: Uses 100 iterations with verbosity disabled for efficiency.
        Each model is fitted to the training data (`X_train`, `y_train`) split from the dataset.

        ### 5. Model Evaluation and Visualization
        Model performance is assessed using the following metrics and visualizations:
        - **Accuracy**: Computed as the percentage of correct predictions on the test set.
        - **Classification Report**: Provides precision, recall, and F1-score for each class (binary: 0/1; multiclass: UC1, UC2, UC3, UC4).
        - **Confusion Matrix**: Visualized using `seaborn` heatmaps with annotations, plotted with `matplotlib`, to show true vs. predicted labels. The matrix is customized with class labels for multiclass cases.
        The test set (`X_test`, `y_test`) is used to generate predictions, and results are printed and displayed graphically.

        ### 6. Main Execution
        The pipeline is executed as follows:
        - Data is loaded from `dataset.csv`.
        - Preprocessing and feature engineering are applied.
        - The dataset is split into training (70% for binary, 80% for multiclass) and testing sets using `train_test_split` with stratification to preserve class distribution.
        - Each model is trained and evaluated, with results output for comparison.
        For multiclass classification, the process is conditional on the presence of the `attack_type` column.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Network Attack Classification with Multiple Models (Multiclass)

        This notebook performs multiclass classification on a network attack dataset using four machine learning models: LightGBM, CatBoost, Random Forest, and XGBoost. The dataset is preprocessed, features are scaled, and each model is trained and evaluated. The notebook is organized into three sections: Code, Algorithms Used, and Results and Discussion/Analysis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Code (Multiclass)""")
    return


@app.cell
def _(os):
    os.environ["LOKY_MAX_CPU_COUNT"] = "16"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1. Data Loading (Multiclass)""")
    return


@app.cell
def _(pd):
    def multi_load_data(file_path):
        """Load dataset from a CSV file for multiclass classification.

        Args:
            file_path (str): Path to the CSV file containing the dataset.

        Returns:
            pandas.DataFrame: Loaded dataset as a DataFrame.
        """
        data = pd.read_csv(file_path)
        return data
    return (multi_load_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2. Data Preprocessing (Multiclass)""")
    return


@app.cell
def _(LabelEncoder):
    def multi_preprocess_data(data):
        """Clean and preprocess the dataset for multiclass modeling.

        Args:
            data (pandas.DataFrame): Input dataset.

        Returns:
            tuple: (processed DataFrame, dictionary of label encoders, list of numeric column names).
        """
        data.columns = data.columns.str.replace(r'\W', '_', regex=True)
        columns_to_drop = ['Unnamed: 0', 'Time']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        for col in categorical_cols:
            if not data[col].isnull().all():
                data[col] = data[col].fillna(data[col].mode()[0])
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        return data, label_encoders, numeric_cols
    return (multi_preprocess_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3. Feature Engineering and Scaling (Multiclass)""")
    return


@app.cell
def _(StandardScaler):
    def multi_prepare_features(data, numeric_cols, target_column='attack_type'):
        """Prepare features and target variables, and scale numeric features for multiclass.

        Args:
            data (pandas.DataFrame): Preprocessed dataset.
            numeric_cols (list): List of numeric column names.
            target_column (str): Name of the target column (default: 'attack_type').

        Returns:
            tuple: (feature matrix X, target vector y, fitted StandardScaler object).
        """
        X = data.drop(columns=[target_column], errors='ignore')
        y = data[target_column]
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
        return X, y, scaler
    return (multi_prepare_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4. Model Evaluation and Visualization (Multiclass)""")
    return


@app.cell
def _(classification_report, confusion_matrix, plt, sns):
    def multi_evaluate_model(clf, X_test, y_test, model_name):
        """Evaluate a trained model and visualize its performance for multiclass.

        Args:
            clf: Trained classifier model.
            X_test (pandas.DataFrame): Test feature matrix.
            y_test (pandas.Series): Test target vector.
            model_name (str): Name of the model for display purposes.

        Returns:
            float: Accuracy of the model.
        """
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = (cm.diagonal().sum()) / cm.sum()
        print(f"\n{model_name} Accuracy: {accuracy * 100:.2f}%")
        target_names = ['UC1', 'UC2', 'UC3', 'UC4']
        print(f"\n{model_name} Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        return accuracy
    return (multi_evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 5. Model Training (Multiclass)""")
    return


@app.cell
def _(lgb):
    def multi_train_lightgbm(X_train, y_train):
        """Train a LightGBM classifier for multiclass.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            lgb.LGBMClassifier: Trained LightGBM model.
        """
        clf = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        clf.fit(X_train, y_train)
        return clf
    return (multi_train_lightgbm,)


@app.cell
def _(CatBoostClassifier):
    def multi_train_catboost(X_train, y_train):
        """Train a CatBoost classifier for multiclass.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            CatBoostClassifier: Trained CatBoost model.
        """
        clf = CatBoostClassifier(iterations=100, random_seed=42, verbose=0)
        clf.fit(X_train, y_train)
        return clf
    return (multi_train_catboost,)


@app.cell
def _(RandomForestClassifier):
    def multi_train_random_forest(X_train, y_train):
        """Train a Random Forest classifier for multiclass.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            RandomForestClassifier: Trained Random Forest model.
        """
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        return clf
    return (multi_train_random_forest,)


@app.cell
def _(xgb):
    def multi_train_xgboost(X_train, y_train):
        """Train an XGBoost classifier for multiclass.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            xgb.XGBClassifier: Trained XGBoost model.
        """
        clf = xgb.XGBClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        return clf
    return (multi_train_xgboost,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 6. Main Execution (Multiclass)""")
    return


@app.cell
def _(
    multi_evaluate_model,
    multi_load_data,
    multi_prepare_features,
    multi_preprocess_data,
    multi_train_catboost,
    multi_train_lightgbm,
    multi_train_random_forest,
    multi_train_xgboost,
    train_test_split,
):
    # Load the dataset
    multi_data_path = "dataset.csv"
    multi_data = multi_load_data(multi_data_path)
    multi_accuracies = []

    if 'attack_type' in multi_data.columns:
        print("\n### Multiclass Classification ###")
        multi_data, multi_label_encoders, multi_numeric_cols = multi_preprocess_data(multi_data)
        multi_X, multi_y, multi_scaler = multi_prepare_features(multi_data, multi_numeric_cols)
        multi_X_train, multi_X_test, multi_y_train, multi_y_test = train_test_split(multi_X, multi_y, test_size=0.3, random_state=42)
        print("\nTraining and Evaluating LightGBM...")
        multi_clf_lightgbm = multi_train_lightgbm(multi_X_train, multi_y_train)
        multi_lightgbm_accuracy = multi_evaluate_model(multi_clf_lightgbm, multi_X_test, multi_y_test, "LightGBM")
        multi_accuracies.append({"Model": "LightGBM", "Accuracy (%)": multi_lightgbm_accuracy * 100})
        print("\nTraining and Evaluating CatBoost...")
        multi_clf_catboost = multi_train_catboost(multi_X_train, multi_y_train)
        multi_catboost_accuracy = multi_evaluate_model(multi_clf_catboost, multi_X_test, multi_y_test, "CatBoost")
        multi_accuracies.append({"Model": "CatBoost", "Accuracy (%)": multi_catboost_accuracy * 100})
        print("\nTraining and Evaluating Random Forest...")
        multi_clf_rf = multi_train_random_forest(multi_X_train, multi_y_train)
        multi_rf_accuracy = multi_evaluate_model(multi_clf_rf, multi_X_test, multi_y_test, "Random Forest")
        multi_accuracies.append({"Model": "Random Forest", "Accuracy (%)": multi_rf_accuracy * 100})
        print("\nTraining and Evaluating XGBoost...")
        multi_clf_xgboost = multi_train_xgboost(multi_X_train, multi_y_train)
        multi_xgboost_accuracy = multi_evaluate_model(multi_clf_xgboost, multi_X_test, multi_y_test, "XGBoost")
        multi_accuracies.append({"Model": "XGBoost", "Accuracy (%)": multi_xgboost_accuracy * 100})

    return (
        multi_X,
        multi_X_test,
        multi_X_train,
        multi_accuracies,
        multi_catboost_accuracy,
        multi_clf_catboost,
        multi_clf_lightgbm,
        multi_clf_rf,
        multi_clf_xgboost,
        multi_data,
        multi_data_path,
        multi_label_encoders,
        multi_lightgbm_accuracy,
        multi_numeric_cols,
        multi_rf_accuracy,
        multi_scaler,
        multi_xgboost_accuracy,
        multi_y,
        multi_y_test,
        multi_y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Algorithms Used (Multiclass)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following machine learning algorithms were used for multiclass classification of network attack types:

        1. **LightGBM**: A gradient boosting framework that uses tree-based learning algorithms. It is optimized for speed and efficiency, particularly with large datasets, and handles categorical features well with proper preprocessing.
        2. **CatBoost**: A gradient boosting algorithm designed to handle categorical features automatically. It reduces overfitting and is robust for datasets with complex patterns.
        3. **Random Forest**: An ensemble method that constructs multiple decision trees and combines their outputs. It is effective for handling high-dimensional data and provides feature importance insights.
        4. **XGBoost**: An optimized gradient boosting algorithm known for its high performance and scalability. It is widely used in machine learning competitions due to its accuracy and flexibility.

        Each model was configured with 100 estimators/iterations and a fixed random seed (42) for reproducibility.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results""")
    return


@app.cell
def _(multi_accuracies, pd):
    # Create a DataFrame for the accuracy table
    multi_accuracy_df = pd.DataFrame(multi_accuracies)
    multi_accuracy_df["Accuracy (%)"] = multi_accuracy_df["Accuracy (%)"].round(2)
    print("\n### Model Accuracy Comparison (Multiclass) ###")
    print(multi_accuracy_df.to_string(index=False))
    return (multi_accuracy_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Network Attack Classification with Multiple Models (Binary)

        This notebook performs binary classification on a network attack dataset using four different classifiers: Random Forest, Support Vector Machine (SVM), LightGBM, and XGBoost. The dataset is preprocessed, features are scaled, and each model is trained and evaluated. The notebook is organized into three sections: Code, Algorithms Used, and Results and Discussion/Analysis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Code (Binary)""")
    return


@app.cell
def _(os):
    os.environ["LOKY_MAX_CPU_COUNT"] = "16"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 1. Data Loading (Binary)""")
    return


@app.cell
def _(pd):
    def binary_load_data(file_path):
        """Load dataset from a CSV file for binary classification.

        Args:
            file_path (str): Path to the CSV file containing the dataset.

        Returns:
            pandas.DataFrame: Loaded dataset as a DataFrame.
        """
        data = pd.read_csv(file_path)
        return data
    return (binary_load_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2. Data Preprocessing (Binary)""")
    return


@app.cell
def _(LabelEncoder):
    def binary_preprocess_data(data):
        """Clean and preprocess the dataset for binary modeling.

        Args:
            data (pandas.DataFrame): Input dataset.

        Returns:
            tuple: (processed DataFrame, dictionary of label encoders, list of numeric column names).
        """
        data.columns = data.columns.str.replace(r'[\W]', '_', regex=True)
        columns_to_drop = ['Unnamed_0', 'Time']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
        data = data.dropna(axis=1, how='all')
        categorical_cols = data.select_dtypes(include=['object']).columns
        label_encoders = {}
        for col in categorical_cols:
            if not data[col].isnull().all():
                data[col] = data[col].fillna(data[col].mode()[0])
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
        return data, label_encoders, numeric_cols
    return (binary_preprocess_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 3. Feature Engineering and Scaling (Binary)""")
    return


@app.cell
def _(StandardScaler, pd):
    def binary_prepare_features(data, numeric_cols, target_column='snort_alert'):
        """Prepare features and target variables, and scale numeric features for binary.

        Args:
            data (pandas.DataFrame): Preprocessed dataset.
            numeric_cols (list): List of numeric column names.
            target_column (str): Name of the target column (default: 'snort_alert').

        Returns:
            tuple: (scaled feature matrix X, target vector y, fitted StandardScaler object).
        """
        if target_column not in data.columns:
            raise ValueError(f"The target column '{target_column}' is missing in the dataset.")
        X = data.drop(columns=[target_column], errors='ignore')
        y = data[target_column]
        numeric_cols = X.select_dtypes(include=['number']).columns
        X[numeric_cols] = X[numeric_cols].fillna(0)
        X = X.dropna(axis=1, how="all")
        zero_variance_cols = X.loc[:, X.nunique() <= 1].columns
        X = X.drop(columns=zero_variance_cols)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        return X_scaled, y, scaler
    return (binary_prepare_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4. Model Training (Binary)""")
    return


@app.cell
def _(RandomForestClassifier, SVC, lgb, xgb):
    def binary_train_random_forest(X_train, y_train):
        """Train a Random Forest classifier for binary.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            RandomForestClassifier: Trained Random Forest model.
        """
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)
        return clf

    def binary_train_svm(X_train, y_train):
        """Train an SVM classifier with RBF kernel for binary.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            SVC: Trained SVM model.
        """
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        clf.fit(X_train, y_train)
        return clf

    def binary_train_lightgbm(X_train, y_train):
        """Train a LightGBM classifier for binary.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            lgb.LGBMClassifier: Trained LightGBM model.
        """
        clf = lgb.LGBMClassifier(
            boosting_type='gbdt',
            max_depth=1,
            learning_rate=0.07,
            n_estimators=50,
            subsample=0.5,
            colsample_bytree=0.8,
            random_state=40,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        return clf

    def binary_train_xgboost(X_train, y_train):
        """Train an XGBoost classifier for binary.

        Args:
            X_train (pandas.DataFrame): Training feature matrix.
            y_train (pandas.Series): Training target vector.

        Returns:
            xgb.XGBClassifier: Trained XGBoost model.
        """
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

    return (
        binary_train_lightgbm,
        binary_train_random_forest,
        binary_train_svm,
        binary_train_xgboost,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 5. Model Evaluation and Visualization (Binary)""")
    return


@app.cell
def _(accuracy_score, classification_report, confusion_matrix, plt, sns):
    def binary_evaluate_model(clf, X_test, y_test, model_name):
        """Evaluate a trained model and visualize its performance for binary.

        Args:
            clf: Trained classifier model.
            X_test (pandas.DataFrame): Test feature matrix.
            y_test (pandas.Series): Test target vector.
            model_name (str): Name of the model for display purposes.

        Returns:
            float: Accuracy of the model.
        """
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
        print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
        conf_matrix = confusion_matrix(y_pred, y_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        return accuracy
    return (binary_evaluate_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 6. Main Execution (Binary)""")
    return


@app.cell
def _(
    binary_evaluate_model,
    binary_load_data,
    binary_prepare_features,
    binary_preprocess_data,
    binary_train_lightgbm,
    binary_train_random_forest,
    binary_train_svm,
    binary_train_xgboost,
    train_test_split,
):
    # Load the dataset
    binary_data_path = "dataset.csv"
    binary_data = binary_load_data(binary_data_path)
    binary_accuracies = []

    binary_data, binary_label_encoders, binary_numeric_cols = binary_preprocess_data(binary_data)
    binary_X, binary_y, binary_scaler = binary_prepare_features(binary_data, binary_numeric_cols)
    binary_X_train, binary_X_test, binary_y_train, binary_y_test = train_test_split(binary_X, binary_y, test_size=0.3, random_state=42, stratify=binary_y)
    models = [
        ("Random Forest", binary_train_random_forest),
        ("SVM", binary_train_svm),
        ("LightGBM", binary_train_lightgbm),
        ("XGBoost", binary_train_xgboost)
    ]
    for model_name, train_func in models:
        print(f"\nTraining and evaluating {model_name}...")
        clf = train_func(binary_X_train, binary_y_train)
        accuracy = binary_evaluate_model(clf, binary_X_test, binary_y_test, model_name)
        binary_accuracies.append({"Model": model_name, "Accuracy (%)": accuracy * 100})

    return (
        accuracy,
        binary_X,
        binary_X_test,
        binary_X_train,
        binary_accuracies,
        binary_data,
        binary_data_path,
        binary_label_encoders,
        binary_numeric_cols,
        binary_scaler,
        binary_y,
        binary_y_test,
        binary_y_train,
        clf,
        model_name,
        models,
        train_func,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Algorithms Used (Binary)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following machine learning algorithms were used for binary classification of network attack alerts:

        1. **Random Forest**: An ensemble method that builds multiple decision trees and combines their outputs via voting. It is robust to high-dimensional data and provides feature importance insights.
        2. **Support Vector Machine (SVM)**: A classifier that finds the optimal hyperplane to separate classes, using an RBF kernel for non-linear data. It is effective for balanced datasets but can be computationally intensive.
        3. **LightGBM**: A gradient boosting framework optimized for speed and efficiency, using tree-based learning. It is particularly effective for large datasets and imbalanced classes.
        4. **XGBoost**: An optimized gradient boosting algorithm known for its high performance and scalability. It handles imbalanced data well and is widely used for its accuracy.

        Each model was configured with specific hyperparameters (e.g., 50 estimators for LightGBM and XGBoost, RBF kernel for SVM) and a fixed random seed for reproducibility.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results and Discussion/Analysis (Binary)""")
    return


@app.cell
def _(binary_accuracies, pd):
    # Create a DataFrame for the accuracy table
    binary_accuracy_df = pd.DataFrame(binary_accuracies)
    binary_accuracy_df["Accuracy (%)"] = binary_accuracy_df["Accuracy (%)"].round(2)
    print("\n### Model Accuracy Comparison (Binary) ###")
    print(binary_accuracy_df.to_string(index=False))
    return (binary_accuracy_df,)


@app.cell(hide_code=True)
def _():
    import os  # For setting environment variables
    import pandas as pd  # For data manipulation and analysis
    import matplotlib.pyplot as plt  # For plotting
    import seaborn as sns  # For enhanced visualization of confusion matrix

    from sklearn.preprocessing import StandardScaler, LabelEncoder  # For feature scaling and encoding categorical variables
    from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  # For model evaluation metrics

    from sklearn.ensemble import RandomForestClassifier  # For Random Forest classifier
    from sklearn.svm import SVC  # For Support Vector Machine classifier
    import lightgbm as lgb  # For LightGBM classifier
    import xgboost as xgb  # For XGBoost classifier
    from catboost import CatBoostClassifier  # For CatBoost classifier

    import marimo as mo

    return (
        CatBoostClassifier,
        LabelEncoder,
        RandomForestClassifier,
        SVC,
        StandardScaler,
        accuracy_score,
        classification_report,
        confusion_matrix,
        lgb,
        mo,
        os,
        pd,
        plt,
        sns,
        train_test_split,
        xgb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References
        [1] A. Dagoumas, "Assessing the impact of cybersecurity attacks on power systems," Energies, vol. 12, no. 4, p. 725, 2019.

        [2] S. Sahoo, T. Dragičević, and F. Blaabjerg, "Multilayer resilience paradigm against cyber attacks in DC microgrids," IEEE Trans. Power Electron., vol. 36, no. 3, pp. 2522–2532, 2020.

        [3] S. Y. Diaba, M. Shafie-Khah, and M. Elmusrati, "Cyber security in power systems using metaheuristic and deep learning algorithms," IEEE Access, vol. 11, pp. 18660–18672, 2023.

        [4] M. Tanveer, N. Kumar, A. Naushad, S. A. Chaudhry, et al., "A robust access control protocol for the smart grid systems," IEEE Internet Things J., vol. 9, no. 9, pp. 6855–6865, 2021.

        [5] Ö. Sen, D. van der Velde, P. Linnartz, I. Hacker, M. Henze, M. Andres, and A. Ulbig, "Investigating man-in-the-middle-based false data injection in a smart grid laboratory environment," in Proc. IEEE PES Innov. Smart Grid Technol. Eur. (ISGT Europe), 2021, pp. 01–06.

        [6] D. Mishchenko, I. Oleinikova, L. Erdődi, and B. R. Pokhrel, "Multidomain cyber-physical testbed for power system vulnerability assessment," IEEE Access, 2024.

        [7] M. S. Rahman, M. A. Mahmud, A. M. T. Oo, and H. R. Pota, "Multi-agent approach for enhancing security of protection schemes in cyber-physical energy systems," IEEE Trans. Ind. Informat., vol. 13, no. 2, pp. 436–447, 2016.

        [8] I. Parvez, M. Aghili, H. Riggs, A. Sundararajan, A. I. Sarwat, and A. K. Srivastava, "A novel authentication management for the data security of smart grid," IEEE Open Access J. Power Energy, 2024.

        [9] S. Darzi, B. Akhbari, and H. Khodaiemehr, "LPM2DA: A lattice-based privacy-preserving multifunctional and multi-dimensional data aggregation scheme for smart grid," Cluster Comput., vol. 25, no. 1, pp. 263–278, 2022.

        [10] H. Shi, L. Xie, and L. Peng, "Detection of false data injection attacks in smart grid based on a new dimensionality-reduction method," Comput. Electr. Eng., vol. 91, p. 107058, 2021.

        [11] M. J. Abdulaal, M. I. Ibrahem, M. M. Mahmoud, J. Khalid, A. J. Aljohani, A. H. Milyani, and A. M. Abusorrah, "Real-time detection of false readings in smart grid AMI using deep and ensemble learning," IEEE Access, vol. 10, pp. 47541–47556, 2022.

        [12] Y. He, G. J. Mendis, and J. Wei, "Real-time detection of false data injection attacks in smart grid: A deep learning-based intelligent mechanism," IEEE Trans. Smart Grid, vol. 8, no. 5, pp. 2505–2516, 2017.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Acknowledgements
        We would like to express our sincere gratitude to our faculty at Amrita Vishwa Vidhyapeetham for their invaluable guidance and support throughout this project. Special thanks are extended to the faculty of the 22MAT122 - Mathematics for Computing 2 course Dr. Sunil Kumar for providing the academic knowledge that enabled this research. This project would not have been possible without the collective efforts and contributions of all involved.
        """
    )
    return


if __name__ == "__main__":
    app.run()
