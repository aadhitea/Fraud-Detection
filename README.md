# XGBoost Fraud Analytics

An end-to-end **Fraud Detection and Risk Scoring platform using XGBoost**, developed using the PaySim synthetic financial transaction dataset.

The project focuses on building a production-oriented machine learning workflow for identifying potentially fraudulent financial transactions while addressing the key challenges associated with fraud analytics, including **high class imbalance, feature engineering, model explainability, threshold optimization, and false-positive management**.

---

## Project Overview

Fraud detection is a highly imbalanced binary classification problem where fraudulent transactions represent only a small fraction of total transactions.

This project uses **XGBoost** to develop a fraud classification model capable of assigning a fraud probability to individual financial transactions.

The solution covers the complete machine learning lifecycle:

```text
Raw Transaction Data
        |
        v
Data Quality & Exploration
        |
        v
Feature Engineering
        |
        v
Train / Validation / Test Split
        |
        v
XGBoost Fraud Classifier
        |
        v
Probability Scoring
        |
        v
Threshold Optimization
        |
        v
Fraud / Non-Fraud Decision
        |
        v
Model Explainability & Analytics
```

---

## Dataset

This project uses the **PaySim Synthetic Financial Dataset**.

Dataset:

**PaySim — Synthetic Financial Datasets for Fraud Detection**

Source:

https://www.kaggle.com/datasets/ealaxi/paysim1

The dataset simulates mobile-money transactions based on a financial transaction simulator.

It contains transaction-level information such as:

| Feature          | Description                            |
| ---------------- | -------------------------------------- |
| `step`           | Time step in the simulation            |
| `type`           | Transaction type                       |
| `amount`         | Transaction amount                     |
| `nameOrig`       | Originating customer                   |
| `oldbalanceOrg`  | Originator balance before transaction  |
| `newbalanceOrig` | Originator balance after transaction   |
| `nameDest`       | Destination customer                   |
| `oldbalanceDest` | Destination balance before transaction |
| `newbalanceDest` | Destination balance after transaction  |
| `isFraud`        | Fraud indicator                        |
| `isFlaggedFraud` | Existing rule-based fraud flag         |

Transaction types include:

```text
CASH_IN
CASH_OUT
DEBIT
PAYMENT
TRANSFER
```

### Important Dataset Consideration

PaySim is a **synthetic dataset**. Therefore, model performance on this dataset should not be interpreted as equivalent to performance on production banking or insurance transaction data.

The dataset is primarily used for:

* Fraud analytics experimentation
* XGBoost development
* Feature engineering
* Imbalanced classification
* Model explainability
* ML engineering demonstrations
* Interview / assessment projects

---

# Business Problem

The objective is to determine whether a financial transaction should be classified as:

```text
0 = Legitimate Transaction
1 = Fraudulent Transaction
```

However, fraud detection should not be treated as a simple accuracy optimization problem.

For example:

If:

```text
1,000,000 transactions
10,000 fraudulent
990,000 legitimate
```

A model predicting every transaction as legitimate could achieve:

```text
99% Accuracy
```

while detecting:

```text
0% of fraud
```

Therefore, the project emphasizes **precision-recall trade-offs, fraud capture rate, false positives and business-oriented threshold selection**.

---

# Objectives

The project aims to:

* Perform exploratory data analysis
* Identify fraud patterns
* Handle highly imbalanced classes
* Engineer meaningful transaction-level features
* Develop an XGBoost fraud classifier
* Optimize model hyperparameters
* Evaluate model performance using fraud-specific metrics
* Optimize the classification threshold
* Explain individual fraud predictions
* Identify important fraud drivers
* Generate transaction-level fraud probabilities
* Build a reusable inference pipeline

---

# Machine Learning Approach

## 1. Data Ingestion

Load the PaySim transaction dataset and perform initial profiling.

Key checks:

* Dataset dimensions
* Data types
* Missing values
* Duplicate transactions
* Cardinality
* Class distribution
* Outliers
* Suspicious variables
* Potential leakage

---

## 2. Exploratory Data Analysis

Analyze fraud patterns across:

* Transaction type
* Transaction amount
* Time step
* Originator balance
* Destination balance
* Fraud flags
* Transaction frequency

Example questions:

```text
Which transaction types contain the highest fraud rate?

Does transaction amount correlate with fraud?

Are fraudulent transactions concentrated in particular transaction types?

What happens to account balances immediately before and after suspicious transactions?
```

---

# 3. Feature Engineering

The project should go beyond simply feeding the raw columns into XGBoost.

Potential engineered features include:

### Transaction Features

```text
log_amount
amount_to_origin_balance_ratio
amount_to_destination_balance_ratio
```

### Balance Consistency Features

```text
origin_balance_change
destination_balance_change
origin_balance_error
destination_balance_error
```

For example:

```text
origin_balance_change =
oldbalanceOrg - newbalanceOrig
```

and:

```text
destination_balance_change =
newbalanceDest - oldbalanceDest
```

### Temporal Features

From `step`:

```text
hour_bucket
time_window
transaction_period
```

### Customer / Account Features

Potential aggregation features:

```text
origin_transaction_count
destination_transaction_count
origin_total_amount
destination_total_amount
origin_average_amount
destination_average_amount
```

These features can help identify unusual transaction behaviour.

---

# 4. Class Imbalance

Fraud datasets are highly imbalanced.

The project should explicitly evaluate the class distribution:

```text
Legitimate >> Fraudulent
```

Possible approaches include:

* `scale_pos_weight`
* Class weighting
* Stratified sampling
* Threshold optimization
* Precision-recall analysis

Avoid blindly applying SMOTE without first evaluating whether synthetic oversampling is appropriate for the transaction structure.

---

# 5. XGBoost Model

The primary model is:

```text
XGBoost Classifier
```

Example configuration:

```python
XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1000,
    subsample=0.8,
    colsample_bytree=0.8
)
```

Hyperparameters should be tuned rather than treated as fixed production values.

Potential parameters to optimize:

```text
max_depth
learning_rate
n_estimators
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda
```

---

# 6. Model Evaluation

Accuracy is **not** the primary evaluation metric.

The project evaluates:

### ROC-AUC

Measures the model's ability to rank fraudulent transactions above legitimate transactions.

### PR-AUC

Particularly important for highly imbalanced fraud datasets.

### Precision

```text
TP / (TP + FP)
```

Answers:

> Of the transactions flagged as fraud, how many were actually fraudulent?

### Recall

```text
TP / (TP + FN)
```

Answers:

> Of all fraudulent transactions, how many did the model detect?

### F1 Score

Balances precision and recall.

### Confusion Matrix

```text
                 Predicted
               Legit   Fraud
Actual Legit     TN      FP
Actual Fraud     FN      TP
```

---

# 7. Fraud-Specific Metrics

Additional metrics should include:

```text
Fraud Capture Rate
False Positive Rate
Precision @ K
Recall @ K
Fraud Detection Rate
False Alarm Rate
```

Where appropriate, evaluate the model from an operational perspective.

For example:

> If investigators can review only the top 1% of transactions, how much fraud can the model capture?

---

# 8. Threshold Optimization

The default classification threshold:

```text
0.50
```

should not automatically be considered optimal.

The model produces:

```text
P(Fraud)
```

For example:

```text
Transaction A -> 0.02
Transaction B -> 0.18
Transaction C -> 0.73
Transaction D -> 0.94
```

The final decision threshold can be optimized based on the business objective.

Example:

```text
Probability >= 0.80
        |
        v
     FRAUD
```

while lower-risk transactions can be:

```text
Probability < 0.80
        |
        v
   LEGITIMATE
```

The threshold should be selected using validation data rather than the test set.

---

# 9. Business Cost Optimization

A more advanced version of the project should incorporate fraud investigation costs.

For example:

```text
False Negative Cost = 1000
False Positive Cost = 10
```

A cost function can then be used to determine the optimal operating threshold.

Conceptually:

```text
Total Cost =
(FN × Fraud Loss)
+
(FP × Investigation Cost)
```

This allows the model to move from:

> "Which model has the highest accuracy?"

to:

> "Which operating point minimizes business loss?"

---

# 10. Model Explainability

Use **SHAP** to explain model behaviour.

Global explanations:

```text
Which features drive fraud predictions?
```

Local explanations:

```text
Why was this transaction classified as fraudulent?
```

Example output:

```text
Transaction ID: XYZ

Fraud Probability: 0.94

Top Contributing Factors:

1. Transaction Amount
2. Transaction Type
3. Origin Balance Change
4. Destination Balance Change
5. Transaction Frequency
```

This is particularly useful for fraud analysts and model governance.

---

# 11. Model Interpretability

The project should provide:

* Feature importance
* SHAP summary plots
* SHAP dependence plots
* Individual transaction explanations
* Fraud probability distribution
* Precision-recall curve
* ROC curve
* Confusion matrix

---

# Project Architecture

```text
                    PaySim Dataset
                           |
                           v
                  Data Validation
                           |
                           v
                    Feature Pipeline
                           |
                           v
                Train / Validation / Test
                           |
                           v
                    XGBoost Training
                           |
                           v
                  Model Evaluation
                           |
             +-------------+-------------+
             |                           |
             v                           v
       SHAP Explainability        Threshold Optimization
             |                           |
             +-------------+-------------+
                           |
                           v
                   Fraud Risk Score
                           |
                           v
              Fraud / Review / Approve
```

---

# Recommended Repository Structure

```text
Fraud-Detection/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_xgboost_model.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── explainability.py
│
├── models/
│   └── .gitkeep
│
├── reports/
│   ├── figures/
│   └── model_metrics/
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_prediction.py
│
├── requirements.txt
├── .gitignore
├── README.md
└── LICENSE
```

---

# Technology Stack

| Component           | Technology                 |
| ------------------- | -------------------------- |
| Language            | Python                     |
| ML Framework        | XGBoost                    |
| Data Processing     | Pandas                     |
| Numerical Computing | NumPy                      |
| Visualization       | Matplotlib / Plotly        |
| Explainability      | SHAP                       |
| ML Utilities        | Scikit-learn               |
| Dataset             | PaySim                     |
| Development         | Jupyter Notebook / VS Code |
| Version Control     | Git / GitHub               |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/Fraud-Detection.git
cd Fraud-Detection
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Setup

Download the PaySim dataset from Kaggle:

https://www.kaggle.com/datasets/ealaxi/paysim1

Place the dataset under:

```text
data/
```

For example:

```text
data/
└── PS_20174392719_1491204439457_log.csv
```

**Do not commit the full dataset to GitHub.**

Add the dataset path to `.gitignore`.

---

# Model Training

Example:

```bash
python src/train.py
```

The training pipeline should:

1. Load the dataset
2. Validate the data
3. Perform preprocessing
4. Engineer features
5. Split the dataset
6. Train XGBoost
7. Evaluate the model
8. Save the trained model

---

# Prediction

Example:

```bash
python src/predict.py
```

Expected output:

```text
Transaction
-----------
Amount: 250000
Type: TRANSFER

Fraud Probability: 0.9234

Prediction: FRAUD
```

---

# Model Output

The final prediction interface should return:

```json
{
    "fraud_probability": 0.9234,
    "prediction": "FRAUD",
    "risk_level": "HIGH"
}
```

Possible risk bands:

```text
0.00 - 0.30  -> LOW
0.30 - 0.70  -> MEDIUM
0.70 - 1.00  -> HIGH
```

These thresholds are illustrative and should be calibrated based on model performance and business requirements.

---

# Fraud Detection Workflow

A production-inspired workflow can be represented as:

```text
Transaction
     |
     v
Feature Generation
     |
     v
XGBoost
     |
     v
Fraud Probability
     |
     +--------------------+
     |                    |
     v                    v
Low Risk              High Risk
     |                    |
     v                    v
Approve             Manual Review
                          |
                          v
                     Investigation
```

---

# Key Challenges

This project intentionally addresses several challenges encountered in real-world fraud analytics.

### Class Imbalance

Fraud is significantly less frequent than legitimate activity.

### False Positives

Aggressive fraud detection can result in legitimate customers being incorrectly flagged.

### False Negatives

Missing fraudulent transactions can result in direct financial loss.

### Data Leakage

Fraud models are particularly susceptible to leakage from future information or post-transaction variables.

### Temporal Drift

Fraud patterns can change over time.

### Explainability

Fraud analysts need to understand why a transaction was flagged.

### Threshold Selection

The best probability threshold depends on operational capacity and financial costs.

---

# Model Governance Considerations

For a production implementation, the following should also be considered:

* Feature lineage
* Data quality monitoring
* Model versioning
* Prediction logging
* Threshold versioning
* Drift monitoring
* Performance monitoring
* Explainability
* Auditability
* Champion/challenger models
* Retraining strategy

---

# Future Enhancements

Potential extensions include:

### Real-Time Fraud API

Expose the trained model using:

```text
FastAPI
```

Architecture:

```text
Transaction
    |
    v
FastAPI
    |
    v
Feature Engineering
    |
    v
XGBoost
    |
    v
Fraud Probability
```

### Fraud Monitoring Dashboard

Build a dashboard displaying:

* Transaction volume
* Fraud rate
* Fraud probability distribution
* High-risk transactions
* False positives
* Fraud capture rate
* Model drift

### Advanced Feature Engineering

Introduce:

* Transaction velocity
* Customer behavioural profiles
* Rolling-window statistics
* Device fingerprints
* Account relationship graphs
* Peer-group anomaly detection

### Advanced Models

Compare XGBoost against:

```text
LightGBM
Random Forest
Logistic Regression
CatBoost
Isolation Forest
Neural Networks
```

---

# Model Development Philosophy

The objective of this project is **not simply to maximize an ML metric**.

The goal is to build a fraud detection system that answers four questions:

```text
1. Can we identify fraud?

2. How many fraudulent transactions can we capture?

3. How many legitimate customers will we incorrectly flag?

4. Can we explain why a transaction was considered suspicious?
```

A successful fraud model must balance all four.

---

# Dataset Attribution

This project uses the PaySim dataset:

> Synthetic Financial Datasets for Fraud Detection

Dataset source:

https://www.kaggle.com/datasets/ealaxi/paysim1

The dataset is used for research, experimentation, education, and demonstration purposes.

---

# Reference Implementation

This project takes inspiration from the following public repository:

https://github.com/aadhitea/Fraud-Detection

The implementation in this repository is intended to be independently developed and extended with a stronger focus on:

* XGBoost
* Feature engineering
* Imbalanced learning
* Fraud-specific evaluation
* Threshold optimization
* SHAP explainability
* Reusable ML pipelines
* Production-oriented design

---

# Disclaimer

This project is intended for **educational, research, and demonstration purposes**.

PaySim is a synthetic dataset and does not represent actual customer transactions.

Model performance obtained from this dataset should not be directly interpreted as production fraud-detection performance.

---

# License

This repository's source code is provided for educational and research purposes.

Please review and comply with the licensing and usage terms of the original dataset before redistributing the dataset or derivative data.
