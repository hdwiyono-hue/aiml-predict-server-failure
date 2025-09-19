# aiml-predict-server-failure
Server Failure Prediction Pipeline

A machine learning pipeline for predicting server failures using multiple approaches:

RandomForest (classification) with feature importance analysis

IsolationForest (unsupervised anomaly detection)

LSTM Autoencoder (deep learning anomaly detection)

The pipeline compares models using accuracy, ROC AUC, and feature importance, and visualizes results with bar charts and ROC curves.

🚀 Features

Synthetic data generation simulating server health metrics (temperature, voltage, memory usage, disk I/O, network traffic)

Handles class imbalance with SMOTE

Trains and evaluates three models:

RandomForest

IsolationForest

LSTM Autoencoder

Visualizations:

Feature importance (RandomForest)

ROC Curves with model predictive ability

Model accuracy comparison

Outputs classification reports and key metrics

📊 Example Output
RandomForest Classification Report
precision    recall  f1-score   support

0 (healthy)   0.91      0.79      0.85      1700
1 (failure)   0.32      0.58      0.41       300

Accuracy: 0.76  
ROC AUC: 0.75

Feature Importance (RandomForest)

Voltage (32%)

Temperature (19%)

Network Traffic (17%)

Disk I/O (16%)

Memory Usage (15%)

ROC Curve

Shows predictive ability of each model beyond raw accuracy. RandomForest outperforms others (AUC ~0.75).

📦 Installation
# Clone repository
git clone https://github.com/your-username/server-failure-prediction.git
cd server-failure-prediction

# Install dependencies
pip install -r requirements.txt

Requirements

Python 3.9+

scikit-learn

imbalanced-learn

matplotlib

seaborn

torch

▶️ Usage

Run the pipeline:

python server_failure_pipeline_v7e.py


This will:

Generate synthetic server data

Train/evaluate models

Print classification reports

Plot ROC curves and feature importance

📈 Model Comparison
Model	Accuracy	ROC AUC	Notes
RandomForest	~0.76	~0.75	Best trade-off, interpretable features
IsolationForest	~0.76	~0.59	Detects anomalies but less predictive
LSTM Autoencoder	~0.53	~0.50	Near random baseline
🌟 Future Work

Use real server logs instead of synthetic data

Hyperparameter tuning for RandomForest and LSTM

Add early warning dashboards

Extend to multivariate time series prediction
