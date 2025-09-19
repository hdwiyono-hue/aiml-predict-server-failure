# server_failure_pipeline_v7e.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

# -----------------------
# LSTM Autoencoder
# -----------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_size=16):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=n_features, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h_repeated = h.repeat(x.size(1), 1, 1).permute(1,0,2)
        out, _ = self.decoder(h_repeated)
        return out

# -----------------------
# Synthetic server data
# -----------------------
def generate_synthetic_data(num_samples=10000, failure_rate=0.15):
    np.random.seed(42)
    df = pd.DataFrame({
        'temperature': np.random.normal(70, 10, num_samples),
        'voltage': np.random.normal(220, 5, num_samples),
        'memory_usage': np.random.normal(50, 20, num_samples),
        'disk_io': np.random.normal(100, 50, num_samples),
        'network_traffic': np.random.normal(1000, 300, num_samples),
    })
    df['failure'] = 0
    failure_indices = np.random.choice(df.index, int(num_samples*failure_rate), replace=False)
    df.loc[failure_indices, 'failure'] = 1
    df.loc[df['failure']==1, ['temperature', 'voltage', 'memory_usage']] += np.random.normal(5,2,(len(failure_indices),3))
    return df

# -----------------------
# RandomForest
# -----------------------
def train_random_forest(X_train, y_train, X_test, y_test):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_res, y_res)

    y_pred = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]   # probabilities
    roc_auc = roc_auc_score(y_test, rf_probs)
    acc = (y_pred == y_test).mean()

    print("Classification report (RandomForest test set):")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc:.3f}, Accuracy: {acc:.3f}")

    # Feature importances (%)
    rf_feat = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_})
    rf_feat['importance_pct'] = rf_feat['importance'] * 100
    rf_feat.sort_values(by='importance', ascending=False, inplace=True)
    print("\n--- RandomForest Feature Importances ---")
    print(rf_feat)

    # Plot feature importances
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x='importance_pct', y='feature', data=rf_feat, palette='viridis')
    for i, (val, feat) in enumerate(zip(rf_feat['importance_pct'], rf_feat['feature'])):
        ax.text(val + 0.5, i, f"{val:.1f}%", va='center')
    plt.title("RandomForest Feature Importances (%)")
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return rf_model, rf_probs, acc, roc_auc

# -----------------------
# ROC plot for all models
# -----------------------
def plot_model_roc(y_test, rf_probs, iso_scores, lstm_errors):
    plt.figure(figsize=(8,6))

    # RandomForest
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    auc_rf = auc(fpr_rf, tpr_rf)
    plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC={auc_rf:.3f})', color='blue')

    # IsolationForest
    iso_probs = (iso_scores.max() - iso_scores) / (iso_scores.max() - iso_scores.min())
    fpr_iso, tpr_iso, _ = roc_curve(y_test, iso_probs)
    auc_iso = auc(fpr_iso, tpr_iso)
    plt.plot(fpr_iso, tpr_iso, label=f'IsolationForest (AUC={auc_iso:.3f})', color='red')

    # LSTM Autoencoder
    lstm_probs = lstm_errors / lstm_errors.max()
    fpr_lstm, tpr_lstm, _ = roc_curve(y_test, lstm_probs)
    auc_lstm = auc(fpr_lstm, tpr_lstm)
    plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM Autoencoder (AUC={auc_lstm:.3f})', color='green')

    # Random chance line
    plt.plot([0,1],[0,1],'k--',label='Random Chance (AUC=0.5)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model Predictive Ability (ROC Curves)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# -----------------------
# Main pipeline
# -----------------------
def main():
    df = generate_synthetic_data(num_samples=10000, failure_rate=0.15)
    features = ['temperature', 'voltage', 'memory_usage', 'disk_io', 'network_traffic']
    X = df[features]
    y = df['failure']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # RandomForest
    rf_model, rf_probs, rf_acc, rf_roc = train_random_forest(X_train, y_train, X_test, y_test)

    # IsolationForest
    iso_model = IsolationForest(contamination=0.15, random_state=42)
    iso_model.fit(X_train)
    iso_scores = iso_model.decision_function(X_test)
    iso_pred = iso_model.predict(X_test)
    iso_pred = (iso_pred == -1).astype(int)
    iso_acc = (iso_pred == y_test).mean()
    iso_roc = roc_auc_score(y_test, (iso_scores.max() - iso_scores)/(iso_scores.max() - iso_scores.min()))
    print(f"IsolationForest Accuracy: {iso_acc:.3f}, ROC AUC: {iso_roc:.3f}")

    # LSTM Autoencoder
    seqs = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1)
    lstm_model = LSTMAutoencoder(seq_len=1, n_features=X.shape[1])
    lstm_model.train()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    X_train_seq = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)
    for epoch in range(5):
        optimizer.zero_grad()
        recon = lstm_model(X_train_seq)
        loss = loss_fn(recon, X_train_seq)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/5, loss: {loss.item():.4f}")
    lstm_model.eval()
    recon_test = lstm_model(seqs)
    lstm_errors = ((recon_test.detach().numpy() - seqs.numpy())**2).mean(axis=(1,2))
    lstm_acc = ((lstm_errors > lstm_errors.mean()).astype(int) == y_test.values).mean()
    lstm_roc = roc_auc_score(y_test, lstm_errors/lstm_errors.max())
    print(f"LSTM Autoencoder Accuracy: {lstm_acc:.3f}, ROC AUC: {lstm_roc:.3f}")

    # ROC comparison plot
    plot_model_roc(y_test, rf_probs, iso_scores, lstm_errors)

if __name__ == "__main__":
    main()
