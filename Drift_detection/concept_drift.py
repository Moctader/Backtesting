import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ffn
import os
from frouros.detectors.concept_drift import DDM, DDMConfig
from frouros.metrics import PrequentialError


# 1. Data Preparation
ticker = 'AAPL'
data = data = pd.read_csv('../../datasets/FIXED_EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(40000)

# Prepare features and target
data['Next_High'] = data['high'].shift(-1)
data['Return'] = data['close'].pct_change()
data['Volatility'] = data['Return'].rolling(window=5).std()
data['High_Low_Range'] = (data['high'] - data['low']) / data['low']
data['Prev_Close_Rel_High'] = (data['close'].shift(1) - data['high']) / data['high']
data.dropna(inplace=True)

# Normalize features and target
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume', 'Return', 'Volatility', 'High_Low_Range', 'Prev_Close_Rel_High']])
target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_target = target_scaler.fit_transform(data[['Next_High']])

# LSTM dataset function
def create_dataset(features, target, time_step=1):
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step), :])
        y.append(target[i + time_step])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_features, scaled_target, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Split data
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)
X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

# 2. Build and Train the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

input_size = X_train.shape[2]
hidden_size = 100
output_size = 1
num_layers = 2
num_epochs = 100
batch_size = 32
learning_rate = 0.001
patience = 8

model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (features, labels) in enumerate(train_loader):
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'lstm_model.pth')
        print("Model saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs without improvement.')
            break

        

# 3. DDM Drift Detection and Stream Test
config = DDM(warning_level=2.0, drift_level=3.0, min_num_instances=25)
detector = DDM(config=config)
metric = PrequentialError(alpha=1.0)

def stream_test(X_test, y_test, metric, detector):
    drift_flag = False
    for i, (X, y) in enumerate(zip(X_test, y_test)):
        y_pred = model(X.unsqueeze(0).to(device)).cpu()
        error = float(abs(y_pred - y))
        metric_error = metric.update(error)
        detector.update(error)
        if detector.drift_detected and not drift_flag:
            drift_flag = True
            print(f"Concept drift detected at step {i}. Accuracy: {1 - metric_error:.4f}")
    if not drift_flag:
        print("No concept drift detected.")
    print(f"Final accuracy: {1 - metric_error:.4f}\n")

# Run stream test
stream_test(X_test, y_test, metric, detector)
