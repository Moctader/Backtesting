# Model
################################################################
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

# 1. Data Preparation

# Load data
data = pd.read_csv('../drift/EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(10000)

# Prepare features and target
data['Next_High'] = data['high'].shift(-1)  # Target variable: next period's high
data['Return'] = data['close'].pct_change()  # Price returns
data['Volatility'] = data['Return'].rolling(window=5).std()  # 5-period rolling volatility
data['High_Low_Range'] = (data['high'] - data['low']) / data['low']  # Intraday high-low range
data['Prev_Close_Rel_High'] = (data['close'].shift(1) - data['high']) / data['high']  # Previous close relative to high
data.dropna(inplace=True)

# Normalize the features
feature_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = feature_scaler.fit_transform(data[['open', 'high', 'low', 'close', 'volume', 'Return', 'Volatility', 'High_Low_Range', 'Prev_Close_Rel_High']])

# Normalize the target
target_scaler = MinMaxScaler(feature_range=(0, 1))
scaled_target = target_scaler.fit_transform(data[['Next_High']])

# Create dataset function for LSTM
def create_dataset(features, target, time_step=1):
    X, y = [], []
    for i in range(len(features) - time_step - 1):
        X.append(features[i:(i + time_step), :])
        y.append(target[i + time_step])
    return np.array(X), np.array(y)

# Define time step for LSTM model
time_step = 60
X, y = create_dataset(scaled_features, scaled_target, time_step)

# Reshape data to fit PyTorch LSTM input: (samples, time_step, features)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

# Split data into training, validation, and testing sets
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)  # 10% for validation
X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Move tensors to the GPU if available
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
        
        out, _ = self.lstm(x, (h0, c0))  # LSTM output
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out

# Hyperparameters
input_size = X_train.shape[2]  # Number of features
hidden_size = 100
output_size = 1
num_layers = 2
num_epochs = 20
batch_size = 16
learning_rate = 0.01
patience = 10  # Early stopping patience

# Create LSTM model
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# DataLoader for batching
train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

# Lists to store the loss values for plotting
train_losses = []
val_losses = []

# Early stopping variables
best_val_loss = float('inf')
epochs_without_improvement = 0

# Train the model
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, (features, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the loss for each batch
        epoch_loss += loss.item()

    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()
        val_losses.append(val_loss)

    # Print the progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        
        # Save the model
        torch.save(model.state_dict(), 'lstm_model.pth')
        print("Model saved.")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs without improvement.')
            break

# 3. Model Evaluation

model.eval()  # Set model to evaluation mode

# Make predictions
with torch.no_grad():
    predictions = model(X_test)
    predictions = predictions.cpu().numpy()
    y_test = y_test.cpu().numpy()

# Inverse transform predictions and actual values to original scale
predictions = target_scaler.inverse_transform(predictions)
y_test = target_scaler.inverse_transform(y_test)

# 4. Plot results

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel('Next High Price')
plt.legend()
plt.show()

