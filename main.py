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
import yaml


# 1. Data Preparation

# Load data
data = pd.read_csv('../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(5000)

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
num_epochs = 100
batch_size = 16
learning_rate = 0.01
patience = 8  # Early stopping patience

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





############################################################################################################
# Trading Strategy

# Load trading strategy from YAML
with open('trading_strategy.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize parameters from YAML
initial_cash = config['strategies']['trading_strategy']['params']['initial_cash']
position_size_factor = config['strategies']['trading_strategy']['params']['position_size_factor']
max_holding_time = config['strategies']['trading_strategy']['params']['max_holding_time']

cash = initial_cash
position = 0
portfolio_value = []
trades = []
total_trades = 0
winning_trades = 0
losing_trades = 0
holding_time = 0

# Backtest parameters
trading_days = len(X_test)  
start_price = data['close'].iloc[train_size]
end_price = data['close'].iloc[-1]

# Backtest loop
for i in range(trading_days):
    current_price = data['close'].iloc[train_size + i]
    predicted_high = predictions[i][0]
    
    daily_portfolio_value = cash  # Start with cash
    print(f"Time {i}: Current Price = {current_price}, Predicted High = {predicted_high}")

    # Buy condition
    if position == 0 and eval(config['signals']['buy_signal']['function'])(
        current_price, predicted_high, config['signals']['buy_signal']['params']['buy_threshold']):
        
        amount_to_invest = position_size_factor * cash  # Invest 10% of cash
        shares_to_buy = amount_to_invest // current_price
        cash -= shares_to_buy * current_price
        position += shares_to_buy
        buy_price = current_price  # Store the buy price
        holding_time = 0  # Reset holding days after purchase
        print(f"Bought {shares_to_buy} shares at {current_price:.2f}")

    # If we are in position, check sell and stop-loss conditions
    if position > 0:
        holding_time += 1
        
        # Stop-loss condition
        if eval(config['signals']['stop_loss_signal']['function'])(
            current_price, buy_price, config['signals']['stop_loss_signal']['params']['stop_loss_threshold']):
            
            cash += position * current_price
            loss = (current_price - buy_price) * position
            trades.append(loss)  # Track the loss
            losing_trades += 1
            print(f"Stop-loss triggered. Sold {position} shares at {current_price:.2f}, Loss: {loss:.2f}")
            position = 0  # Reset position after selling
        
        # Sell condition
        elif eval(config['signals']['sell_signal']['function'])(
            current_price, predicted_high):
            
            cash += position * current_price
            profit = (current_price - buy_price) * position  # Profit based on the position
            trades.append(profit)  # Store the result of the trade
            winning_trades += 1
            print(f"Sold {position} shares at {current_price:.2f}, Profit: {profit:.2f}")
            position = 0  # Reset position after selling

        # Exit condition: hold too long
        elif holding_time >= max_holding_time:
            cash += position * current_price
            exit_profit = (current_price - buy_price) * position
            trades.append(exit_profit)
            if exit_profit > 0:
                winning_trades += 1
            else:
                losing_trades += 1
            print(f"Exited after holding for {max_holding_time} mins. Sold {position} shares at {current_price:.2f}, Profit: {exit_profit:.2f}")
            position = 0  # Reset position after selling

    total_trades += 1  # Track total trades

    # Update portfolio value
    daily_portfolio_value = cash + position * current_price
    portfolio_value.append(daily_portfolio_value)

# Calculate final portfolio stats
final_portfolio_value = portfolio_value[-1]
hit_ratio = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

# Print final stats
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
print(f"Total Trades: {total_trades}")
print(f"Winning Trades: {winning_trades}")
print(f"Losing Trades: {losing_trades}")
print(f"Win Rate: {hit_ratio:.2f}%")



############################################################################################################
# Performance Metrics
portfolio_series = pd.Series(portfolio_value, index=pd.date_range(start=data.index[-len(portfolio_value)], periods=len(portfolio_value)))
hit_ratio = (winning_trades / total_trades)  * 100 
buy_and_hold_gain = (end_price - start_price) / start_price * 100
strategy_gain = (portfolio_value[-1] - initial_cash) / initial_cash * 100
print(f"Final portfolio value: ${portfolio_value[-1]:.2f}")
print(f"Hit Ratio: {hit_ratio:.2f}%")
print(f"Buy-and-Hold Gain: {buy_and_hold_gain:.2f}%")
print(f"Strategy Gain: {strategy_gain:.2f}%")

# Plot portfolio value over time
plt.plot(portfolio_series.index, portfolio_value, label='Portfolio Value')
plt.xlabel('Minutes')
plt.ylabel('Portfolio Value (USD)')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()

# Display performance stats
perf = portfolio_series.calc_stats()
perf.display()




############################################################################################################

buy_and_hold_returns = data['close'].pct_change().iloc[train_size:train_size + trading_days]
buy_and_hold_cumulative_returns = (1 + buy_and_hold_returns).cumprod()
trading_strategy_returns = pd.Series(portfolio_value).pct_change()
trading_strategy_cumulative_returns = (1 + trading_strategy_returns).cumprod()
buy_and_hold_cumulative_returns = buy_and_hold_cumulative_returns[:len(trading_strategy_cumulative_returns)]
cumulative_difference = trading_strategy_cumulative_returns.values - buy_and_hold_cumulative_returns.values

# Plot cumulative difference
plt.plot(trading_strategy_cumulative_returns.index, cumulative_difference)
plt.xlabel('Minutes')
plt.ylabel('Cumulative Difference')
plt.title('Cumulative Difference Between Trading Strategy and Buy-and-Hold')
plt.legend()
plt.show()