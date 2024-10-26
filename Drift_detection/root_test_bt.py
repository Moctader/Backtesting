import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BackTesting:
    def __init__(self, framework):
        self.framework = framework
        
        # Set up data handling based on the framework
        if framework == 'pytorch':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.data_structure = self.initialize_pytorch_structure()
        else:
            self.data_structure = self.initialize_other_structure()

        # Initialize state for event-based functionality
        self.cash = 100000  # Initial cash
        self.position = 0  # Current position (positive for long, negative for short)
        self.portfolio_value = []
        self.predictions = {}  # Store predictions with timestamps
        self.actuals = {}  # Store actual values with timestamps

    def initialize_pytorch_structure(self):
        # Set up data structure specific to PyTorch
        return {"data": torch.tensor([])}  # Example

    def initialize_other_structure(self):
        # Set up some other data structure, e.g., NumPy or Pandas
        return {"data": np.array([])}  # Example

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data['log_return'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data.dropna(inplace=True)

    def prepare_data(self, time_step=60):
        self.time_step = time_step
        self.features = self.data[['log_return']].values
        self.targets = self.data['close'].shift(-1).dropna().values
        self.features = self.features[:-1]  # Align features and targets

    def create_datasets(self):
        X, y = [], []
        for i in range(len(self.features) - self.time_step):
            X.append(self.features[i:i + self.time_step])
            y.append(self.targets[i + self.time_step])
        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]

    def build_model(self):
        self.model = nn.Sequential(
            nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True),
            nn.Linear(50, 1)
        ).to(self.device)

    def train_model(self, epochs=10, batch_size=32, learning_rate=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.train()
        for epoch in range(epochs):
            for i in range(0, len(self.X_train), batch_size):
                X_batch = torch.tensor(self.X_train[i:i + batch_size], dtype=torch.float32).to(self.device)
                y_batch = torch.tensor(self.y_train[i:i + batch_size], dtype=torch.float32).to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            X_test = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            y_test = torch.tensor(self.y_test, dtype=torch.float32).to(self.device)
            predictions = self.model(X_test)
            mse = nn.MSELoss()(predictions, y_test)
            print(f'Test MSE: {mse.item()}')

    def backtest_strategy(self):
        self.model.eval()
        with torch.no_grad():
            X_test = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)
            predictions = self.model(X_test).cpu().numpy()
        self.data['predicted_close'] = np.nan
        self.data.iloc[-len(predictions):, self.data.columns.get_loc('predicted_close')] = predictions
        self.data['signal'] = 0
        self.data.loc[self.data['predicted_close'] > self.data['close'], 'signal'] = 1
        self.data.loc[self.data['predicted_close'] < self.data['close'], 'signal'] = -1
        self.data['strategy_return'] = self.data['signal'].shift(1) * self.data['log_return']
        self.data['cumulative_return'] = self.data['log_return'].cumsum().apply(np.exp)
        self.data['cumulative_strategy_return'] = self.data['strategy_return'].cumsum().apply(np.exp)
        self.plot_results()

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['cumulative_return'], label='Buy and Hold')
        plt.plot(self.data['cumulative_strategy_return'], label='Strategy')
        plt.legend()
        plt.show()

    def event(self, datetime_index, prediction_value):
        # Store the prediction
        self.predictions[datetime_index] = prediction_value

    def update_actual(self, datetime_index, actual_value):
        # Store the actual value
        self.actuals[datetime_index] = actual_value

        # Check if we have a prediction for this datetime_index
        if datetime_index in self.predictions:
            prediction_value = self.predictions[datetime_index]
            self.current_price = actual_value

            # Example strategy: Buy if prediction is higher than current price, sell if lower
            if prediction_value > self.current_price and self.position <= 0:
                # Buy signal
                amount_to_invest = 0.1 * self.cash  # Example: invest 10% of cash
                shares_to_buy = amount_to_invest // self.current_price
                self.cash -= shares_to_buy * self.current_price
                self.position += shares_to_buy
                print(f"Bought {shares_to_buy} shares at {self.current_price}. Cash left: {self.cash}")
            elif prediction_value < self.current_price and self.position > 0:
                # Sell signal
                self.cash += self.position * self.current_price
                print(f"Sold {self.position} shares at {self.current_price}. Cash now: {self.cash}")
                self.position = 0

            # Update portfolio value
            self.portfolio_value.append(self.cash + self.position * self.current_price)

# Example usage
bt = BackTesting('pytorch')
bt.load_data('historical_data.csv')
bt.prepare_data()
bt.create_datasets()
bt.build_model()
bt.train_model()
bt.evaluate_model()

# Simulate event-based backtesting
for i in range(len(bt.data)):
    datetime_index = bt.data.index[i]
    prediction_value = bt.data['predicted_close'].iloc[i] if 'predicted_close' in bt.data.columns else bt.data['close'].iloc[i]
    bt.event(datetime_index, prediction_value)

# Simulate updating actual values
for i in range(len(bt.data)):
    datetime_index = bt.data.index[i]
    actual_value = bt.data['close'].iloc[i]
    bt.update_actual(datetime_index, actual_value)

# Plot final portfolio value
plt.figure(figsize=(12, 6))
plt.plot(bt.portfolio_value, label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Time')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()