import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class BackTesting:
    def __init__(self, framework):
        self.framework = framework
        if framework == 'pytorch':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("CUDA Available:", torch.cuda.is_available())
            self.data_structure = self.initialize_pytorch_structure()
        else:
            self.data_structure = self.initialize_other_structure()

        # Trading and tracking parameters
        self.cash = 100000
        self.position = 0
        self.portfolio_value = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.buy_price = None

    def initialize_pytorch_structure(self):
        return {"data": torch.tensor([])}

    def initialize_other_structure(self):
        return {"data": np.array([])}

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path).head(4000)
        self.data['log_return'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data['Next_High'] = self.data['high'].shift(-1)
        self.data['Return'] = self.data['close'].pct_change()
        self.data['Volatility'] = self.data['Return'].rolling(window=5).std()
        self.data['High_Low_Range'] = (self.data['high'] - self.data['low']) / self.data['low']
        self.data['Prev_Close_Rel_High'] = (self.data['close'].shift(1) - self.data['high']) / self.data['high']
        self.data.dropna(inplace=True)
        return self.data

    def prepare_data(self, time_step=60):
        self.time_step = time_step
        self.features = self.data[['open', 'high', 'low', 'close', 'volume', 'Return', 'Volatility', 'High_Low_Range', 'Prev_Close_Rel_High']].values
        self.targets = self.data['Next_High'].values
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.features = self.feature_scaler.fit_transform(self.features)
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.targets = self.target_scaler.fit_transform(self.targets.reshape(-1, 1))

    def create_datasets(self):
        X, y = [], []
        for i in range(len(self.features) - self.time_step):
            X.append(self.features[i:i + self.time_step])
            y.append(self.targets[i + self.time_step])
        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))
        self.X_train = torch.tensor(X[:split], dtype=torch.float32)
        self.X_test = torch.tensor(X[split:], dtype=torch.float32)
        self.y_train = torch.tensor(y[:split], dtype=torch.float32)
        self.y_test = torch.tensor(y[split:], dtype=torch.float32)

    def build_model(self):
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size, num_layers, device):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.device = device
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        self.model = LSTMModel(input_size=self.X_train.shape[2], hidden_size=100, output_size=1, num_layers=2, device=self.device).to(self.device)
            
    def train_model(self, epochs, batch_size, learning_rate, patience):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        train_loader = torch.utils.data.DataLoader(dataset=list(zip(self.X_train, self.y_train)), batch_size=batch_size, shuffle=True)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
    
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            self.model.eval()
            with torch.no_grad():
                X_val = self.X_test.to(self.device)
                y_val = self.y_test.to(self.device)
                val_loss = criterion(self.model(X_val), y_val).item()
    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), 'lstm_model.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    def evaluate_model(self):
        self.model.eval()
        with torch.no_grad():
            X_test = self.X_test.to(self.device)
            predictions = self.model(X_test).cpu().numpy()
            y_test = self.y_test.numpy()
        self.predictions = self.target_scaler.inverse_transform(predictions)
        self.y_test = self.target_scaler.inverse_transform(y_test)
        plt.plot(y_test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.show()

    def optimized_trading_strategy(self):
        self.initial_cash = 100000  
        position = 0  
        cash = self.initial_cash
        self.portfolio_value = []  
        trades = []  
        self.total_trades = 0  
        self.winning_trades = 0  
        self.losing_trades = 0  
        buy_threshold = 0.000
        sell_threshold = 0.000
        position_size_factor = 0.3
        trading_days = len(self.X_test)  
        self.start_price= self.data['close'].iloc[int(0.8 * len(self.data))]
        self.end_price  = self.data['close'].iloc[-1]
        buy_price = None

        # Record start and end prices for buy-and-hold strategy
        self.start_price= self.data['close'].iloc[int(0.8 * len(self.data))]
        self.end_price  = self.data['close'].iloc[-1]

        for i in range(trading_days):
            current_price = self.data['close'].iloc[int(0.8 * len(self.data)) + i]
            predicted_high = self.predictions[i][0]
            buy_signal = current_price < predicted_high * (1 + buy_threshold)
            sell_signal = (current_price > (buy_price * sell_threshold)) if buy_price else False

            #print(f"Day {i}: Current Price = {current_price:.2f}, Predicted High = {predicted_high:.2f}")


            if position == 0 and buy_signal:
                amount_to_invest = position_size_factor * cash
                shares_to_buy = amount_to_invest // current_price
                cash -= shares_to_buy * current_price
                position += shares_to_buy
                buy_price = current_price

            elif position > 0 and sell_signal:
                trade_value = position * current_price
                cash += trade_value
                profit_or_loss = trade_value - (position * buy_price)
                trades.append(profit_or_loss)
                if profit_or_loss > 0:
                    self.winning_trades  += 1
                else:
                    self.losing_trades += 1
                self.total_trades += 1
                position = 0
                buy_price = None


                amount_to_invest = position_size_factor * cash
                shares_to_short = amount_to_invest // current_price
                cash += shares_to_short * current_price  # Short proceeds added to cash
                position -= shares_to_short  

            elif position < 0 and buy_signal:
                trade_value = abs(position) * current_price
                cash -= trade_value
                profit_or_loss = -trade_value
                trades.append(profit_or_loss)
                if profit_or_loss > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                self.total_trades += 1
                position = 0
                
                amount_to_invest = position_size_factor * cash
                shares_to_buy = amount_to_invest // current_price
                cash -= shares_to_buy * current_price
                position += shares_to_buy
                buy_price = current_price

            self.portfolio_value.append(cash + position * current_price)

        final_portfolio_value = self.portfolio_value[-1]
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Total Trades Executed: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades }")
        print(f"Losing Trades: {self.losing_trades}")


    def calculate_performance_metrics(self):
        portfolio_series = pd.Series(self.portfolio_value)
        print(self.winning_trades )
        print(self.total_trades)
        hit_ratio = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        buy_and_hold_gain = (self.end_price - self.start_price) / self.start_price * 100
        strategy_gain = (self.portfolio_value[-1] - self.initial_cash) / self.initial_cash * 100

        print(f"Final portfolio value: ${self.portfolio_value[-1]:.2f}")
        print(f"Hit Ratio: {hit_ratio:.2f}%")
        print(f"Buy-and-Hold Gain: {buy_and_hold_gain:.2f}%")
        print(f"Strategy Gain: {strategy_gain:.2f}%")

        plt.plot(portfolio_series.index, self.portfolio_value, label='Portfolio Value')
        plt.xlabel('Minutes')
        plt.ylabel('Portfolio Value (USD)')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.show()

        

    def run_backtest(self, data_file_path):
        self.load_data(data_file_path)
        self.prepare_data(time_step=60)
        self.create_datasets()
        self.build_model()
        self.train_model(epochs=100, batch_size=32, learning_rate=0.001, patience=8)
        self.evaluate_model()
        self.optimized_trading_strategy()
        self.calculate_performance_metrics()

# Usage
if __name__ == '__main__':
    bt = BackTesting('pytorch')
    bt.run_backtest('../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
