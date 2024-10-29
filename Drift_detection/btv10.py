import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load configuration from YAML
def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config





# Define Signal class to use dynamic thresholds from YAML configuration
class Signal:
    def __init__(self, current_price, buy_signal, sell_signal, stop_loss_signal):
        self.current_price = current_price
        self.buy_signal = eval(buy_signal["function"])
        self.sell_signal = eval(sell_signal["function"])
        self.stop_loss_signal = eval(stop_loss_signal["function"])
        self.buy_threshold = buy_signal["params"]["buy_threshold"]
        self.sell_threshold = sell_signal["params"]["sell_threshold"]
        self.stop_loss_threshold = stop_loss_signal["params"]["stop_loss_threshold"]

    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high, self.buy_threshold)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price, self.sell_threshold) if buy_price else False

    def generate_stop_loss_signal(self, buy_price):
        return self.stop_loss_signal(self.current_price, buy_price, self.stop_loss_threshold) if buy_price else False






import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load configuration from YAML
def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config





# Define Signal class to use dynamic thresholds from YAML configuration
class Signal:
    def __init__(self, current_price, buy_signal, sell_signal, stop_loss_signal):
        self.current_price = current_price
        self.buy_signal = eval(buy_signal["function"])
        self.sell_signal = eval(sell_signal["function"])
        self.stop_loss_signal = eval(stop_loss_signal["function"])
        self.buy_threshold = buy_signal["params"]["buy_threshold"]
        self.sell_threshold = sell_signal["params"]["sell_threshold"]
        self.stop_loss_threshold = stop_loss_signal["params"]["stop_loss_threshold"]

    def generate_buy_signal(self, predicted_high):
        return self.buy_signal(self.current_price, predicted_high, self.buy_threshold)

    def generate_sell_signal(self, buy_price):
        return self.sell_signal(self.current_price, buy_price, self.sell_threshold) if buy_price else False

    def generate_stop_loss_signal(self, buy_price):
        return self.stop_loss_signal(self.current_price, buy_price, self.stop_loss_threshold) if buy_price else False








from datetime import timedelta

class Strategy:
    def __init__(self, config):
        strategy_config = config["strategies"]["trading_strategy"]["params"]
        self.initial_cash = strategy_config["initial_cash"]
        self.position_size_factor = strategy_config["position_size_factor"]
        self.cash = self.initial_cash
        self.position = 0
        self.buy_price = None
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.portfolio_value = []
        self.current_price = None
        self.decisions = []
        self.X_test_df = None
        self.max_holding_time = strategy_config["max_holding_time"]
        self.last_trade_timestamp = None  # Track the timestamp of the last trade

    def make_decision(self, timestamp, prediction, action, current_price):
        """Log each decision with debugging information."""
        self.decisions.append({
            "timestamp": timestamp,
            "prediction": prediction,
            "action": action,
            "current_price": current_price,
            "actual_price": None,
        })



    def execute_trade(self, signal, future_timestamp, predicted_high, current_price):
        self.current_price = current_price

        # Buy decision
        if self.position == 0 and signal.generate_buy_signal(predicted_high):
            amount_to_invest = self.position_size_factor * self.cash
            shares_to_buy = amount_to_invest // self.current_price
            self.cash -= shares_to_buy * self.current_price
            self.position += shares_to_buy
            self.buy_price = self.current_price
            self.make_decision(future_timestamp, predicted_high, "buy", self.current_price)  # Log buy event

        # Sell decision
        elif self.position > 0 and signal.generate_sell_signal(self.buy_price):
            trade_value = self.position * self.current_price
            self.cash += trade_value
            profit_or_loss = trade_value - (self.position * self.buy_price)
            if profit_or_loss > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.total_trades += 1
            self.position = 0
            self.buy_price = None
            self.make_decision(future_timestamp, predicted_high, "sell", self.current_price)  # Log sell event

            # Optionally: Initiate short if strategy allows
            amount_to_invest = self.position_size_factor * self.cash
            shares_to_short = amount_to_invest // self.current_price
            self.cash += shares_to_short * self.current_price  # Short proceeds added to cash
            self.position -= shares_to_short

        # Buy to cover short position
        elif self.position < 0 and signal.generate_buy_signal(predicted_high):
            trade_value = abs(self.position) * self.current_price
            self.cash -= trade_value
            profit_or_loss = -trade_value
            if profit_or_loss > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.total_trades += 1
            self.position = 0

            # Re-buy
            amount_to_invest = self.position_size_factor * self.cash
            shares_to_buy = amount_to_invest // self.current_price
            self.cash -= shares_to_buy * self.current_price
            self.position += shares_to_buy
            self.buy_price = self.current_price
            self.make_decision(future_timestamp, predicted_high, "buy", self.current_price)  # Log re-buy event

            # Update portfolio value for the day
        self.portfolio_value.append(self.cash + self.position * self.current_price)



    def print_trade_statistics(self):
        """Print the total trades and winning trades."""
        print(f"Total Trades: {self.total_trades}, Winning Trades: {self.winning_trades}")

    def fetch_actual_price(self, timestamp):
        try:
            mask = self.X_test_df.index == timestamp
            actual_row = self.X_test_df.loc[mask].iloc[0] if not self.X_test_df.loc[mask].empty else None

            if actual_row is not None:
                return actual_row['high']
            else:
                return None
        except KeyError:
            return None

    def evaluate_decisions(self):
        for decision in self.decisions:
            actual_price = self.fetch_actual_price(decision["timestamp"])
            decision["actual_price"] = actual_price
            print(f"Evaluated Decision at {decision['timestamp']}: predicted {decision['prediction']}, "
                  f"actual {decision['actual_price']}, action: {decision['action']}")









class BackTesting:
    def __init__(self, framework):
        self.framework = framework
        if framework == 'pytorch':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("CUDA Available:", torch.cuda.is_available())
            self.data_structure = self.initialize_pytorch_structure()
        else:
            self.data_structure = self.initialize_other_structure()

    def initialize_pytorch_structure(self):
        return {"data": torch.tensor([])}

    def initialize_other_structure(self):
        return {"data": np.array([])}

    def run_backtest(self, data_file_path):
        model = Model(input_size=9, hidden_size=50, output_size=1, num_layers=1, device=self.device)
        model.load_data(data_file_path)
        model.prepare_data(time_step=60)
        model.create_datasets()
        model.train(model.X_train, model.y_train, model.X_test, model.y_test, epochs=100, batch_size=16, learning_rate=0.001, patience=8)
        self.predictions, self.y_test = model.evaluate(model.X_test, model.y_test, model.target_scaler)

        # Load YAML configuration
        config = load_config("./config.yaml")

        # Create signals
        buy_signal_config = config["signals"]["buy_signal"]
        sell_signal_config = config["signals"]["sell_signal"]
        stop_loss_signal_config = config["signals"]["stop_loss_signal"]

        # Create strategy
        strategy = Strategy(config)

        strategy.X_test_df = model.X_test_df  # Pass the DataFrame to the strategy
        for i in range(len(model.X_test)):
            current_price = model.data['close'].iloc[int(0.8 * len(model.data)) + i]
            predicted_high = self.predictions[i][0]
            current_timestamp = model.data.index[int(0.8 * len(model.data)) + i]
            future_timestamp = model.data.index[int(0.8 * len(model.data)) + i + 1]

            signal = Signal(current_price, buy_signal_config, sell_signal_config, stop_loss_signal_config)
            strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        strategy.evaluate_decisions()  # Evaluate and print decisions

        performance_metrics = PerformanceMetrics(model.data, strategy)
        performance_metrics.calculate_performance_metrics()








class PerformanceMetrics:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy

    def calculate_performance_metrics(self):
        import ffn
        # Prepare the data
        self.data['date'] = pd.to_datetime(self.data.index)
        self.data.set_index('date', inplace=True)

        # Create portfolio_series with the appropriate frequency
        portfolio_series = pd.Series(
            self.strategy.portfolio_value,
            index=pd.date_range(
                start=self.data.index[-len(self.strategy.portfolio_value)],
                periods=len(self.strategy.portfolio_value),
                freq='T'  # Set the appropriate frequency
            )
        )
        print('portfolio_series')
        print(portfolio_series)

        # Calculate hit ratio and strategy gain
        hit_ratio = (self.strategy.winning_trades / self.strategy.total_trades) * 100 if self.strategy.total_trades > 0 else 0
        strategy_gain = (self.strategy.portfolio_value[-1] - self.strategy.initial_cash) / self.strategy.initial_cash * 100

        print(f"Final portfolio value: ${self.strategy.portfolio_value[-1]:.2f}")
        print(f"Hit Ratio: {hit_ratio:.2f}%")
        print(f"Strategy Gain: {strategy_gain:.2f}%")

        # Plot portfolio value over time
        plt.plot(portfolio_series.index, portfolio_series, label='Portfolio Value')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (USD)')
        plt.title('Portfolio Value Over Time')
        plt.legend()
        plt.show()

        # Convert the Series to a DataFrame for ffn
        portfolio_df = pd.DataFrame(portfolio_series)
        print('dataframe')
        print(portfolio_df)
        print(portfolio_df.columns)
        portfolio_df['returns'] = portfolio_df[0].pct_change()
        max_drawdown = ffn.calc_max_drawdown(portfolio_series)
        print(f"Maximum Drawdown: {max_drawdown:.4%}" if max_drawdown is not None else "Maximum Drawdown: N/A")

        # Calculate Sharpe Ratio
        minute_returns = portfolio_df['returns'].dropna()
        risk_free_rate = 0.01 / (252 * 390) 
        excess_returns = minute_returns - risk_free_rate

        # Calculate annualized Sharpe Ratio
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(390)  # Annualized for minute returns
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252 * 390)  # Annualized downside deviation

        # Calculate annualized Sortino Ratio
        sortino_ratio = (excess_returns.mean() / downside_deviation)
        print(f"Sortino Ratio: {sortino_ratio:.2f}")







class Model:
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        self.device = device
        self.model = self.LSTMModel(input_size, hidden_size, output_size, num_layers, device).to(device)
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers, device):
            super(Model.LSTMModel, self).__init__()
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

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp').head(4000)
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
        self.features = self.feature_scaler.fit_transform(self.features)
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

        # Extract timestamps for the test set
        self.timestamps = self.data.index[split + self.time_step:]
        X_test_numpy = self.X_test.numpy()
        X_test_flat = X_test_numpy.reshape(-1, X_test_numpy.shape[-1])

        # Inverse transform the flattened X_test values
        X_test_flat_original = self.feature_scaler.inverse_transform(X_test_flat)

        # Repeat the timestamps to match the flattened shape
        timestamps_repeated = np.repeat(self.timestamps, self.X_test.shape[1])
        timestamps_repeated = timestamps_repeated[:X_test_flat_original.shape[0]]  # Ensure the lengths match

        # Create DataFrame with the inverse transformed X_test values and repeated timestamps
        self.X_test_df = pd.DataFrame(X_test_flat_original, columns=['open', 'high', 'low', 'close', 'volume', 'Return', 'Volatility', 'High_Low_Range', 'Prev_Close_Rel_High'], index=timestamps_repeated)

    def train(self, X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate, patience):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        train_loader = torch.utils.data.DataLoader(dataset=list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
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
                X_val = X_test.to(self.device)
                y_val = y_test.to(self.device)
                val_loss = criterion(self.model(X_val), y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), 'lstm_model.pth')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

    def evaluate(self, X_test, y_test, target_scaler):
        self.model.eval()
        with torch.no_grad():
            X_test = X_test.to(self.device)
            predictions = self.model(X_test).cpu().numpy()
            y_test = y_test.numpy()
        predictions = target_scaler.inverse_transform(predictions)
        y_test = target_scaler.inverse_transform(y_test)

        plt.plot(y_test, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.legend()
        plt.show()
        return predictions, y_test




# Usage
if __name__ == '__main__':
    bt = BackTesting('pytorch')
    #bt.run_backtest('../../datasets/FIXED_EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')
    bt.run_backtest('../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')












