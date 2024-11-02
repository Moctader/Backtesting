import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional

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
        self.previous_signal = None
        self.signal_strength = 1

    def make_decision(self, timestamp: pd.Timestamp, prediction: float, action: str, current_price: float):
        """Log each decision with debugging information."""
        self.decisions.append({
            "timestamp": timestamp,
            "prediction": prediction,
            "action": action,
            "current_price": current_price,
            "actual_price": None,
        })

    def buy_position(self, amount_to_invest: float):
        """Execute a buy by calculating shares and updating cash."""
        shares_to_buy = amount_to_invest // self.current_price
        self.cash -= shares_to_buy * self.current_price
        self.position += shares_to_buy
        self.buy_price = self.current_price

    def sell_position(self) -> float:
        """Execute a sell and return the profit or loss."""
        trade_value = self.position * self.current_price
        profit_or_loss = trade_value - (self.position * self.buy_price)
        self.cash += trade_value
        self.position = 0
        self.buy_price = None
        return profit_or_loss

    def execute_trade(self, signal, future_timestamp: pd.Timestamp, predicted_high: float, current_price: float):
        """Execute trades based on signals and manage positions."""
        self.current_price = current_price

        # Determine the signal strength
        if self.previous_signal == signal:
            self.signal_strength = min(self.signal_strength + 1, 3)  # Cap signal strength at 3
        else:
            self.signal_strength = 1
        self.previous_signal = signal

        # Decide on trade action
        amount_to_invest = self.position_size_factor * self.cash * self.signal_strength

        if self.position == 0 and signal.generate_buy_signal(predicted_high):
            # Initial Buy
            self.buy_position(amount_to_invest)
            self.make_decision(future_timestamp, predicted_high, "buy", self.current_price)

        elif self.position > 0 and signal.generate_buy_signal(predicted_high):
            # Add to Position
            self.buy_position(amount_to_invest)
            self.make_decision(future_timestamp, predicted_high, "buy", self.current_price)

        elif self.position > 0 and signal.generate_sell_signal(self.buy_price):
            # Sell Position
            profit_or_loss = self.sell_position()
            if profit_or_loss > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            self.total_trades += 1
            self.make_decision(future_timestamp, predicted_high, "sell", self.current_price)

        elif self.position > 0 and not signal.generate_buy_signal(predicted_high) and not signal.generate_sell_signal(self.buy_price):
            # Exit Position
            self.sell_position()
            self.total_trades += 1
            self.make_decision(future_timestamp, predicted_high, "exit", self.current_price)

        # Update portfolio value
        self.portfolio_value.append(self.cash + self.position * self.current_price)

    def print_trade_statistics(self):
        """Print the total trades and winning trades."""
        print(f"Total Trades: {self.total_trades}, Winning Trades: {self.winning_trades}")

    def fetch_actual_price(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Fetch the actual price for a given timestamp from test data."""
        try:
            mask = self.X_test_df.index == timestamp
            actual_row = self.X_test_df.loc[mask].iloc[0] if not self.X_test_df.loc[mask].empty else None
            return actual_row['high'] if actual_row is not None else None
        except KeyError:
            return None

    def evaluate_decisions(self):
        """Evaluate decisions by comparing predicted vs. actual price changes and display confusion matrix."""
        y_true, y_pred = [], []
        for decision in self.decisions:
            actual_price = self.fetch_actual_price(decision["timestamp"])
            decision["actual_price"] = actual_price
            print(f"Evaluated Decision at {decision['timestamp']}: predicted {decision['prediction']}, "
                  f"actual {decision['actual_price']}, action: {decision['action']}")
            if actual_price is not None:
                if decision["action"] == "buy":
                    y_pred.append(1)  # Predicted as buy
                    y_true.append(1 if actual_price > decision["current_price"] else 0)  # Actual buy if price increased
                elif decision["action"] == "sell":
                    y_pred.append(0)  # Predicted as sell
                    y_true.append(0 if actual_price < decision["current_price"] else 1)  # Actual sell if price decreased

        # Calculate and display confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Buy", "Sell"]).plot()
        plt.title("Confusion Matrix of Trading Decisions")
        plt.show()
        print("Confusion Matrix:")
        print(cm)

class Signal:
    def __init__(self, current_price: float, buy_signal: dict, sell_signal: dict):
        self.current_price = current_price
        self.buy_signal = eval(buy_signal["function"])
        self.sell_signal = eval(sell_signal["function"])
        self.buy_threshold = buy_signal["params"]["buy_threshold"]
        self.sell_threshold = sell_signal["params"]["sell_threshold"]

    def generate_buy_signal(self, predicted_high: float) -> bool:
        """Generate a buy signal based on prediction and threshold."""
        return self.buy_signal(self.current_price, predicted_high, self.buy_threshold)

    def generate_sell_signal(self, buy_price: Optional[float]) -> bool:
        """Generate a sell signal based on current price and buy price."""
        return self.sell_signal(self.current_price, buy_price, self.sell_threshold) if buy_price else False

# Example configuration usage
config = {
    "strategies": {
        "trading_strategy": {
            "params": {
                "initial_cash": 10000,
                "position_size_factor": 0.1,
                # other strategy parameters as needed
            }
        }
    }
}

# Example usage:
data = pd.read_csv('../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv').head(1000)
data['Next_High'] = data['high'].shift(-1)
data['MA3'] = data['Next_High'].rolling(window=3).mean()

# Initialize strategy
strategy = Strategy(config)

# Create signals
buy_signal_config = {
    "function": "lambda current_price, predicted_high, threshold: predicted_high > current_price * (1 + threshold)",
    "params": {"buy_threshold": 0.01}
}
sell_signal_config = {
    "function": "lambda current_price, buy_price, threshold: current_price < buy_price * (1 - threshold)",
    "params": {"sell_threshold": 0.01}
}

# Run backtest
for i in range(len(data) - int(0.8 * len(data))-1):
    current_price = data['close'].iloc[int(0.8 * len(data)) + i]
    predicted_high = data['MA3'].iloc[int(0.8 * len(data)) + i]
    current_timestamp = data.index[int(0.8 * len(data)) + i]
    future_timestamp = data.index[int(0.8 * len(data)) + i + 1]

    signal = Signal(current_price, buy_signal_config, sell_signal_config)
    strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

# Evaluate decisions
strategy.evaluate_decisions()