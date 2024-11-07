import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ffn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional
from abc import ABC, abstractmethod
from signals.binary_signal import BinarySignal
from signals.binary_plus_exit_signal import BinaryPlusExitSignal

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class BaseStrategy(ABC):
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
        self.previous_signal = None
        self.signal_strength = 1
        self.signal = None
        self.X_test_df = None  # Initialize X_test_df as an attribute

    def make_decision(self, timestamp, prediction, action, current_price):
        """Log each decision with debugging information."""
        self.decisions.append({
            "timestamp": timestamp,
            "prediction": prediction,
            "action": action,
            "current_price": current_price,
            "actual_price": None,
        })

    def buy(self, future_timestamp, predicted_high):
        amount_to_invest = self.position_size_factor * self.cash * self.signal_strength
        shares_to_buy = amount_to_invest // self.current_price
        self.cash -= shares_to_buy * self.current_price
        self.position += shares_to_buy
        self.buy_price = self.current_price
        self.total_trades += 1  # Update total trades count
        self.make_decision(future_timestamp, predicted_high, "buy", self.current_price)  # Log buy event

    def sell(self, future_timestamp, predicted_high):
        trade_value = self.position * self.current_price
        self.cash += trade_value
        profit_or_loss = trade_value - (self.position * self.buy_price)
        self.track_trade_outcome(profit_or_loss)
        self.position = 0
        self.buy_price = None
        self.total_trades += 1  # Update total trades count
        self.make_decision(future_timestamp, predicted_high, "sell", self.current_price)  # Log sell event

    def exit(self, future_timestamp, predicted_high):
        """Execute an exit decision."""
        trade_value = self.position * self.current_price
        self.cash += trade_value
        self.total_trades += 1
        self.position = 0
        self.buy_price = None
        self.make_decision(future_timestamp, predicted_high, "exit", self.current_price)  

    def buy_to_cover(self, future_timestamp, predicted_high):
        trade_value = abs(self.position) * self.current_price
        self.cash -= trade_value
        profit_or_loss = -trade_value
        self.track_trade_outcome(profit_or_loss)
        self.position = 0
        self.total_trades += 1  # Update total trades count
        self.make_decision(future_timestamp, predicted_high, "buy_to_cover", self.current_price)  # Log buy to cover event

    def update_portfolio_value(self):
        portfolio_value = self.cash + self.position * self.current_price
        self.portfolio_value.append(portfolio_value)
        #logging.debug(f"Updated portfolio value: {portfolio_value}")

    def initiate_short_position(self):
        amount_to_invest = self.position_size_factor * self.cash
        shares_to_short = amount_to_invest // self.current_price
        self.total_trades += 1
        self.cash += shares_to_short * self.current_price  # Short proceeds added to cash
        self.position -= shares_to_short

    def track_trade_outcome(self, profit_or_loss):
        """Track the outcome of a trade."""
        if profit_or_loss > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

    def generate_signals(self, signal, predicted_high):
        """Generate buy, sell, and exit signals."""
        buy_signal = signal.generate_buy_signal(predicted_high)
        sell_signal = False
        if self.buy_price is not None:
            sell_signal = signal.generate_sell_signal(self.buy_price)
        logging.debug(f"self.current_price={self.current_price}, self.buy_price={self.buy_price}, predicted_high={predicted_high}, buy_signal={buy_signal}, sell_signal={sell_signal}")
        
        # Only generate exit signal if the signal is not an instance of BinarySignal
        if isinstance(signal, BinarySignal):
            exit_signal = False
            return buy_signal, sell_signal
        else:
            exit_signal = signal.generate_exit_signal(self.buy_price, predicted_high) if hasattr(signal, 'generate_exit_signal') else False
        
        return buy_signal, sell_signal, exit_signal

    @abstractmethod
    def execute_trade(self, signal, future_timestamp, predicted_high, current_price):
        """Execute trades based on signals and manage positions."""
        pass

    def fetch_actual_price(self, timestamp):
        """Fetch the actual price for a given timestamp."""
        try:
            actual_row = self.X_test_df.loc[timestamp] if timestamp in self.X_test_df.index else None
            if actual_row is not None:
                if 'high' in actual_row:
                    if pd.isna(actual_row['high']):
                        logging.debug(f"'high' column is NaN for timestamp: {timestamp}")
                        return None
                    else:
                        return actual_row['high']
                else:
                    logging.debug(f"'high' column not found in actual_row for timestamp: {timestamp}")
                    return None
            else:
                logging.debug(f"No actual row found for this {timestamp}.")
                logging.debug(f"Data around the timestamp {timestamp}:")
                return None
        except KeyError:
            logging.debug(f"KeyError: No data found for timestamp: {timestamp}")
            return None
        except IndexError:
            logging.debug(f"IndexError: No data found for timestamp: {timestamp}")
            return None

    def evaluate_decisions(self):
        """Evaluate the decisions made by the strategy."""
        y_true_signal = []
        y_pred_signal = []
        y_true_trade = []
        y_pred_trade = []

        for decision in self.decisions:
            actual_price = self.fetch_actual_price(decision["timestamp"])
            decision["actual_price"] = actual_price

            if actual_price is not None:
                # Evaluate signals
                if decision["action"] == "buy":
                    y_pred_signal.append(1)  # Predicted buy signal
                    y_true_signal.append(1 if decision["prediction"] > decision["current_price"] else 0)  # Actual buy signal if price increased
                elif decision["action"] == "sell":
                    y_pred_signal.append(0)  # Predicted sell signal
                    y_true_signal.append(0 if decision["prediction"] < decision["current_price"] else 1)  # Actual sell signal if price decreased
                
                # Evaluate trade outcomes
                if decision["action"] == "buy":
                    y_pred_trade.append(1)  # Predicted buy signal
                    y_true_trade.append(1 if actual_price > decision["current_price"] else 0)  # Actual buy signal if price increased
                elif decision["action"] == "sell":
                    y_pred_trade.append(0)  # Predicted sell signal
                    y_true_trade.append(0 if actual_price < decision["current_price"] else 1)  # Actual sell signal if price decreased

        self._plot_confusion_matrices(y_true_signal, y_pred_signal, y_true_trade, y_pred_trade)

    def _plot_confusion_matrices(self, y_true_signal, y_pred_signal, y_true_trade, y_pred_trade):
        """Plot confusion matrices for signals and trade outcomes."""
        # Signal Confusion Matrix
        cm_signal = confusion_matrix(y_true_signal, y_pred_signal, labels=[1, 0])
        disp_signal = ConfusionMatrixDisplay(confusion_matrix=cm_signal, display_labels=["Buy_signal", "Sell_signal"])
        disp_signal.plot()
        plt.title("Signal Confusion Matrix")
        plt.show()

        logging.debug("Signal Confusion Matrix:")
        logging.debug(cm_signal)

        # Trade Outcome Confusion Matrix
        cm_trade = confusion_matrix(y_true_trade, y_pred_trade, labels=[1, 0])
        disp_trade = ConfusionMatrixDisplay(confusion_matrix=cm_trade, display_labels=["Buy", "Sell"])
        disp_trade.plot()
        plt.title("Trade Outcome Confusion Matrix")
        plt.show()

        logging.debug("Trade Outcome Confusion Matrix:")
        logging.debug(cm_trade)

        # Relative Signal Confusion Matrix
        cm_signal_relative = cm_signal.astype('float') / cm_signal.sum(axis=1)[:, np.newaxis]
        logging.debug("Relative Signal Confusion Matrix:")
        logging.debug(cm_signal_relative)
        disp_signal_relative = ConfusionMatrixDisplay(confusion_matrix=cm_signal_relative, display_labels=["Buy", "Sell"])
        disp_signal_relative.plot()
        plt.title("Relative Signal Confusion Matrix")
        plt.show()

        # Relative Trade Outcome Confusion Matrix
        cm_trade_relative = cm_trade.astype('float') / cm_trade.sum(axis=1)[:, np.newaxis]
        logging.debug("Relative Trade Outcome Confusion Matrix:")
        logging.debug(cm_trade_relative)
        disp_trade_relative = ConfusionMatrixDisplay(confusion_matrix=cm_trade_relative, display_labels=["Buy", "Sell"])
        disp_trade_relative.plot()
        plt.title("Relative Trade Outcome Confusion Matrix")
        plt.show()