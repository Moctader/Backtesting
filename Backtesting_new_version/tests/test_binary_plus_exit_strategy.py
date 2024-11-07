import unittest
from unittest.mock import MagicMock
import pandas as pd
from strategies.binary_plus_exit_strategy import BinaryPlusExitStrategy
from signals.binary_plus_exit_signal import BinaryPlusExitSignal

class TestBinaryPlusExitStrategy(unittest.TestCase):
    def setUp(self):
        # Mock config for initializing the strategy
        self.config = {
            "strategies": {
                "trading_strategy": {
                    "params": {
                        "initial_cash": 100000,
                        "position_size_factor": 0.1,
                    }
                }
            }
        }
        # Instantiate the BinaryPlusExitStrategy
        self.strategy = BinaryPlusExitStrategy(self.config)

    def test_initialization(self):
        """Test if the strategy is initialized with correct attributes."""
        self.assertEqual(self.strategy.initial_cash, 100000)
        self.assertEqual(self.strategy.position_size_factor, 0.1)
        self.assertEqual(self.strategy.cash, 100000)
        self.assertEqual(self.strategy.position, 0)
        self.assertEqual(self.strategy.total_trades, 0)
        self.assertEqual(len(self.strategy.portfolio_value), 0)

    def test_execute_trade_buy(self):
        """Test the execute_trade method for a buy signal."""
        future_timestamp = "2024-01-01"
        predicted_high = 60
        current_price = 50

        # Create a BinaryPlusExitSignal instance with appropriate configurations
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}
        exit_signal_config = {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high and current_price <= buy_price"}

        signal = BinaryPlusExitSignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)

        # Mock the make_decision method to avoid side effects and check calls
        self.strategy.make_decision = MagicMock()

        # Call the execute_trade method with a buy signal
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Expected calculations
        expected_shares_to_buy = (self.strategy.initial_cash * self.strategy.position_size_factor) // current_price

        # Assertions
        self.assertEqual(self.strategy.position, expected_shares_to_buy)
        self.assertLess(self.strategy.cash, self.strategy.initial_cash)  # Cash should decrease
        self.assertEqual(self.strategy.total_trades, 1)
        #self.assertEqual(self.strategy.decisions[-1]["action"], "buy")

    def test_execute_trade_sell(self):
        """Test the execute_trade method for a sell signal."""
        future_timestamp = "2024-01-02"
        predicted_high = 40
        current_price = 55

        # Create a BinaryPlusExitSignal instance with appropriate configurations
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}
        exit_signal_config = {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high and current_price <= buy_price"}

        signal = BinaryPlusExitSignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)

        # Mock the make_decision method to avoid side effects and check calls
        self.strategy.make_decision = MagicMock()

        # Set initial position and buy price
        self.strategy.position = 100
        self.strategy.buy_price = 50

        # Call the execute_trade method with a sell signal
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assertions
        self.assertEqual(self.strategy.position, 0)  # Position should be reset to 0
        self.assertGreater(self.strategy.cash, self.strategy.initial_cash)  # Cash should increase if sold at profit
        self.assertEqual(self.strategy.total_trades, 1)
        #self.assertEqual(self.strategy.decisions[-1]["action"], "sell")

    def test_execute_trade_exit(self):
        """Test the execute_trade method for an exit signal."""
        future_timestamp = "2024-01-03"
        predicted_high = 45
        current_price = 55

        # Create a BinaryPlusExitSignal instance with appropriate configurations
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}
        exit_signal_config = {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high and current_price <= buy_price"}

        signal = BinaryPlusExitSignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)

        # Mock the make_decision method to avoid side effects and check calls
        self.strategy.make_decision = MagicMock()

        # Set initial position and buy price
        self.strategy.position = 100
        self.strategy.buy_price = 50

        # Call the execute_trade method with an exit signal
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assertions
        self.assertEqual(self.strategy.position, 0)  # Position should be reset to 0
        self.assertGreater(self.strategy.cash, self.strategy.initial_cash)  # Cash should increase if sold at profit
        self.assertEqual(self.strategy.total_trades, 1)
        #self.assertEqual(self.strategy.decisions[-1]["action"], "exit")

if __name__ == '__main__':
    unittest.main()