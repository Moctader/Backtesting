import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
from strategies.base_strategy import BaseStrategy
from strategies.binary_strategy import BinaryStrategy
from signals.binary_signal import BinarySignal



class TestBaseStrategy(unittest.TestCase):
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
        # Instantiate the MockStrategy (a concrete subclass of BaseStrategy)
        self.strategy = BinaryStrategy(self.config)

    def test_initialization(self):
        """Test if the strategy is initialized with correct attributes."""
        self.assertEqual(self.strategy.initial_cash, 100000)
        self.assertEqual(self.strategy.position_size_factor, 0.1)
        self.assertEqual(self.strategy.cash, 100000)
        self.assertEqual(self.strategy.position, 0)
        self.assertEqual(self.strategy.total_trades, 0)
        self.assertEqual(len(self.strategy.portfolio_value), 0)


    def test_buy(self):
        """Test the buy method to ensure it performs correctly."""
        future_timestamp = "2024-01-01"
        predicted_high = 55
        self.strategy.signal_strength = 1  
        self.strategy.current_price=50

        # Mock the make_decision method to avoid side effects and check calls
        self.strategy.make_decision = MagicMock()
        
        # Expected calculations
        expected_amount_to_invest = self.strategy.position_size_factor * self.strategy.cash * self.strategy.signal_strength
        expected_shares_to_buy = expected_amount_to_invest // self.strategy.current_price
        expected_cash_after_buy = self.strategy.cash - (expected_shares_to_buy * self.strategy.current_price)

        # Call the buy method
        self.strategy.buy(future_timestamp, predicted_high)

        # Assertions
        self.assertEqual(self.strategy.position, expected_shares_to_buy, "Position (number of shares) is incorrect after buy.")
        self.assertEqual(self.strategy.cash, expected_cash_after_buy, "Cash is incorrect after buy.")
        self.assertEqual(self.strategy.buy_price, self.strategy.current_price, "Buy price should be set to current price.")
        self.assertEqual(self.strategy.total_trades, 1, "Total trades should increment by 1 after buy.")
        
        # Check that make_decision was called with correct parameters
        self.strategy.make_decision.assert_called_once_with(
            future_timestamp, predicted_high, "buy", self.strategy.current_price
        )


    def test_sell(self):
        """Test the sell method to verify updates to cash, position, and trade outcome logging."""
        self.strategy.current_price = 55  # Mock current price
        self.strategy.position = 100
        self.strategy.buy_price = 50  # Mock initial buy price

        # Call sell method
        self.strategy.sell("2024-01-02", 60)

        self.assertEqual(self.strategy.position, 0)  # Position should be reset to 0
        self.assertGreater(self.strategy.cash, self.strategy.initial_cash)  # Cash should increase if sold at profit
        self.assertEqual(self.strategy.total_trades, 1)
        self.assertEqual(self.strategy.decisions[-1]["action"], "sell")



    def test_generate_signals_with_binary_signal(self):
        """Test generate_signals with BinarySignal which should not generate exit signal."""
        # Create a BinarySignal instance with appropriate configurations
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}
        exit_signal_config = {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high or current_price <= buy_price"}

        binary_signal = BinarySignal(50, buy_signal_config, sell_signal_config, exit_signal_config)

        # Set the buy_price for the strategy to test the sell signal
        self.strategy.buy_price = 50

        # Generate signals
        buy_signal, sell_signal = self.strategy.generate_signals(binary_signal, 60)

        # Check that the buy and sell signals are correct, and no exit signal is returned
        self.assertTrue(buy_signal, "Expected buy signal to be True")
        self.assertFalse(sell_signal, "Expected sell signal to be False")


    def test_update_portfolio_value(self):
        """Test that portfolio value updates correctly."""
        self.strategy.current_price = 100
        self.strategy.position = 10
        self.strategy.cash = 90000

        self.strategy.update_portfolio_value()
        self.assertEqual(self.strategy.portfolio_value[-1], 90000 + 10 * 100)


    @patch.object(BaseStrategy, 'fetch_actual_price')
    def test_evaluate_decisions(self, mock_fetch_actual_price):
        """Test evaluate_decisions for correct comparison and matrix generation."""
        # Set up some mock data for decisions
        self.strategy.decisions = [
            {"timestamp": "2024-01-01", "prediction": 60, "action": "buy", "current_price": 50, "actual_price": 60},
            {"timestamp": "2024-01-02", "prediction": 45, "action": "sell", "current_price": 50, "actual_price": 45}
        ]
        mock_fetch_actual_price.side_effect = [60, 45]

        # Execute evaluate_decisions
        self.strategy.evaluate_decisions()

        # Assert outcomes are as expected
        self.assertEqual(self.strategy.decisions[0]["actual_price"], 60)
        self.assertEqual(self.strategy.decisions[1]["actual_price"], 45)

    @patch('matplotlib.pyplot.show')
    def test_plot_confusion_matrices(self, mock_show):
        """Test that _plot_confusion_matrices generates and displays without error."""
        y_true_signal = [1, 0]
        y_pred_signal = [1, 1]
        y_true_trade = [1, 0]
        y_pred_trade = [1, 0]

        # No assertions needed; test passes if no errors
        self.strategy._plot_confusion_matrices(y_true_signal, y_pred_signal, y_true_trade, y_pred_trade)

# Run the tests
if __name__ == '__main__':
    unittest.main()