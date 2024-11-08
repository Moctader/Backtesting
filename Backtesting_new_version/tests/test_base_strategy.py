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


    def test_exit(self):
        """Test the exit method to verify updates to cash, position, and trade outcome logging."""
        self.strategy.current_price = 55  # Mock current price
        self.strategy.position = 100
        self.strategy.buy_price = 50  # Mock initial buy price
        self.strategy.cash = 10000  # Mock initial cash
        self.strategy.total_trades = 0  # Mock initial total trades

        # Mock the make_decision method to avoid side effects and check calls
        self.strategy.make_decision = MagicMock()

        # Expected calculations
        expected_trade_value = self.strategy.position * self.strategy.current_price
        expected_cash_after_exit = self.strategy.cash + expected_trade_value

        # Call exit method
        self.strategy.exit("2024-01-02", 60)

        # Assertions
        self.assertEqual(self.strategy.position, 0, "Position should be reset to 0 after exit.")
        self.assertEqual(self.strategy.cash, expected_cash_after_exit, "Cash is incorrect after exit.")
        self.assertEqual(self.strategy.buy_price, None, "Buy price should be reset to None after exit.")
        self.assertEqual(self.strategy.total_trades, 1, "Total trades should increment by 1 after exit.")

        # Check that make_decision was called with correct parameters
        self.strategy.make_decision.assert_called_once_with(
            "2024-01-02", 60, "exit", self.strategy.current_price
        )

    def test_buy_to_cover(self):
        """Test the buy_to_cover method to verify updates to cash, position, and trade outcome logging."""
        self.strategy.current_price = 55  # Mock current price
        self.strategy.position = -100  # Mock short position
        self.strategy.cash = 10000  # Mock initial cash
        self.strategy.total_trades = 0  # Mock initial total trades

        # Mock the make_decision and track_trade_outcome methods to avoid side effects and check calls
        self.strategy.make_decision = MagicMock()
        self.strategy.track_trade_outcome = MagicMock()

        # Expected calculations
        expected_trade_value = abs(self.strategy.position) * self.strategy.current_price
        expected_cash_after_buy_to_cover = self.strategy.cash - expected_trade_value
        expected_profit_or_loss = -expected_trade_value

        # Call buy_to_cover method
        self.strategy.buy_to_cover("2024-01-02", 60)

        # Assertions
        self.assertEqual(self.strategy.position, 0, "Position should be reset to 0 after buy to cover.")
        self.assertEqual(self.strategy.cash, expected_cash_after_buy_to_cover, "Cash is incorrect after buy to cover.")
        self.assertEqual(self.strategy.total_trades, 1, "Total trades should increment by 1 after buy to cover.")

        # Check that track_trade_outcome was called with correct parameters
        self.strategy.track_trade_outcome.assert_called_once_with(expected_profit_or_loss)

        # Check that make_decision was called with correct parameters
        self.strategy.make_decision.assert_called_once_with(
            "2024-01-02", 60, "buy_to_cover", self.strategy.current_price
        )


    def test_update_portfolio_value(self):
        """Test that portfolio value updates correctly."""
        self.strategy.current_price = 100
        self.strategy.position = 10
        self.strategy.cash = 90000

        self.strategy.update_portfolio_value()
        self.assertEqual(self.strategy.portfolio_value[-1], 90000 + 10 * 100)


    def test_initiate_short_position(self):
        """Test the initiate_short_position method to verify updates to cash, position, and trade count."""
        self.strategy.current_price = 55  # Mock current price
        self.strategy.cash = 10000  # Mock initial cash
        self.strategy.position_size_factor = 0.1  # Mock position size factor
        self.strategy.total_trades = 0  # Mock initial total trades

        # Expected calculations
        expected_amount_to_invest = self.strategy.position_size_factor * self.strategy.cash
        expected_shares_to_short = expected_amount_to_invest // self.strategy.current_price
        expected_cash_after_short = self.strategy.cash + (expected_shares_to_short * self.strategy.current_price)
        expected_position_after_short = -expected_shares_to_short

        # Call initiate_short_position method
        self.strategy.initiate_short_position()

        # Assertions
        self.assertEqual(self.strategy.cash, expected_cash_after_short, "Cash is incorrect after initiating short position.")
        self.assertEqual(self.strategy.position, expected_position_after_short, "Position is incorrect after initiating short position.")
        self.assertEqual(self.strategy.total_trades, 1, "Total trades should increment by 1 after initiating short position.")




    def test_track_trade_outcome(self):
        """Test the track_trade_outcome method to verify updates to winning and losing trades."""
        self.strategy.winning_trades = 0  # Mock initial winning trades
        self.strategy.losing_trades = 0  # Mock initial losing trades

        # Call track_trade_outcome with a profit
        self.strategy.track_trade_outcome(100)
        self.assertEqual(self.strategy.winning_trades, 1, "Winning trades should increment by 1 for a profit.")
        self.assertEqual(self.strategy.losing_trades, 0, "Losing trades should remain the same for a profit.")

        # Call track_trade_outcome with a loss
        self.strategy.track_trade_outcome(-50)
        self.assertEqual(self.strategy.winning_trades, 1, "Winning trades should remain the same for a loss.")
        self.assertEqual(self.strategy.losing_trades, 1, "Losing trades should increment by 1 for a loss.")




    def test_generate_signals_with_binary_signal(self):
        """Test generate_signals method with BinarySignal instance"""
        predicted_high = 150
        current_price = 100

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, predicted_high: current_price > predicted_high"}

        # Create an instance of BinarySignal
        signal = BinarySignal(current_price, buy_signal_config, sell_signal_config)

        # Mock the generate_buy_signal and generate_sell_signal methods
        signal.generate_buy_signal = MagicMock(return_value=True)
        signal.generate_sell_signal = MagicMock(return_value=False)

        # Run the method
        buy_signal, sell_signal = self.strategy.generate_signals(signal, predicted_high)

        # Assert the signals
        self.assertTrue(buy_signal)
        self.assertFalse(sell_signal)

    def test_generate_signals_with_non_binary_signal(self):
        """Test generate_signals method with non-BinarySignal instance"""
        predicted_high = 150
        current_price = 100

        # Define a mock signal class
        class MockSignal:
            def generate_buy_signal(self, predicted_high):
                return True

            def generate_sell_signal(self, buy_price):
                return False

            def generate_exit_signal(self, predicted_high):
                return True

        # Create an instance of MockSignal
        signal = MockSignal()

        # Set buy_price to a valid value
        self.strategy.buy_price = 90

        # Run the method
        buy_signal, sell_signal, exit_signal = self.strategy.generate_signals(signal, predicted_high)

        # Assert the signals
        self.assertTrue(buy_signal)
        self.assertFalse(sell_signal)
        self.assertTrue(exit_signal)

    def test_fetch_actual_price(self):
        """Test the fetch_actual_price method to verify correct price fetching."""
        # Mock the X_test_df DataFrame
        self.strategy.X_test_df = pd.DataFrame({
            'high': [100, 105, 110],
        }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))

        # Test with a valid timestamp
        timestamp = pd.to_datetime('2024-01-02')
        actual_price = self.strategy.fetch_actual_price(timestamp)
        self.assertEqual(actual_price, 105, "The actual price should be 105 for the given timestamp.")

        # Test with a timestamp not in the DataFrame
        timestamp = pd.to_datetime('2024-01-04')
        actual_price = self.strategy.fetch_actual_price(timestamp)
        self.assertIsNone(actual_price, "The actual price should be None for a timestamp not in the DataFrame.")

        # Test with a NaN value in the 'high' column
        self.strategy.X_test_df.loc[pd.to_datetime('2024-01-02'), 'high'] = float('nan')
        actual_price = self.strategy.fetch_actual_price(pd.to_datetime('2024-01-02'))
        self.assertIsNone(actual_price, "The actual price should be None for a NaN value in the 'high' column.")

        # Test with the 'high' column missing
        self.strategy.X_test_df.drop(columns=['high'], inplace=True)
        actual_price = self.strategy.fetch_actual_price(pd.to_datetime('2024-01-01'))
        self.assertIsNone(actual_price, "The actual price should be None if the 'high' column is missing.")


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