import unittest
from unittest.mock import MagicMock
from strategies.binary_strategy import BinaryStrategy
from signals.binary_signal import BinarySignal

class TestBinaryStrategy(unittest.TestCase):
    def setUp(self):
        # Initialize BinaryStrategy with a mock config and mock BaseStrategy methods
        mock_config = MagicMock()
        self.strategy = BinaryStrategy(mock_config)
        self.strategy.buy = MagicMock()
        self.strategy.sell = MagicMock()
        self.strategy.buy_to_cover = MagicMock()
        self.strategy.update_portfolio_value = MagicMock()
        self.strategy.initiate_short_position = MagicMock()

        # Set initial position (0 indicates no current position)
        self.strategy.position = 0

    def test_execute_trade_no_position_buy_signal(self):
        """Test Case 1: No position, buy signal is active"""
        future_timestamp = "2024-11-07T10:00:00"
        predicted_high = 101
        current_price = 100

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}

        # Create an instance of BinarySignal
        signal = BinarySignal(current_price, buy_signal_config, sell_signal_config)
        
        # Mock the generate_signals method to use the BinarySignal instance
        self.strategy.generate_signals = MagicMock(return_value=(
            signal.generate_buy_signal(predicted_high),
            signal.generate_sell_signal(predicted_high),
        ))

        # Run the method
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assert that buy was called once
        self.strategy.buy.assert_called_once_with(future_timestamp, predicted_high)
        # Assert that update_portfolio_value was called once
        self.strategy.update_portfolio_value.assert_called_once()
        # Ensure other actions were not taken
        self.strategy.sell.assert_not_called()
        self.strategy.buy_to_cover.assert_not_called()
        self.strategy.initiate_short_position.assert_not_called()

    def test_execute_trade_buy_position_sell_signal(self):
        """Test Case 2: In a buy position, sell signal is active"""
        self.strategy.position = 1  # Assume in a buy position
        future_timestamp = "2024-11-07T10:00:00"
        predicted_high = 90
        current_price = 100
        self.strategy.buy_price=90

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}

        # Create an instance of BinarySignal
        signal = BinarySignal(current_price, buy_signal_config, sell_signal_config)

        # Mock the generate_signals method to use the BinarySignal instance
        self.strategy.generate_signals = MagicMock(return_value=(
            signal.generate_buy_signal(predicted_high),
            signal.generate_sell_signal(predicted_high),
        ))

        # Run the method
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assert that sell was called once and initiate_short_position was called once
        self.strategy.sell.assert_called_once_with(future_timestamp, predicted_high)
        self.strategy.initiate_short_position.assert_called_once()
        # Assert that update_portfolio_value was called once
        self.strategy.update_portfolio_value.assert_called_once()
        # Ensure other actions were not taken
        self.strategy.buy.assert_not_called()
        self.strategy.buy_to_cover.assert_not_called()

    def test_execute_trade_short_position_buy_signal(self):
        """Test Case 3: In a short position, buy signal is active"""
        self.strategy.position = -1  # Assume in a short position
        future_timestamp = "2024-11-07T10:00:00"
        predicted_high = 90
        current_price = 100

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}

        # Create an instance of BinarySignal
        signal = BinarySignal(current_price, buy_signal_config, sell_signal_config)

        # Mock the generate_signals method to use the BinarySignal instance
        self.strategy.generate_signals = MagicMock(return_value=(
            signal.generate_buy_signal(predicted_high),
            signal.generate_sell_signal(predicted_high),
        ))

        # Run the method
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assert that buy_to_cover and buy were each called once
        #self.strategy.buy_to_cover.assert_called_once_with(future_timestamp, predicted_high)
        #self.strategy.buy.assert_called_once_with(future_timestamp, predicted_high)
        # Assert that update_portfolio_value was called once
        self.strategy.update_portfolio_value.assert_called_once()
        # Ensure other actions were not taken
        self.strategy.sell.assert_not_called()
        self.strategy.initiate_short_position.assert_not_called()


if __name__ == '__main__':
    unittest.main()