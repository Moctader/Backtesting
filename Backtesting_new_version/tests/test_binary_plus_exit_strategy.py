import unittest
from unittest.mock import MagicMock
from strategies.binary_plus_exit_strategy import BinaryPlusExitStrategy
from signals.binary_signal import BinarySignal
from signals.binary_plus_exit_signal import BinaryPlusExitSignal

class TestBinaryPlusExitStrategy(unittest.TestCase):
    def setUp(self):
        # Initialize BinaryPlusExitStrategy with a mock config and mock BaseStrategy methods
        mock_config = MagicMock()
        self.strategy = BinaryPlusExitStrategy(mock_config)
        self.strategy.buy = MagicMock()
        self.strategy.sell = MagicMock()
        self.strategy.exit = MagicMock()
        self.strategy.update_portfolio_value = MagicMock()

        # Set initial position (0 indicates no current position)
        self.strategy.position = 0
        self.strategy.previous_signal = None
        self.strategy.signal_strength = 0

    def test_execute_trade_no_position_buy_signal(self):
        """Test Case 1: No position, buy signal is active"""
        future_timestamp = "2024-11-07T10:00:00"
        predicted_high = 101
        current_price = 100

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, predicted_high: current_price > predicted_high"}
        exit_signal_config = {"function": "lambda current_price, predicted_high: current_price == predicted_high"}

        # Create an instance of BinarySignal
        signal = BinaryPlusExitSignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)
        
        # Mock the generate_signals method to use the BinarySignal instance
        self.strategy.generate_signals = MagicMock(return_value=(
            signal.generate_buy_signal(predicted_high),
            signal.generate_sell_signal(predicted_high),
            signal.generate_exit_signal(predicted_high)
        
        ))

        # Run the method
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assert that buy was called once
        self.strategy.buy.assert_called_once_with(future_timestamp, predicted_high)
        # Assert that update_portfolio_value was called once
        self.strategy.update_portfolio_value.assert_called_once()
        # Ensure other actions were not taken
        self.strategy.sell.assert_not_called()
        self.strategy.exit.assert_not_called()

    def test_execute_trade_holding_position_buy_signal(self):
        """Test Case 2: Holding a position, buy signal is active"""
        self.strategy.position = 1  # Assume holding a position
        future_timestamp = "2024-11-07T10:00:00"
        predicted_high = 120
        current_price = 100

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, predicted_high: current_price > predicted_high"}
        exit_signal_config = {"function": "lambda current_price, predicted_high: current_price == predicted_high"}

        # Create an instance of BinarySignal
        signal = BinaryPlusExitSignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)

        self.strategy.generate_signals = MagicMock(return_value=(
            signal.generate_buy_signal(predicted_high),
            signal.generate_sell_signal(predicted_high),
            signal.generate_exit_signal(predicted_high)
        
        ))

        # Run the method
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assert that buy was called once
        self.strategy.buy.assert_called_once_with(future_timestamp, predicted_high)
        # Assert that update_portfolio_value was called once
        self.strategy.update_portfolio_value.assert_called_once()
        # Ensure other actions were not taken
        self.strategy.sell.assert_not_called()
        self.strategy.exit.assert_not_called()

    def test_execute_trade_holding_position_sell_signal(self):
        """Test Case 3: Holding a position, sell signal is active"""
        self.strategy.position = 1  # Assume holding a position
        future_timestamp = "2024-11-07T10:00:00"
        predicted_high = 90
        current_price = 100

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, predicted_high: current_price > predicted_high"}
        exit_signal_config = {"function": "lambda current_price, predicted_high: current_price == predicted_high"}

        # Create an instance of BinarySignal
        signal = BinaryPlusExitSignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)

        # Mock the generate_signals method to use the BinarySignal instance
        self.strategy.generate_signals = MagicMock(return_value=(
                signal.generate_buy_signal(predicted_high),
                signal.generate_sell_signal(predicted_high),
                signal.generate_exit_signal(predicted_high)
            
            ))

        # Run the method
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assert that sell was called once
        self.strategy.sell.assert_called_once_with(future_timestamp, predicted_high)
        # Assert that update_portfolio_value was called once
        self.strategy.update_portfolio_value.assert_called_once()
        # Ensure other actions were not taken
        self.strategy.buy.assert_not_called()
        self.strategy.exit.assert_not_called()

    def test_execute_trade_holding_position_exit_signal(self):
        """Test Case 4: Holding a position, exit signal is active"""
        self.strategy.position = 1  # Assume holding a position
        future_timestamp = "2024-11-07T10:00:00"
        predicted_high = 100
        current_price = 100

        # Define the configurations for the BinarySignal
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, predicted_high: current_price > predicted_high"}
        exit_signal_config = {"function": "lambda current_price, predicted_high: current_price == predicted_high"}

        # Create an instance of BinarySignal
        signal = BinaryPlusExitSignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)

        # Mock the generate_signals method to use the BinarySignal instance
        self.strategy.generate_signals = MagicMock(return_value=(
            False,  # buy_signal=False
            False,  # sell_signal=False
            signal.generate_exit_signal(predicted_high)  # exit_signal=True
        ))

        # Run the method
        self.strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

        # Assert that exit was called once
        self.strategy.exit.assert_called_once_with(future_timestamp, predicted_high)
        # Assert that update_portfolio_value was called once
        self.strategy.update_portfolio_value.assert_called_once()
        # Ensure other actions were not taken
        self.strategy.buy.assert_not_called()
        self.strategy.sell.assert_not_called()


if __name__ == '__main__':
    unittest.main()