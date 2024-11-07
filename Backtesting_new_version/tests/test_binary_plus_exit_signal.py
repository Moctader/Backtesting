import unittest
from signals.binary_plus_exit_signal import BinaryPlusExitSignal

class TestBinaryPlusExitSignal(unittest.TestCase):
    def setUp(self):
        # Mock configurations for buy, sell, and exit signals
        self.buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        self.sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}
        self.exit_signal_config = {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high and current_price <= buy_price"}

        # Instantiate the BinaryPlusExitSignal with mock current price and signal configurations
        self.signal = BinaryPlusExitSignal(
            current_price=50,
            buy_signal=self.buy_signal_config,
            sell_signal=self.sell_signal_config,
            exit_signal=self.exit_signal_config
        )

    def test_generate_buy_signal(self):
        """Test generate_buy_signal method."""
        predicted_high = 60
        buy_signal = self.signal.generate_buy_signal(predicted_high)
        self.assertTrue(buy_signal, "Expected buy signal to be True")

    def test_generate_sell_signal(self):
        """Test generate_sell_signal method."""
        buy_price = 45
        sell_signal = self.signal.generate_sell_signal(buy_price)
        self.assertTrue(sell_signal, "Expected sell signal to be True")

        buy_price = 55
        sell_signal = self.signal.generate_sell_signal(buy_price)
        self.assertFalse(sell_signal, "Expected sell signal to be False")

    def test_generate_exit_signal(self):
        """Test generate_exit_signal method."""
        buy_price = 55
        predicted_high = 46
        exit_signal = self.signal.generate_exit_signal(buy_price, predicted_high)
        self.assertTrue(exit_signal, "Expected exit signal to be True")

        predicted_high = 60
        exit_signal = self.signal.generate_exit_signal(buy_price, predicted_high)
        self.assertFalse(exit_signal, "Expected exit signal to be False")

if __name__ == '__main__':
    unittest.main()
