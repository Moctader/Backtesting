import unittest
from signals.binary_signal import BinarySignal
from signals.binary_plus_exit_signal import BinaryPlusExitSignal
from signals.base_signal import BaseSignal

class TestSignals(unittest.TestCase):
    def setUp(self):
        self.current_price = 100
        self.buy_signal = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        self.sell_signal = {"function": "lambda current_price, predicted_high: current_price > predicted_high"}
        self.exit_signal = {"function": "lambda current_price, predicted_high: current_price == predicted_high"}
        
    def test_binary_signal(self):
        signal = BinarySignal(self.current_price, self.buy_signal, self.sell_signal)
        self.assertTrue(signal.generate_buy_signal(110))
        self.assertFalse(signal.generate_buy_signal(90))
        self.assertTrue(signal.generate_sell_signal(90))
        self.assertFalse(signal.generate_sell_signal(110))


if __name__ == '__main__':
    unittest.main()