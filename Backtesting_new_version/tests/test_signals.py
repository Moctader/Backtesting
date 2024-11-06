import unittest
from signals.binary_signal import BinarySignal
from signals.binary_plus_exit_signal import BinaryPlusExitSignal
from signals.base_signal import BaseSignal

class TestSignals(unittest.TestCase):
    def setUp(self):
        self.current_price = 100
        self.buy_signal = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        self.sell_signal = {"function": "lambda current_price, buy_price: current_price > buy_price"}
        self.exit_signal = {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high or current_price <= buy_price"}

    def test_binary_signal(self):
        signal = BinarySignal(self.current_price, self.buy_signal, self.sell_signal)
        self.assertTrue(signal.generate_buy_signal(110))
        self.assertFalse(signal.generate_buy_signal(90))
        self.assertTrue(signal.generate_sell_signal(90))
        self.assertFalse(signal.generate_sell_signal(110))

    def test_base_signal(self):
        class TestSignal(BaseSignal):
            def generate_buy_signal(self, predicted_high):
                return self.buy_signal(self.current_price, predicted_high)

            def generate_sell_signal(self, buy_price):
                return self.sell_signal(self.current_price, buy_price)

            def generate_exit_signal(self, buy_price, predicted_high):
                return super().generate_exit_signal(buy_price, predicted_high)

        signal = TestSignal(self.current_price, self.buy_signal, self.sell_signal, self.exit_signal)
        self.assertTrue(signal.generate_buy_signal(110))
        self.assertFalse(signal.generate_buy_signal(90))
        self.assertTrue(signal.generate_sell_signal(90))
        self.assertFalse(signal.generate_sell_signal(110))
        # self.assertTrue(signal.generate_exit_signal(90, 110))
        # self.assertTrue(signal.generate_exit_signal(110, 90))
        # self.assertFalse(signal.generate_exit_signal(105, 110))


if __name__ == '__main__':
    unittest.main()