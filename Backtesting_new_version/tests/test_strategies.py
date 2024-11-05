import unittest
from strategies.binary_strategy import BinaryStrategy
from strategies.binary_plus_exit_strategy import BinaryPlusExitStrategy
from signals.binary_signal import BinarySignal
from signals.binary_plus_exit_signal import BinaryPlusExitSignal
import pandas as pd

class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.config = {
            "strategies": {
                "trading_strategy": {
                    "params": {
                        "initial_cash": 100000,
                        "position_size_factor": 0.1
                    }
                }
            }
        }
        self.mock_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2022-01-01', periods=10, freq='D'),
            'current_price': [100, 105, 102, 108, 110, 107, 105, 103, 101, 99],
            'predicted_high': [110, 108, 107, 112, 115, 110, 108, 106, 104, 102],
            'buy_price': [None, 100, 105, 102, 108, 110, 107, 105, 103, 101]
        })

    def test_binary_strategy(self):
        strategy = BinaryStrategy(self.config)
        signal = BinarySignal(100, {"function": "lambda current_price, predicted_high: current_price < predicted_high"},
                              {"function": "lambda current_price, buy_price: current_price > buy_price"})
        strategy.X_test_df = self.mock_df
        for index, row in self.mock_df.iterrows():
            strategy.execute_trade(signal, row['timestamp'], row['predicted_high'], row['current_price'])
        self.assertEqual(strategy.total_trades, 10)

    def test_binary_plus_exit_strategy(self):
        strategy = BinaryPlusExitStrategy(self.config)
        signal = BinaryPlusExitSignal(100, {"function": "lambda current_price, predicted_high: current_price < predicted_high"},
                                      {"function": "lambda current_price, buy_price: current_price > buy_price"},
                                      {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high or current_price <= buy_price"})
        strategy.X_test_df = self.mock_df
        for index, row in self.mock_df.iterrows():
            strategy.execute_trade(signal, row['timestamp'], row['predicted_high'], row['current_price'])
        self.assertEqual(strategy.total_trades, 10)

if __name__ == '__main__':
    unittest.main()



# coverage run -m unittest discover -s tests
# coverage report -m