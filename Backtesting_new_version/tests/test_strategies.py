import unittest
import pandas as pd
from strategies.binary_strategy import BinaryStrategy
from signals.binary_signal import BinarySignal

class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.config = {
            "strategies": {
                "trading_strategy": {
                    "params": {
                        "initial_cash": 900000,
                        "position_size_factor": 0.1
                    }
                }
            }
        }
        self.mock_df = pd.DataFrame({
            'timestamp': pd.date_range(start='2022-01-01', periods=10, freq='D'),
            'current_price': [100, 105, 90, 108, 110, 89, 105, 103, 101, 99],
            'predicted_high': [150, 100, 40, 60, 115, 110, 190, 106, 200, 102]
        })

    def test_binary_strategy(self):
        buy_signal_config = {"function": "lambda current_price, predicted_high: current_price < predicted_high"}
        sell_signal_config = {"function": "lambda current_price, buy_price: current_price > buy_price"}
        exit_signal_config = {"function": "lambda current_price, buy_price, predicted_high: current_price >= predicted_high or current_price <= buy_price"}

        strategy = BinaryStrategy(self.config)
        strategy.X_test_df = self.mock_df
        for index, row in self.mock_df.iterrows():
            current_price = row['current_price']
            signal = BinarySignal(current_price, buy_signal_config, sell_signal_config, exit_signal_config)
            strategy.execute_trade(signal, row['timestamp'], row['predicted_high'], current_price)
            print(f"Index: {index}, Current Price: {current_price}, Buy Price: {strategy.buy_price}")
        self.assertEqual(strategy.total_trades, 6)

if __name__ == '__main__':
    unittest.main()