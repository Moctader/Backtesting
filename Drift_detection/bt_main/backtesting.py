import torch
import numpy as np
from models import Model
from strategies import BinaryPlusExitStrategy
from performance_metrics import PerformanceMetrics
from utils import load_config
from signals import Signal

class BackTesting:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("CUDA Available:", torch.cuda.is_available())
    


    def run_backtest(self, data_file_path):
        model = Model(input_size=9, hidden_size=50, output_size=1, num_layers=1, device=self.device)
        model.load_data(data_file_path)
        model.prepare_data(time_step=60)
        model.create_datasets()
        model.train(model.X_train, model.y_train, model.X_test, model.y_test, epochs=100, batch_size=16, learning_rate=0.001, patience=8)
        self.predictions, self.y_test = model.evaluate(model.X_test, model.y_test, model.target_scaler)

        config = load_config("./config2.yaml")

        buy_signal_config = config["signals"]["buy_signal"]
        sell_signal_config = config["signals"]["sell_signal"]
        exit_signal = config["signals"]["exit_signal"]

        strategy = BinaryPlusExitStrategy(config)
        strategy.X_test_df = model.X_test_df

        # Run the backtest
        self._execute_trades(strategy, model, buy_signal_config, sell_signal_config, exit_signal)
        strategy.evaluate_decisions()
        performance_metrics = PerformanceMetrics(model.data, strategy)
        performance_metrics.calculate_performance_metrics()

    def _execute_trades(self, BinaryPlusExitStrategy, model, buy_signal_config, sell_signal_config, exit_signal):
        """Execute trades based on the model predictions and strategy."""
        start_index = int(0.8 * len(model.data))
        for i in range(len(model.X_test_df) - model.time_step):
            test_index = start_index + i
            current_price = model.data['close'].iloc[test_index]
            predicted_high = self.predictions[i][0]
            future_timestamp = model.X_test_df.index[i + 1]

            if i == 0:
                print(f"First future_timestamp: {future_timestamp}")

            # Create signal instance
            signal = Signal(current_price, buy_signal_config, sell_signal_config, exit_signal)
            BinaryPlusExitStrategy.execute_trade(signal, future_timestamp, predicted_high, current_price)

if __name__ == '__main__':
    bt = BackTesting()
    bt.run_backtest('../../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')



            # Use the appropriate signal type
            #signal_type = "BinarySignal"  # or "BinaryPlusExitSignal" or "MulticlassSignal"
            #exit_threshold = 30  # Example threshold for BinaryPlusExitSignal