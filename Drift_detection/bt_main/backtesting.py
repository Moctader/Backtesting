import torch
from models import Model
from strategies import Strategy
from performance_metrics import PerformanceMetrics
from utils import load_config
import numpy as np
from signals import Signal


class BackTesting:
    def __init__(self, framework):
        self.framework = framework
        if framework == 'pytorch':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("CUDA Available:", torch.cuda.is_available())
            self.data_structure = self.initialize_pytorch_structure()
        else:
            self.data_structure = self.initialize_other_structure()

    def initialize_pytorch_structure(self):
        return {"data": torch.tensor([])}

    def initialize_other_structure(self):
        return {"data": np.array([])}

    def run_backtest(self, data_file_path):
        model = Model(input_size=9, hidden_size=50, output_size=1, num_layers=1, device=self.device)
        model.load_data(data_file_path)
        model.prepare_data(time_step=60)
        model.create_datasets()
        model.train(model.X_train, model.y_train, model.X_test, model.y_test, epochs=100, batch_size=16, learning_rate=0.001, patience=8)
        self.predictions, self.y_test = model.evaluate(model.X_test, model.y_test, model.target_scaler)

        # Load YAML configuration
        config = load_config("./config2.yaml")

        # Create signals
        buy_signal_config = config["signals"]["buy_signal"]
        sell_signal_config = config["signals"]["sell_signal"]
        exit_signal = config["signals"]["exit_signal"]

        # Create strategy
        strategy = Strategy(config)

        strategy.X_test_df = model.X_test_df
        start_index = int(0.8 * len(model.data))
        for i in range(len(self.predictions)):
            test_index = start_index + i
            current_price = model.data['close'].iloc[test_index]
            predicted_high = self.predictions[i]
            current_timestamp = model.data.index[test_index]
            future_timestamp = model.data.index[test_index + 1]

            if i == 0:
                print(f"First future_timestamp: {future_timestamp}")

            # Use the appropriate signal type
            #signal_type = "BinarySignal"  # or "BinaryPlusExitSignal" or "MulticlassSignal"
            #exit_threshold = 30  # Example threshold for BinaryPlusExitSignal

            signal = Signal(current_price, buy_signal_config, sell_signal_config, exit_signal)
            strategy.execute_trade(signal, future_timestamp, predicted_high, current_price)


            #strategy.execute_trade(signal_type, future_timestamp, predicted_high, current_price, exit_threshold)

        strategy.evaluate_decisions()

        performance_metrics = PerformanceMetrics(model.data, strategy)
        performance_metrics.calculate_performance_metrics()