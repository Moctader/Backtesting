import yaml
import torch
import logging
from backtesting import BackTesting

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Load configuration from YAML
def load_config(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == '__main__':
    bt = BackTesting('pytorch')
    bt.run_backtest('../../../EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv')