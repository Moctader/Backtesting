# Backtesting Framework

This project is a backtesting framework for evaluating trading strategies. The framework supports two different types of trading strategies, including Binary, Binary Plus Exit, 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Strategies](#strategies)
- [Signals](#signals)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Moctader/Backtesting
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your historical data file (e.g., `EODHD_EURUSD_HISTORICAL_2019_2024_1min.csv`).

2. Run the backtest:
    ```sh
    python backtesting.py
    ```

## Configuration

The configuration for the signals and strategies is defined in a YAML file (`config2.yaml`). Here is an example configuration:

```yaml
signals:
  buy_signal:
    function: "lambda current_price, predicted_high: current_price < predicted_high"

  sell_signal:
    function: "lambda current_price, buy_price: current_price > buy_price"

  exit_signal:
    function: "lambda current_price, predicted_high, buy_price: current_price >= predicted_high or current_price <= buy_price"

strategies:
  trading_strategy:
    description: "Strategy combining buy, sell, and stop-loss signals."
    buy_signal: "buy_signal"
    sell_signal: "sell_signal"
    exit_signal: "exit_signal"
    params:
      initial_cash: 100000  
      position_size_factor: 0.1