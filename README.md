# Trading Strategy Configuration

This project demonstrates a flexible trading strategy configuration using YAML and lambda expressions. The configuration allows for dynamic specification of buy, sell, and stop-loss signals, as well as strategy parameters.

## Overview

The trading strategy is defined in a YAML file (`trading_strategy.yaml`). This file includes:
- Signal definitions using lambda expressions.
- Strategy parameters.
- A combination of signals to form a complete trading strategy.

## YAML Configuration

### Signals

Signals are defined using lambda expressions. These expressions are used for evaluation at runtime to determine buy, sell, and stop-loss conditions.

#### Buy Signal

The buy signal is triggered when the current price is less than the predicted high price adjusted by a buy threshold.

```yaml
buy_signal:
  function: "lambda current_price, predicted_high, buy_threshold: current_price < predicted_high * (1 + buy_threshold)"
  params:
    buy_threshold: 0.005

```
#### Sell Signal

The sell signal is triggered when the current price is greater than or equal to the predicted high price.


```yaml
sell_signal:
  function: "lambda current_price, predicted_high: current_price >= predicted_high"
  params: {}

```

#### Stop-Loss Signal
The stop-loss signal is triggered when the current price falls below the buy price adjusted by a stop-loss threshold.

```yaml
stop_loss_signal:
  function: "lambda current_price, buy_price, stop_loss_threshold: current_price <= buy_price * (1 - stop_loss_threshold)"
  params:
    stop_loss_threshold: 0.02
```


### Strategies
The strategy combines the buy, sell, and stop-loss signals and includes additional parameters such as initial cash, position size factor, and maximum holding time.

```yaml
strategies:
  trading_strategy:
    description: "Strategy combining buy, sell, and stop-loss signals."
    buy_signal: "buy_signal"
    sell_signal: "sell_signal"
    stop_loss_signal: "stop_loss_signal"
    params:
      initial_cash: 100000
      position_size_factor: 0.1
      max_holding_time: 5
```
