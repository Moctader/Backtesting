# Model
################################################################
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import ffn
import os



################################################################
# Trading Strategy
import yaml

# Load YAML configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract trading strategy parameters
initial_cash = config['trading_strategy']['initial_cash']
position_size_factor = config['trading_strategy']['position_size_factor']
buy_threshold = config['trading_strategy']['buy_threshold']
buy_expression = config['trading_strategy']['buy_strategy']['condition']['expression']
sell_expression = config['trading_strategy']['sell_strategy']['condition']['expression']
exit_expression = config['trading_strategy']['exit_strategy']['condition']['expression']
# Initialize parameters
position = 0  # Initially no position
cash = initial_cash
portfolio_value = []
trades = []
trading_days = len(X_test)

# Record start and end prices for buy-and-hold strategy
start_price = data['close'].iloc[train_size]
end_price = data['close'].iloc[-1]


# Backtest
for i in range(trading_days):
    current_price = data['close'].iloc[train_size + i]
    current_price = current_price if isinstance(current_price, (float, int)) else current_price.item()
    predicted_high = predictions[i][0]
    daily_portfolio_value = cash  # Start with cash

    # Buy condition: predicted high is 0.5% higher than current price and no current position
    if eval(buy_expression):
        amount_to_invest = position_size_factor * cash  # Invest 10% of cash
        shares_to_buy = amount_to_invest // current_price
        cash -= shares_to_buy * current_price
        position += shares_to_buy
        buy_price = current_price  # Store the buy price
        print(f"Bought {shares_to_buy} shares at {current_price:.2f}")

    # Sell condition: current price reaches or exceeds predicted high
    if eval(sell_expression):
        cash += position * current_price
        profit = (current_price - buy_price) * position  # Calculate profit based on buy price
        trades.append(profit)  # Store the result of the trade
        print(f"Sold {position} shares at {current_price:.2f}, Profit: {profit:.2f}")
        position = 0  # Reset position after selling

    # Update portfolio value
    daily_portfolio_value = cash + position * current_price
    portfolio_value.append(daily_portfolio_value)


################################################################

# Performance metrics
# Convert portfolio value to pandas Series for performance analysis
portfolio_series = pd.Series(portfolio_value, index=pd.date_range(start=data.index[-len(portfolio_value)], periods=len(portfolio_value)))

# Calculate hit ratio
winning_trades = [trade for trade in trades if trade > 0]
hit_ratio = (len(winning_trades) / len(trades)) * 100 if trades else 0

print()
# Calculate buy-and-hold gain
buy_and_hold_gain = (end_price - start_price) / start_price * 100

# Calculate strategy gain
strategy_gain = (portfolio_value[-1] - initial_cash) / initial_cash * 100
print(f"Final portfolio value: ${portfolio_value[-1]:.2f}")
print(f"Hit Ratio: {hit_ratio:.2f}%")
print(f"Buy-and-Hold Gain: {buy_and_hold_gain:.2f}%")
print(f"Strategy Gain: {strategy_gain:.2f}%")

# Plot portfolio value over time
plt.plot(portfolio_series.index, portfolio_value, label='Portfolio Value')
plt.xlabel('Minutes')
plt.ylabel('Portfolio Value (USD)')
plt.title('Portfolio Value Over Time')
plt.legend()
plt.show()

# Display performance stats
perf = portfolio_series.calc_stats()
perf.display()





################################################################
# Calculate and Plot Cumulative Difference

buy_and_hold_returns = data['close'].pct_change().iloc[train_size:train_size + trading_days]
buy_and_hold_cumulative_returns = (1 + buy_and_hold_returns).cumprod()
trading_strategy_returns = pd.Series(portfolio_value).pct_change()
trading_strategy_cumulative_returns = (1 + trading_strategy_returns).cumprod()
buy_and_hold_cumulative_returns = buy_and_hold_cumulative_returns[:len(trading_strategy_cumulative_returns)]
cumulative_difference = trading_strategy_cumulative_returns.values - buy_and_hold_cumulative_returns.values

# Plot cumulative difference
plt.plot(trading_strategy_cumulative_returns.index, cumulative_difference)
plt.xlabel('Days')
plt.ylabel('Cumulative Difference')
plt.title('Cumulative Difference Between Trading Strategy and Buy-and-Hold')
plt.legend()
plt.show()