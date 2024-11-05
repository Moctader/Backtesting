import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import ffn
import logging
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from typing import Optional
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)



class PerformanceMetrics:
    def __init__(self, data, strategy):
        self.data = data
        self.strategy = strategy
        self.portfolio_series = None
        self.portfolio_df = None
        self.trading_per_min = None
        self.total_minutes = None
        self.trading_minutes_per_day = None  

    def prepare_data(self):
        try:
            self.data.index = pd.to_datetime(self.data.index)
            self.data['date'] = self.data.index
            logging.debug(f"Data index start: {self.data.index[0]}, end: {self.data.index[-1]}")

            self.portfolio_series = pd.Series(
                self.strategy.portfolio_value,
                index=pd.date_range(
                    start=self.data.index[-len(self.strategy.portfolio_value)],
                    periods=len(self.strategy.portfolio_value),
                    freq='min' 
                )
            )

            logging.debug(f"Length of self.portfolio_series: {len(self.portfolio_series)}")

            logging.debug(f"First few entries of self.portfolio_series:\n{self.portfolio_series.head()}")

        except Exception as e:
            logging.error(f"Error in prepare_data: {e}")

        print(self.portfolio_series)
        

    def calculate_trading_minutes_per_day(self):
        try:
            if self.portfolio_series is None or not isinstance(self.portfolio_series.index, pd.DatetimeIndex):
                raise ValueError("portfolio_series is not set or does not have a DateTimeIndex")
    
            total_num_trading = self.portfolio_series.groupby(self.portfolio_series.index.date).size()    
            num_trading_days = total_num_trading.size
            logging.info(f"Total Number of Trading Days: {num_trading_days}")
    
            average_trading_minutes_per_day = total_num_trading.mean()
            self.trading_minutes_per_day = average_trading_minutes_per_day
            logging.info(f"Average Trading Minutes Per Day: {self.trading_minutes_per_day:.2f}")

            for date, minutes in total_num_trading.items():
                logging.debug(f"Date: {date}, Minutes: {minutes}")
    
        except Exception as e:
            logging.error(f"Error in calculate_trading_minutes_per_day: {e}")



    def calculate_hit_ratio(self):
        """Calculate the hit ratio of the strategy."""
        try:
            if self.strategy.total_trades > 0:
                hit_ratio = (self.strategy.winning_trades / self.strategy.total_trades) * 100
                logging.debug(f"Hit Ratio: {hit_ratio}")
                return hit_ratio
            return 0
        except Exception as e:
            logging.error(f"Error in calculate_hit_ratio: {e}")
            return np.nan

    def calculate_strategy_gain(self):
        try:
            strategy_gain = (self.strategy.portfolio_value[-1] - self.strategy.initial_cash) / self.strategy.initial_cash * 100
            logging.debug(f"Strategy Gain: {strategy_gain}")
            return strategy_gain
        except Exception as e:
            logging.error(f"Error in calculate_strategy_gain: {e}")
            return np.nan

    def plot_portfolio_value(self):
        try:
            plt.plot(self.portfolio_series.index, self.portfolio_series, label='Portfolio Value')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value (USD)')
            plt.title('Portfolio Value Over Time')
            plt.legend()
            plt.show()
        except Exception as e:
            logging.error(f"Error in plot_portfolio_value: {e}")

    def convert_to_dataframe(self):
        try:
            self.portfolio_df = pd.DataFrame(self.portfolio_series)
            self.portfolio_df['returns'] = self.portfolio_df[0].pct_change()
        except Exception as e:
            logging.error(f"Error in convert_to_dataframe: {e}")

    def calculate_max_drawdown(self):
        try:
            max_drawdown = ffn.calc_max_drawdown(self.portfolio_series)
            logging.debug(f"Maximum Drawdown: {max_drawdown}")
            return max_drawdown
        except Exception as e:
            logging.error(f"Error in calculate_max_drawdown: {e}")
            return np.nan

    def calculate_sharpe_ratio(self, risk_free_rate=0.01):
        try:
            minute_returns = self.portfolio_df['returns'].dropna()
            expected_portfolio_return = minute_returns.mean()
            logging.debug(f"Expected Portfolio Return (Rx): {expected_portfolio_return}")

            portfolio_volatility = minute_returns.std()
            logging.debug(f"Standard Deviation of Portfolio Return (StdDev Rx): {portfolio_volatility}")

            sharpe_ratio = (expected_portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else np.nan
            logging.debug(f"Risk-Free Rate (Rf): {risk_free_rate}, Sharpe Ratio: {sharpe_ratio}")

            return sharpe_ratio
        except Exception as e:
            logging.error(f"Error in calculate_sharpe_ratio: {e}")
            return np.nan
            
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.0001):
        """Calculate the Sortino Ratio."""
        try:
            average_return = returns.mean()
            logging.debug(f"Average Return (Rp): {average_return}")

            target_return = risk_free_rate
            logging.debug(f"Target Rate of Return (T): {target_return}")

            downside_returns = returns[returns < target_return]
            downside_deviation = np.sqrt(np.mean(np.square(downside_returns - target_return)))
            logging.debug(f"Downside Deviation (Dd): {downside_deviation}")

            sortino_ratio = (average_return - target_return) / downside_deviation if downside_deviation != 0 else np.nan
            logging.debug(f"Sortino Ratio: {sortino_ratio}")

            return sortino_ratio
        except Exception as e:
            logging.error(f"Error in calculate_sortino_ratio: {e}")
            return np.nan
        
    def compare_with_buy_and_hold(self, initial_cash, train_size, trading_days):
        try:
            # Buy and Hold Strategy
            buy_and_hold_value = [initial_cash]
            for price in self.data['close'].iloc[train_size:train_size + trading_days]:
                buy_and_hold_value.append(buy_and_hold_value[-1] * (1 + price / self.data['close'].iloc[train_size] - 1))

            # Convert to Series
            trading_strategy_series = pd.Series(self.strategy.portfolio_value)
            buy_and_hold_series = pd.Series(buy_and_hold_value[1:])

            # Calculate the absolute difference between the two strategies
            absolute_difference = trading_strategy_series - buy_and_hold_series

            # Plotting the results
            plt.figure(figsize=(12, 6))
            plt.plot(trading_strategy_series, label='Trading Strategy', marker='o')
            plt.plot(buy_and_hold_series, label='Buy and Hold', marker='o')
            plt.plot(absolute_difference, label='Absolute Difference', marker='o', linestyle='--')
            plt.title('Trading Strategy vs. Buy and Hold Performance')
            plt.xlabel('Time Period')
            plt.ylabel('Portfolio Value')
            plt.legend()
            plt.grid()
            plt.show()

            # The final values for comparison
            print("Final Trading Strategy Value:", trading_strategy_series.iloc[-1])
            print("Final Buy and Hold Value:", buy_and_hold_series.iloc[-1])
            print("Absolute Difference:", absolute_difference.iloc[-1])
        except Exception as e:
            logging.error(f"Error in compare_with_buy_and_hold: {e}")
            

    def calculate_performance_metrics(self):
        """Calculate and print all performance metrics."""
        self.prepare_data()
        self.calculate_trading_minutes_per_day()
        self.convert_to_dataframe()

        hit_ratio = self.calculate_hit_ratio()
        strategy_gain = self.calculate_strategy_gain()
        max_drawdown = self.calculate_max_drawdown()
        sharpe_ratio = self.calculate_sharpe_ratio()
        sortino_ratio = self.calculate_sortino_ratio(self.portfolio_df['returns'].dropna())

        # logging.debug('portfolio_series')
        # logging.debug(self.portfolio_series)

        print(f"Final portfolio value: ${self.strategy.portfolio_value[-1]:.2f}")
        print(f"Hit Ratio: {hit_ratio:.2f}%")
        print(f"Strategy Gain: {strategy_gain:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.4%}" if max_drawdown is not None else "Maximum Drawdown: N/A")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")

        self.plot_portfolio_value()
        self.compare_with_buy_and_hold(self.strategy.initial_cash, train_size=0, trading_days=len(self.portfolio_series))